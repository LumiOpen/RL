# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export a native LoRA checkpoint and check fixed probes when recorded."""

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--require-probe", action="store_true")
    parser.add_argument("--reuse-export", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--validate-merge", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    checkpoint = run / f"checkpoints/step_{args.step}"
    config = yaml.safe_load((checkpoint / "config.yaml").read_text())
    model_name = config["policy"]["model_name"]
    adapter = checkpoint / "policy/weights/iter_0000000"
    native_config = yaml.safe_load((adapter / "run_config.yaml").read_text())
    base = Path(native_config["checkpoint"]["pretrained_checkpoint"])
    if not (base / "run_config.yaml").exists():
        base /= "iter_0000000"
    assert (base / "run_config.yaml").is_file()
    output = run / f"hf_step_{args.step}"
    entry_path = (
        Path(__file__).resolve().parents[2]
        / "examples/converters/convert_lora_to_hf.py"
    )
    spec = importlib.util.spec_from_file_location("native_lora_export", entry_path)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    if not args.reuse_export:
        converter.merge_lora_to_hf(
            base_ckpt=str(base),
            adapter_ckpt=str(adapter),
            hf_model_name=model_name,
            hf_ckpt_path=str(output),
        )
    else:
        assert (output / "model.safetensors.index.json").is_file()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output)
    audit = json.loads((run / "logs/correctness_audit.json").read_text())
    probe = audit.get("student_probe")
    if args.require_probe:
        assert probe is not None, "Missing native checkpoint probe"
    report = {
        "source_checkpoint": str(checkpoint),
        "base": str(base),
        "output": str(output),
    }
    if args.diagnose or args.validate_merge:
        from diagnose_export import diagnose

        hf_adapter = run / f"hf_adapter_step_{args.step}"
        if not hf_adapter.exists():
            converter.export_lora_adapter_to_hf(
                base_ckpt=str(base),
                adapter_ckpt=str(adapter),
                hf_model_name=model_name,
                hf_ckpt_path=str(hf_adapter),
            )
        report.update(diagnose(model_name, hf_adapter, output, probe))
        (output / "export_diagnosis.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print("EXPORT_DIAGNOSIS", report, flush=True)
        if args.validate_merge:
            assert report["weight_elements_checked"] > 0, report
            assert report["weight_elements_mismatched"] == 0, report
            assert report["weight_max_abs_error"] == 0, report
            assert report["fp32_merge_equivalence"]["max"] < 1e-4, report
            assert report["unmerged_bf16_vs_native"]["max"] < 0.3, report
            assert report["merged_bf16_vs_native"]["max"] < 0.3, report
            report.update(
                passed=True,
                native_probe_checked=True,
                validation_method="exact_weights_and_fp32_merge_equivalence",
            )
            (output / "export_validation.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print("NATIVE_EXPORT_PASSED", report, flush=True)
        return
    if probe is not None:
        model = (
            AutoModelForCausalLM.from_pretrained(
                output,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            .cuda()
            .eval()
        )
        ids = torch.tensor(probe["input_ids"], device="cuda")
        lengths = torch.tensor(probe["input_lengths"], device="cuda")
        mask = torch.arange(ids.shape[1], device="cuda")[None, :] < lengths[:, None]
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits.float()
            logprobs = (
                logits[:, :-1].log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
            )
        reference = torch.tensor(probe["logprobs"], device="cuda")[:, 1:]
        errors = (logprobs - reference)[mask[:, 1:]].abs()
        report.update(
            max_abs_logprob_error=errors.max().item(),
            mean_abs_logprob_error=errors.mean().item(),
        )
        assert report["max_abs_logprob_error"] < 0.3, report
        assert report["mean_abs_logprob_error"] < 0.05, report
    report.update(passed=True, native_probe_checked=probe is not None)
    (output / "export_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("NATIVE_EXPORT_PASSED", report, flush=True)


if __name__ == "__main__":
    main()
