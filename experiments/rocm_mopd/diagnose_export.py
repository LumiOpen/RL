# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Separate native/HF numerical differences from adapter merge correctness."""

import json
from pathlib import Path

from safetensors.torch import load_file
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM


def diagnose(base_path, adapter_path, merged_path, probe):
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    adapters = load_file(str(adapter_path / "adapter_model.safetensors"))
    cfg = json.loads((adapter_path / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    pairs = {}
    for key, value in adapters.items():
        if key.endswith(".lora_A.weight"):
            name = key.removeprefix("base_model.model.").removesuffix(".lora_A.weight")
            pairs[name] = (
                value,
                adapters[key.replace(".lora_A.weight", ".lora_B.weight")],
            )
    assert len(pairs) * 2 == len(adapters) and pairs
    model = (
        AutoModelForCausalLM.from_pretrained(
            base_path, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )
    ids = torch.tensor(probe["input_ids"], device="cuda")
    lengths = torch.tensor(probe["input_lengths"], device="cuda")
    mask = torch.arange(ids.shape[1], device="cuda")[None, :] < lengths[:, None]
    reference = torch.tensor(probe["logprobs"], device="cuda")[:, 1:]

    def scores():
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask).logits.float()
            return (
                logits[:, :-1].log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
            )

    def errors(a, b):
        difference = (a - b)[mask[:, 1:]].abs()
        return {"max": difference.max().item(), "mean": difference.mean().item()}

    def attach(dtype):
        handles = []
        for name, (a, b) in pairs.items():
            a, b = a.to(device="cuda", dtype=dtype), b.to(device="cuda", dtype=dtype)

            def hook(module, inputs, output, a=a, b=b):
                return output + F.linear(F.linear(inputs[0], a), b) * scale

            handles.append(model.get_submodule(name).register_forward_hook(hook))
        return handles

    handles = attach(torch.bfloat16)
    unmerged_bf16 = scores()
    for h in handles:
        h.remove()
    model.float()
    handles = attach(torch.float32)
    unmerged_fp32 = scores()
    for h in handles:
        h.remove()

    index = json.loads(
        (Path(merged_path) / "model.safetensors.index.json").read_text()
    )["weight_map"]
    exported = {}
    for shard in sorted(set(index.values())):
        exported.update(load_file(str(Path(merged_path) / shard)))
    assert set(exported) == set(dict(model.named_parameters()))
    checked, mismatched, largest = 0, 0, 0.0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            module_name = name.removesuffix(".weight")
            expected = parameter.cpu()
            if module_name in pairs:
                a, b = pairs[module_name]
                expected = expected + (b.float() @ a.float()) * scale
                parameter.copy_(expected.cuda())
            actual = exported[name]
            rounded = expected.to(actual.dtype)
            checked += actual.numel()
            mismatched += (actual != rounded).sum().item()
            largest = max(
                largest, (actual.float() - rounded.float()).abs().max().item()
            )
    merged_fp32 = scores()
    fp32_equivalence = errors(unmerged_fp32, merged_fp32)
    model.bfloat16()
    merged_bf16 = scores()
    return {
        "adapter_modules": len(pairs),
        "weight_elements_checked": checked,
        "weight_elements_mismatched": mismatched,
        "weight_max_abs_error": largest,
        "unmerged_bf16_vs_native": errors(unmerged_bf16, reference),
        "merged_bf16_vs_native": errors(merged_bf16, reference),
        "merged_vs_unmerged_bf16": errors(merged_bf16, unmerged_bf16),
        "fp32_merge_equivalence": fp32_equivalence,
    }
