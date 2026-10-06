# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate matched coding evaluations without executing model-produced code."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def worker(args, protocol):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import nemo_rl  # noqa: F401

    rows = [json.loads(line) for line in args.data.read_text().splitlines()]
    tokenizer = AutoTokenizer.from_pretrained(protocol["base_model"])
    tokenizer.chat_template = (
        ROOT.parent / "tml_opd_replication/qwen3_historical.jinja"
    ).read_text()
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
        for r in rows
    ]
    if max(map(len, prompts)) >= protocol["max_total_tokens"]:
        raise ValueError("Evaluation prompt exceeds context budget")
    model = args.model_path or protocol[args.model + "_model"]
    engine = LLM(
        model=str(model),
        tokenizer=protocol["base_model"],
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=protocol["max_total_tokens"],
        max_num_seqs=16,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        enable_prefix_caching=True,
        generation_config="vllm",
        seed=protocol["seed"],
    )
    output = args.output / f"{args.model}-rank{args.rank}.jsonl"
    done = (
        {json.loads(x)["id"] for x in output.read_text().splitlines()}
        if output.exists()
        else set()
    )
    tasks = [
        (repeat, i)
        for repeat in range(protocol["evaluation_repeats"])
        for i in range(len(rows))
        if (repeat * len(rows) + i) % args.workers == args.rank
        and f"{repeat}:{rows[i]['id']}" not in done
    ]
    with output.open("a", buffering=1) as stream:
        for start in range(0, len(tasks), 16):
            batch = tasks[start : start + 16]
            params = [
                SamplingParams(
                    temperature=protocol["temperature"],
                    top_p=protocol["top_p"],
                    top_k=protocol["top_k"],
                    min_p=0.0,
                    max_tokens=protocol["max_total_tokens"] - len(prompts[i]),
                    seed=protocol["seed"] + repeat * len(rows) + i,
                    stop_token_ids=[151643, 151645],
                )
                for repeat, i in batch
            ]
            generated = engine.generate(
                [{"prompt_token_ids": prompts[i]} for _, i in batch],
                params,
                use_tqdm=False,
            )
            for (repeat, i), result in zip(batch, generated, strict=True):
                out = result.outputs[0]
                stream.write(
                    json.dumps(
                        dict(
                            id=f"{repeat}:{rows[i]['id']}",
                            problem_id=rows[i]["id"],
                            repeat=repeat,
                            text=out.text,
                            generated_tokens=len(out.token_ids),
                            prompt_tokens=len(prompts[i]),
                            finish_reason=out.finish_reason,
                        )
                    )
                    + "\n"
                )
            print(
                f"PROGRESS {args.model} rank={args.rank} samples={start + len(batch)}/{len(tasks)}",
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["student", "teacher", "opd"], required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/eval_candidates.jsonl"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--rank", type=int)
    args = parser.parse_args()
    protocol = json.loads((ROOT / "protocol.json").read_text())
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.rank is not None:
        worker(args, protocol)
        return
    import torch

    if torch.cuda.device_count() != args.workers:
        raise ValueError(
            f"Allocated {torch.cuda.device_count()} GPUs, requested {args.workers}"
        )
    (args.output / f"{args.model}-protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )
    command = [
        "uv",
        "run",
        "--no-project",
        "--python",
        sys.executable,
        "python",
        str(Path(__file__).resolve()),
        "--model",
        args.model,
        "--data",
        str(args.data.resolve()),
        "--output",
        str(args.output),
        "--workers",
        str(args.workers),
    ]
    if args.model_path:
        command += ["--model-path", str(args.model_path.resolve())]
    processes, logs = [], []
    try:
        for rank in range(args.workers):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank))
            env.pop("ROCR_VISIBLE_DEVICES", None)
            env.pop("HIP_VISIBLE_DEVICES", None)
            env["TRITON_CACHE_DIR"] = f"{os.environ['TRITON_CACHE_DIR']}/rank{rank}"
            log = (args.output / f"{args.model}-rank{rank}.log").open("a")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    command + ["--rank", str(rank)],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        while any(p.poll() is None for p in processes):
            failures = [
                (i, p.returncode)
                for i, p in enumerate(processes)
                if p.poll() not in (None, 0)
            ]
            if failures:
                raise RuntimeError(f"Generation worker failures: {failures}")
            counts = {
                p.name: sum(1 for _ in p.open())
                for p in args.output.glob(f"{args.model}-rank*.jsonl")
            }
            status = dict(time=time.time(), samples=counts)
            (args.output / f"{args.model}-status.json").write_text(
                json.dumps(status) + "\n"
            )
            print(json.dumps(status), flush=True)
            time.sleep(30)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError("Generation failed")
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            p.wait()
        for log in logs:
            log.close()
    print("GENERATION_COMPLETE", args.model, flush=True)


if __name__ == "__main__":
    main()
