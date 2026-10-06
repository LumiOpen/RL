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

"""Prepare and score a 120-sample AIME check paired with the original SFT draws."""

import argparse
import json
from pathlib import Path

from experiments.tml_opd_replication.eval_aime import DATA, PROTOCOL, last_box

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parents[2] / "RL/experiments/tml_opd_replication/run_aime_49575"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prepare", "score"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.output / "aime_prompts.jsonl"
    if args.mode == "prepare":
        import pyarrow.parquet as pq

        rows = pq.read_table(DATA).to_pylist()
        assert len(rows) == 30
        data.write_text(
            "".join(
                json.dumps(
                    {
                        "id": f"aime:{i}",
                        "question": row["problem"],
                        "prompt": row["problem"] + PROTOCOL["instruction"],
                        "answer": str(row["answer"]),
                    }
                )
                + "\n"
                for i, row in enumerate(rows)
            )
        )
        return
    import numpy as np
    import wandb
    from math_verify import parse, verify

    questions = {
        row["id"]: row for row in map(json.loads, data.read_text().splitlines())
    }
    records = [
        json.loads(line)
        for path in args.output.glob("opd-rank*.jsonl")
        for line in path.read_text().splitlines()
    ]
    expected = {f"{repeat}:aime:{i}" for repeat in range(4) for i in range(30)}
    assert len(records) == 120 and {row["id"] for row in records} == expected
    scores = np.zeros((30, 4))
    graded = []
    for row in records:
        boxed = last_box(row["text"])
        correct = boxed is not None and bool(
            verify(
                parse(questions[row["problem_id"]]["answer"]), parse("$" + boxed + "$")
            )
        )
        i = int(row["problem_id"].split(":")[1])
        scores[i, row["repeat"]] = correct
        graded.append(
            {key: value for key, value in row.items() if key != "text"}
            | {"correct": correct, "boxed": boxed}
        )
    baseline_rows = [
        json.loads(line)
        for path in BASELINE.glob("sft-rank*.jsonl")
        for line in path.read_text().splitlines()
    ]
    baseline = {row["id"]: row for row in baseline_rows}
    reference = np.array(
        [
            [baseline[f"{repeat}:{i}"]["correct"] for repeat in range(4)]
            for i in range(30)
        ],
        dtype=float,
    )
    delta = scores.mean(axis=1) - reference.mean(axis=1)
    rng = np.random.default_rng(20260922)
    ci = np.quantile(
        delta[rng.integers(0, 30, (10000, 30))].mean(axis=1), [0.025, 0.975]
    ).tolist()
    report = {
        "samples": 120,
        "accuracy": float(scores.mean()),
        "matched_sft_accuracy": float(reference.mean()),
        "accuracy_difference": float(delta.mean()),
        "paired_problem_bootstrap_95_ci": ci,
        "limitations": "Small regression screen: 30 problems and four samples each. Reference uses the corresponding original SFT seeds.",
    }
    (args.output / "aime_graded.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in graded)
    )
    temporary = args.output / "aime_summary.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output / "aime_summary.json")
    name = "coding-aime-" + args.output.name.removeprefix("run_aime_")
    run = wandb.init(
        project="opd-tests",
        group="coding-opd",
        id=name,
        name=name,
        resume="allow",
        save_code=False,
        settings=wandb.Settings(disable_code=True),
    )
    run.summary.update(report)
    (args.output / "wandb_url.txt").write_text(run.url + "\n")
    run.finish()
    print("AIME_REGRESSION_COMPLETE", report, flush=True)


if __name__ == "__main__":
    main()
