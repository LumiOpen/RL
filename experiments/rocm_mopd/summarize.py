# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare native MOPD and its matched control by held-out problem."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics


def paired(after, before):
    assert after.keys() == before.keys()
    delta = [after[key] - before[key] for key in sorted(after)]
    rng = random.Random(20260929)
    draws = sorted(
        statistics.mean(rng.choices(delta, k=len(delta))) for _ in range(10000)
    )
    return {
        "difference": statistics.mean(delta),
        "paired_problem_bootstrap_95_ci": [draws[250], draws[9749]],
        "problems": len(delta),
    }


def summarize(state_path, step=20):
    state = json.loads(state_path.read_text())
    root = state_path.parent
    arms, lcb, aime = {}, {}, {}
    for arm in ["routed", "control"]:
        run = root / f"run_{state['jobs'][arm]}"
        evaluation = run if step == 20 else run / f"evaluation_step_{step}"
        coding = json.loads((evaluation / "eval_lcb/summary.json").read_text())
        assert coding["problems"] == 175 and coding["samples_per_problem"] == 4
        math_dir = evaluation / f"aime_{run.name}"
        math = json.loads((math_dir / "aime_summary.json").read_text())
        assert math["samples"] == 120
        arms[arm] = {
            "lcb": coding["models"]["opd"]["pass_at_1"],
            "aime": math["accuracy"],
            "run": str(run),
            "lcb_vs_sft": coding["opd_gain"],
            "aime_vs_sft": {
                "difference": math["accuracy_difference"],
                "paired_problem_bootstrap_95_ci": math[
                    "paired_problem_bootstrap_95_ci"
                ],
            },
        }
        lcb[arm] = coding["models"]["opd"]["per_problem_pass_at_1"]
        scores = defaultdict(list)
        for row in map(
            json.loads, (math_dir / "aime_graded.jsonl").read_text().splitlines()
        ):
            scores[row["problem_id"]].append(float(row["correct"]))
        assert len(scores) == 30 and all(len(values) == 4 for values in scores.values())
        aime[arm] = {key: statistics.mean(values) for key, values in scores.items()}
    report = {
        "training_steps": step,
        "arms": arms,
        "routed_minus_control": {
            "lcb": paired(lcb["routed"], lcb["control"]),
            "aime": paired(aime["routed"], aime["control"]),
        },
        "limitations": f"Single-seed, {step}-update native experiment; AIME has only four samples per problem. Fresh Megatron LoRA adapters differ from historical DTensor continuation.",
    }
    output = root / (
        "pilot_results.json" if step == 20 else f"extension_{step}_results.json"
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--step", type=int, default=20)
    args = parser.parse_args()
    report = summarize(args.state.resolve(), args.step)
    if args.wandb:
        import wandb

        state = json.loads(args.state.read_text())
        with wandb.init(
            entity="rahular",
            project="opd-tests",
            group="mopd-math-coding-pilot"
            if args.step == 20
            else f"mopd-math-coding-extension-{args.step}",
            id=f"mopd-comparison-{state['jobs']['routed']}-{state['jobs']['control']}",
            resume="allow",
            save_code=False,
            settings=wandb.Settings(disable_code=True),
        ) as run:
            run.summary.update(report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
