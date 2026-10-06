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

"""Publish evaluation progress and scalar results without uploading generated code."""

import argparse
import json
from pathlib import Path
import time

import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--name", required=True)
args = parser.parse_args()
protocol = json.loads((Path(__file__).resolve().parent / "protocol.json").read_text())
run = wandb.init(
    project=protocol["wandb_project"],
    group=protocol["wandb_group"],
    name=args.name,
    id=args.name,
    resume="allow",
    config=protocol,
    save_code=False,
    settings=wandb.Settings(disable_code=True),
)
(args.input / "wandb_url.txt").write_text(run.url + "\n")
for tick in range(300):
    metrics = {}
    for path in args.input.glob("*-status.json"):
        state = json.loads(path.read_text())
        if "samples" in state:
            metrics[path.stem.removesuffix("-status") + "/generated_samples"] = sum(
                state["samples"].values()
            )
        elif "graded" in state:
            metrics["evaluation/graded_samples"] = state["graded"]
    if metrics:
        run.log(metrics)
    summary = args.input / "summary.json"
    if summary.exists():
        report = json.loads(summary.read_text())
        for model, result in report["models"].items():
            for key, value in result.items():
                if isinstance(value, (float, int)):
                    run.summary[f"{model}/{key}"] = value
        for key in ("teacher_gap", "opd_gain", "comparison_to_4k"):
            if key in report:
                run.summary[key] = report[key]
        run.summary["evaluation_complete"] = True
        run.finish()
        break
    time.sleep(60)
else:
    run.summary["evaluation_complete"] = False
    run.finish(exit_code=1)
    raise TimeoutError("Evaluation tracker timed out after five hours")
