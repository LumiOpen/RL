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

"""Watch the coding pilot, resume infrastructure failures, and evaluate complete checkpoints."""

import argparse
import fcntl
import getpass
import json
import os
from pathlib import Path
import subprocess
import time

from experiments.tml_opd_replication.monitor_opd import (
    ACTIVE,
    RECOVERABLE,
    completed_step,
    infrastructure_failure,
    latest_checkpoint,
)

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def persist(path, state):
    state["heartbeat_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def command(args):
    return subprocess.run(
        args, cwd=REPO, check=True, text=True, capture_output=True, timeout=45
    ).stdout


def states(jobs):
    output = command(
        [
            "sacct",
            "-X",
            "-j",
            ",".join(map(str, jobs)),
            "-n",
            "-P",
            "--format=JobID,State,User%80,JobName%80",
        ]
    )
    result = {}
    for line in output.splitlines():
        job, status, user, name = line.split("|")[:4]
        if job not in {str(j) for j in jobs}:
            continue
        if user != getpass.getuser() or not name.startswith("coding-"):
            raise ValueError(f"Unexpected managed job {job}: {user}/{name}")
        result[job] = status.split()[0].rstrip("+")
    return result


def submit(state, path, key, args):
    state["status"] = "submitting"
    state["submission_key"] = key
    persist(path, state)
    # Keep an uncertain submission marked for inspection, never retry it blindly.
    output = command(args).strip()
    if not output.isdecimal():
        raise ValueError(f"Unexpected submission output: {output}")
    state[key] = output
    persist(path, state)
    print("SUBMITTED", key, output, flush=True)
    return output


def launch_training(state, path):
    os.environ["TML_OPD_RUN_ID"] = state["run_id"]
    if "training_config" in state:
        os.environ["TML_OPD_CONFIG"] = state["training_config"]
    args = [
        "bash",
        str(ROOT / "launch_pilot.sh"),
        f"distillation.max_num_steps={state['target_steps']}",
    ]
    job = submit(state, path, "training_job", args)
    state["training_jobs"].append(job)
    state["phase"] = "training"
    state["progress_at"] = time.time()
    state["completed_step"] = 0
    state["running_since"] = None
    state["status"] = "watching"
    persist(path, state)


def launch_evaluation(state, path):
    step = state["target_steps"]
    adapter = (
        REPO
        / f"experiments/tml_opd_replication/run_opd_{state['run_id']}/checkpoints/step_{step}/policy/weights/model"
    )
    os.environ["CODING_ADAPTER"] = str(adapter)
    job = submit(
        state,
        path,
        "evaluation_job",
        ["sbatch", "--parsable", str(ROOT / "evaluate_opd.sbatch")],
    )
    output = ROOT / f"run_eval_{job}"
    output.mkdir(exist_ok=True)
    state["evaluation_output"] = str(output)
    comparison_args = (
        ["--comparison-summary", state["comparison_summary"]]
        if "comparison_summary" in state
        else []
    )
    submit(
        state,
        path,
        "grading_job",
        [
            "sbatch",
            "--parsable",
            str(ROOT / "grade.sbatch"),
            "--input",
            str(output),
            "--models",
            "opd",
            "--baseline",
            str(Path(state["baseline_output"]) / "graded.jsonl"),
            *comparison_args,
        ],
    )
    interpreter = (
        "/shared_silo/scratch/rahul.aralikatte@amd.com/prime-rl/.venv/bin/python"
    )
    with (output / "tracker.log").open("a") as log:
        process = subprocess.Popen(
            [
                "bash",
                "experiments/tml_opd_replication/container.sh",
                "uv",
                "run",
                "--no-project",
                "--python",
                interpreter,
                "python",
                str(ROOT / "track.py"),
                "--input",
                str(output),
                "--name",
                f"{state['run_id']}-eval-{step}",
            ],
            cwd=REPO,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    state["tracker_pid"] = process.pid
    state["phase"] = "evaluation"
    state["status"] = "watching"
    persist(path, state)


def tick(state, path):
    if state["phase"] == "preflight":
        job = state["preflight_job"]
        statuses = states([job])
        state["slurm_states"] = statuses
        status = statuses.get(job, "UNKNOWN")
        if status == "COMPLETED":
            run = REPO / f"experiments/tml_opd_replication/run_opd_{job}"
            _, step = latest_checkpoint(run / "checkpoints")
            log = (REPO / f"experiments/tml_opd_replication/opd-{job}.log").read_text()
            if step != 1 or "OPD_PARAMETER_AUDIT_PASSED" not in log:
                raise ValueError(
                    "Preflight did not complete its update and parameter audit"
                )
            audit_path = run / "logs/parameter_audit.json"
            audit = json.loads(audit_path.read_text())
            for before, after in zip(audit["before"], audit["after"], strict=True):
                assert before["frozen_sha256"] == after["frozen_sha256"]
                assert before["adapter_sha256"] != after["adapter_sha256"]
            state["preflight_audit"] = str(audit_path)
            launch_training(state, path)
        elif status not in ACTIVE | {"UNKNOWN"}:
            state["status"] = "needs_attention"
            state["reason"] = (
                f"Long-rollout preflight ended {status}; inspect before training"
            )
    elif state["phase"] == "baseline":
        summary = Path(state["baseline_output"]) / "summary.json"
        if summary.exists():
            report = json.loads(summary.read_text())
            state["baseline_results"] = report["teacher_gap"]
            if not report["teacher_gap"]["qualifies_for_pilot"]:
                state["status"] = "needs_attention"
                state["reason"] = (
                    "8B teacher lacks predeclared gap; evaluate the 32B teacher before training"
                )
                return
            launch_training(state, path)
            return
        statuses = states(state["baseline_jobs"])
        state["slurm_states"] = statuses
        if any(value not in ACTIVE | {"COMPLETED"} for value in statuses.values()):
            state["status"] = "needs_attention"
            state["reason"] = (
                "Baseline generation or grading failed; inspect and resume completed samples"
            )
    elif state["phase"] == "training":
        job = state["training_job"]
        statuses = states([job])
        status = statuses.get(job, "UNKNOWN")
        state["slurm_states"] = statuses
        log_path = REPO / f"experiments/tml_opd_replication/opd-{job}.log"
        log = log_path.read_text() if log_path.exists() else ""
        progress = completed_step(log)
        if progress is not None and progress != state["completed_step"]:
            state["completed_step"] = progress
            state["progress_at"] = time.time()
        if status == "RUNNING":
            if state["running_since"] is None:
                state["running_since"] = time.time()
                state["progress_at"] = time.time()
            if time.time() - state["progress_at"] > 3600:
                state["status"] = "needs_attention"
                state["reason"] = (
                    "No completed training update in one hour; inspect before canceling"
                )
        elif status == "COMPLETED":
            _, step = latest_checkpoint(
                REPO
                / f"experiments/tml_opd_replication/run_opd_{state['run_id']}/checkpoints"
            )
            if step != state["target_steps"] or "OPD_PARAMETER_AUDIT_PASSED" not in log:
                raise ValueError(
                    f"Training ended without complete checkpoint/audit: {step}"
                )
            run = REPO / f"experiments/tml_opd_replication/run_opd_{state['run_id']}"
            audit = json.loads((run / "logs/parameter_audit.json").read_text())
            (run / f"parameter_audit_step{step}.json").write_text(
                json.dumps(audit, indent=2) + "\n"
            )
            launch_evaluation(state, path)
        elif status not in ACTIVE | {"UNKNOWN"}:
            recoverable = status in RECOVERABLE or (
                status == "FAILED" and infrastructure_failure(log)
            )
            if not recoverable or state["restarts"] >= 3:
                state["status"] = "needs_attention"
                state["reason"] = f"Training ended {status}; automatic retry stopped"
            else:
                checkpoint, step = latest_checkpoint(
                    REPO
                    / f"experiments/tml_opd_replication/run_opd_{state['run_id']}/checkpoints"
                )
                state["resumed_checkpoint"] = str(checkpoint)
                state["restarts"] += 1
                launch_training(state, path)
    elif state["phase"] == "evaluation":
        summary = Path(state["evaluation_output"]) / "summary.json"
        if summary.exists():
            report = json.loads(summary.read_text())
            step = state["target_steps"]
            state["evaluations"][str(step)] = {
                "output": state["evaluation_output"],
                "gain": report["opd_gain"],
            }
            if step == 20 and report["opd_gain"]["qualifies_for_extension"]:
                state["target_steps"] = 100
                state["restarts"] = 0
                launch_training(state, path)
            else:
                merged = (
                    REPO
                    / f"experiments/tml_opd_replication/checkpoints/inference_run_opd_{state['run_id']}_step_{step}"
                )
                os.environ["CODING_MERGED"] = str(merged)
                job = submit(
                    state,
                    path,
                    "aime_job",
                    ["sbatch", "--parsable", str(ROOT / "aime.sbatch")],
                )
                state["aime_output"] = str(ROOT / f"run_aime_{job}")
                state["phase"] = "aime"
                state["status"] = "watching"
                persist(path, state)
            return
        statuses = states([state["evaluation_job"], state["grading_job"]])
        state["slurm_states"] = statuses
        if any(value not in ACTIVE | {"COMPLETED"} for value in statuses.values()):
            state["status"] = "needs_attention"
            state["reason"] = "Checkpoint evaluation/grading failed"
    elif state["phase"] == "aime":
        summary = Path(state["aime_output"]) / "aime_summary.json"
        if summary.exists():
            state["aime_results"] = json.loads(summary.read_text())
            assert state["aime_results"]["samples"] == 120
            state["status"] = "complete"
            state["reason"] = (
                "Coding evaluation and AIME regression check finished; inspect results before follow-up controls"
            )
            return
        statuses = states([state["aime_job"]])
        state["slurm_states"] = statuses
        if any(value not in ACTIVE | {"COMPLETED"} for value in statuses.values()):
            state["status"] = "needs_attention"
            state["reason"] = "AIME regression evaluation failed"
    else:
        raise ValueError(f"Unknown monitor phase: {state['phase']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    with args.state.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(args.state.read_text())
        if state["status"] != "watching":
            raise ValueError(f"Monitor requires inspection: {state['status']}")
        while state["status"] == "watching":
            try:
                tick(state, args.state)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                if state["status"] == "submitting":
                    state["reason"] = (
                        "Submission uncertain; inspect Slurm before restarting"
                    )
                    persist(args.state, state)
                    raise
                print("Transient Slurm query failure", str(error), flush=True)
            persist(args.state, state)
            print(
                state["heartbeat_utc"],
                state["phase"],
                state["status"],
                state.get("slurm_states"),
                state.get("reason", ""),
                flush=True,
            )
            if args.once:
                break
            time.sleep(30)


if __name__ == "__main__":
    main()
