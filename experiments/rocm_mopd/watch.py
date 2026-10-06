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

"""Advance native ROCm backend gates and retain their Slurm evidence."""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"}


def save(path, state):
    state["heartbeat_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def advance(path, state):
    job = state["job"]
    result = subprocess.run(
        ["sacct", "-X", "-j", job, "-n", "-P", "--format=JobID,State,ExitCode"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    rows = [line.split("|") for line in result.stdout.splitlines()]
    row = next((row for row in rows if row[0] == job), None)
    if row is None:
        return
    state["slurm_state"] = row[1]
    if row[1] in ACTIVE:
        return
    if row[1] != "COMPLETED" or row[2] != "0:0":
        state.update(status="needs_attention", reason=f"{job}: {row[1]} / {row[2]}")
        return
    if state["phase"] == "imports":
        log = (ROOT / f"probe-{job}.log").read_text()
        required = [
            "ROCM_BACKWARD_PASSED",
            "NATIVE_TEACHER_IMPORTED",
            "GYM_INTEGRATION_IMPORTED",
        ]
        if not all(marker in log for marker in required):
            raise ValueError("Completed import probe is missing required pass markers")
        state["status"] = "submitting"
        save(path, state)
        submission = (
            subprocess.run(
                [
                    "sbatch",
                    "--parsable",
                    "--exclude=tus1-p15-g5,tus1-p15-g62",
                    str(ROOT / "backend_sft.sbatch"),
                ],
                cwd=REPO,
                check=True,
                capture_output=True,
                text=True,
                timeout=45,
            )
            .stdout.strip()
            .splitlines()[-1]
            .split(";")[0]
        )
        if not submission.isdecimal():
            raise ValueError(f"Ambiguous Slurm submission: {submission!r}")
        state.update(job=submission, phase="backend_sft", status="watching")
        state["jobs"].append(submission)
    elif state["phase"] == "native_smoke":
        checkpoint_root = ROOT / f"run_{job}/checkpoints"
        checkpoints = sorted(str(p) for p in checkpoint_root.glob("step_*"))
        if not (checkpoint_root / "step_2").is_dir():
            raise ValueError(
                "Native smoke exited successfully without step_2 checkpoint"
            )
        state.update(
            status="needs_attention",
            reason="Native MOPD smoke finished; inspect routing, numerical metrics, and checkpoint state before the accuracy pilot.",
            checkpoint_paths=checkpoints,
        )
    else:
        checkpoints = sorted(
            str(p) for p in (ROOT / f"run_{job}/checkpoints").glob("**/iter_*")
        )
        state.update(
            status="needs_attention",
            reason="Native backend SFT finished; inspect checkpoints before the two-teacher MOPD gate. This is not a MOPD success result.",
            checkpoint_paths=checkpoints,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    with args.state.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(args.state.read_text())
        while state["status"] == "watching":
            try:
                advance(args.state, state)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                if state["status"] == "submitting":
                    state.update(
                        status="needs_attention",
                        reason=f"Submission uncertain; inspect Slurm before retry: {error}",
                    )
                else:
                    print("Transient Slurm query failure:", error, flush=True)
            except (ValueError, OSError) as error:
                state.update(status="needs_attention", reason=str(error))
            save(args.state, state)
            print(
                state["heartbeat_utc"],
                state["phase"],
                state["status"],
                state.get("reason", ""),
                flush=True,
            )
            if state["status"] == "watching":
                time.sleep(30)


if __name__ == "__main__":
    main()
