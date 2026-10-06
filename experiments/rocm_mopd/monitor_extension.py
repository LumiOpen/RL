# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run both continuation arms, recover infrastructure failures, and evaluate them."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

from monitor_pilot import ACTIVE, REPO, ROOT, save

STAGES = ["train", "export", "evaluate", "grade"]
BASELINE = REPO / "experiments/coding_opd/run_eval_50305/graded.jsonl"
RETRY_STATES = {"TIMEOUT", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}
INFRA_ERRORS = (
    "ActorDiedError",
    "NodeDisconnectedError",
    "Connection reset by peer",
    "RCCL error",
    "NCCL error",
    "Connection timed out",
)


def read_json(path):
    return json.loads(path.read_text())


def validate_baseline():
    clean = {
        json.loads(line)["id"]
        for line in (REPO / "experiments/coding_opd/data/eval_clean.jsonl")
        .read_text()
        .splitlines()
    }
    rows = [
        row
        for row in map(json.loads, BASELINE.read_text().splitlines())
        if row["model"] == "student"
    ]
    expected = {(key, repeat) for key in clean for repeat in range(4)}
    assert (
        len(rows) == 700 and {(r["problem_id"], r["repeat"]) for r in rows} == expected
    )


def initialize(path, preflight):
    assert not path.exists(), "Continuation state already exists"
    validate_baseline()
    arms = {}
    for arm in ("routed", "control"):
        run = ROOT / f"run_extend50_{arm}"
        assert (
            read_json(run / "checkpoints/step_20/training_info.json")["current_step"]
            == 20
        )
        arms[arm] = {
            "run": str(run),
            "config": str(ROOT / f"extend_{arm}_50.yaml"),
            "gpus": 8 if arm == "routed" else 6,
            "stage": "train",
            "status": "ready",
            "jobs": {},
            "attempts": {},
            "retries": {},
            "retry_progress": {},
            "last_progress": 20,
        }
    state = {
        "target_step": 50,
        "status": "watching",
        "preflight": preflight,
        "preflight_passed": False,
        "arms": arms,
        "jobs": {arm: f"extend50_{arm}" for arm in arms},
        "report": {
            "status": "ready",
            "jobs": {},
            "attempts": {},
            "retries": {},
            "retry_progress": {},
        },
    }
    save(path, state)
    return state


def job_states(state):
    jobs = {state["preflight"]}
    for item in [*state["arms"].values(), state["report"]]:
        jobs.update(item["jobs"].values())
    result = subprocess.run(
        [
            "sacct",
            "-X",
            "-j",
            ",".join(jobs),
            "-n",
            "-P",
            "--format=JobID,State,ExitCode",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return {
        row[0]: (row[1].split()[0], row[2])
        for line in result.stdout.splitlines()
        if (row := line.split("|"))[0] in jobs
    }


def evaluation_dir(item, step):
    return Path(item["run"]) / f"evaluation_step_{step}"


def arguments(state, arm, stage, path):
    if stage == "report":
        return [
            str(ROOT / "report.sbatch"),
            str(path.resolve()),
            "--step",
            str(state["target_step"]),
        ], {}
    item = state["arms"][arm]
    run = Path(item["run"])
    step = state["target_step"]
    if stage == "train":
        return [
            f"--gpus-per-node={item['gpus']}",
            "--cpus-per-task=64",
            "--mem=512G",
            "--time=24:00:00",
            str(ROOT / "native_smoke.sbatch"),
        ], {
            "NRL_MOPD_RUN_ID": run.name.removeprefix("run_"),
            "NRL_MOPD_CONFIG": item["config"],
            "NRL_MOPD_ENTRYPOINT": str(ROOT / "audit_native.py"),
            "NRL_MOPD_RUN_LABEL": "mopd",
        }
    if stage == "export":
        return [
            str(ROOT / "export.sbatch"),
            "--run",
            str(run),
            "--step",
            str(step),
            "--require-probe",
            "--validate-merge",
        ], {}
    if stage == "evaluate":
        return [str(ROOT / "evaluate.sbatch"), str(run), str(step)], {}
    assert stage == "grade"
    return [
        str(REPO / "experiments/coding_opd/grade.sbatch"),
        "--input",
        str(evaluation_dir(item, step) / "eval_lcb"),
        "--models",
        "opd",
        "--baseline",
        str(BASELINE),
    ], {}


def submit(state, path, arm, stage, dependency=None):
    item = state["report"] if stage == "report" else state["arms"][arm]
    args, extra_env = arguments(state, arm, stage, path)
    item.update(status="submitting", submitting_stage=stage)
    save(path, state)
    dependency_args = (
        [f"--dependency=afterok:{dependency}", "--kill-on-invalid-dep=yes"]
        if dependency
        else []
    )
    result = subprocess.run(
        [
            "sbatch",
            "--parsable",
            "--exclude=tus1-p15-g5,tus1-p15-g62",
            f"--job-name=mopd50-{arm}-{stage}",
            *dependency_args,
            *args,
        ],
        cwd=REPO,
        env=dict(os.environ, **extra_env),
        capture_output=True,
        text=True,
        check=True,
        timeout=45,
    )
    job = result.stdout.strip().splitlines()[-1].split(";")[0]
    assert job.isdecimal(), result.stdout
    item["jobs"][stage] = job
    item["attempts"].setdefault(stage, []).append(job)
    item.update(status="watching", progress_job=job, progress_time=time.time())
    item.pop("submitting_stage", None)
    save(path, state)
    print("SUBMITTED", arm, stage, job, flush=True)
    return job


def queue_remaining(state, path, arm):
    item = state["arms"][arm]
    stage = item["stage"]
    if stage == "train":
        submit(state, path, arm, stage)
        return
    dependency = None
    for stage in STAGES[STAGES.index(stage) :]:
        dependency = submit(state, path, arm, stage, dependency)


def log_text(item, stage):
    job = item["jobs"][stage]
    log = (
        REPO / "experiments/coding_opd" if stage == "grade" else ROOT
    ) / f"{ {'train': 'native', 'evaluate': 'eval'}.get(stage, stage) }-{job}.log"
    if not log.exists():
        return ""
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 2_000_000))
        return stream.read().decode(errors="replace")


def checkpoint_step(item):
    return read_json(Path(item["run"]) / "checkpoints/latest_checkpoint_status.json")[
        "last_checkpoint_step"
    ]


def retryable(slurm_state, text, *, partial_timeout=False, cancelled_stall=False):
    if any(
        error in text
        for error in (
            "OutOfMemoryError",
            "out of memory",
            "AssertionError",
            "non-finite",
        )
    ):
        return False
    return (
        slurm_state in RETRY_STATES
        or partial_timeout
        or cancelled_stall
        or (slurm_state == "FAILED" and any(error in text for error in INFRA_ERRORS))
    )


def retry(state, path, arm, stage, progress):
    item = state["report"] if stage == "report" else state["arms"][arm]
    previous = item["retry_progress"].get(stage)
    count = item["retries"].get(stage, 0)
    count = 1 if previous is not None and progress > previous else count + 1
    if count > 3 or len(item["attempts"][stage]) >= 8:
        raise ValueError(f"{arm}/{stage}: exhausted retries without progress")
    item["retry_progress"][stage] = progress
    item["retries"][stage] = count
    if stage == "report":
        submit(state, path, arm, stage)
        return
    for later in STAGES[STAGES.index(stage) + 1 :]:
        job = item["jobs"].pop(later, None)
        if job:
            subprocess.run(["scancel", job], check=True, timeout=30)
    item.pop("cancelled_stall", None)
    queue_remaining(state, path, arm)


def validate_stage(item, stage, target):
    run = Path(item["run"])
    if stage == "train":
        step = checkpoint_step(item)
        info = read_json(run / f"checkpoints/step_{step}/training_info.json")
        assert info["current_step"] == info["total_steps"] == target
        assert info["consumed_samples"] == target * 32
        audit = read_json(run / "logs/correctness_audit.json")
        assert audit["passed"] and audit["teacher_probes_unchanged"]
        assert audit["http_prompt_tokens_match_training_template"]
    elif stage == "export":
        report = read_json(run / f"hf_step_{target}/export_validation.json")
        assert report["passed"] and report["native_probe_checked"]
    elif stage == "evaluate":
        evaluation = evaluation_dir(item, target)
        report = read_json(evaluation / f"aime_{run.name}/aime_summary.json")
        assert report["samples"] == 120
        rows = [
            json.loads(line)
            for p in (evaluation / "eval_lcb").glob("opd-rank*.jsonl")
            for line in p.read_text().splitlines()
        ]
        assert len(rows) == len({r["id"] for r in rows}) == 700
    else:
        report = read_json(evaluation_dir(item, target) / "eval_lcb/summary.json")
        assert report["problems"] == 175 and report["samples_per_problem"] == 4
        assert set(report["models"]) == {"opd", "student"}


def advance_arm(state, path, arm, rows):
    item = state["arms"][arm]
    if item["status"] in {"complete", "needs_attention"}:
        return
    if item["status"] == "submitting":
        raise ValueError("Interrupted submission; reconcile Slurm before retrying")
    stage = item["stage"]
    if stage not in item["jobs"]:
        queue_remaining(state, path, arm)
        return
    job = item["jobs"][stage]
    if job not in rows:
        return
    slurm, code = rows[job]
    item["slurm_state"] = slurm
    text = log_text(item, stage)
    progress = checkpoint_step(item)
    if stage == "train":
        steps = [int(s) for s in re.findall(r"train step (\d+)/", text)]
        current = max([progress, *steps])
        if (
            current > item["last_progress"]
            or item["progress_job"] != job
            or slurm != "RUNNING"
        ):
            item.update(
                last_progress=current, progress_job=job, progress_time=time.time()
            )
        if (
            slurm == "RUNNING"
            and time.time() - item["progress_time"] > 3 * 3600
            and not item.get("cancelled_stall")
        ):
            subprocess.run(["scancel", job], check=True, timeout=30)
            item["cancelled_stall"] = True
            save(path, state)
            return
    if slurm in ACTIVE or slurm == "REQUEUED":
        return
    partial_timeout = (
        stage == "train"
        and slurm == "COMPLETED"
        and progress < state["target_step"]
        and "Timeout has been reached, stopping training early" in text
    )
    if retryable(
        slurm,
        text,
        partial_timeout=partial_timeout,
        cancelled_stall=item.get("cancelled_stall", False),
    ):
        retry(state, path, arm, stage, progress)
        return
    if slurm != "COMPLETED" or code != "0:0":
        raise ValueError(f"{job}: {slurm}/{code}; inspect the {stage} log")
    validate_stage(item, stage, state["target_step"])
    if stage == "grade":
        item["status"] = "complete"
    else:
        item["stage"] = STAGES[STAGES.index(stage) + 1]
        if item["stage"] not in item["jobs"]:
            queue_remaining(state, path, arm)


def tick(state, path):
    rows = job_states(state)
    if not state["preflight_passed"]:
        status = rows.get(state["preflight"])
        if status is None or status[0] in ACTIVE:
            return
        assert status == ("COMPLETED", "0:0"), f"Resume preflight failed: {status}"
        run = ROOT / "run_extend_resume_smoke"
        assert (
            read_json(run / "checkpoints/step_3/training_info.json")["current_step"]
            == 3
        )
        assert read_json(run / "logs/correctness_audit.json")["passed"]
        state["preflight_passed"] = True
    for arm in state["arms"]:
        try:
            advance_arm(state, path, arm, rows)
        except (
            AssertionError,
            ValueError,
            KeyError,
            OSError,
            subprocess.SubprocessError,
        ) as error:
            state["arms"][arm].update(
                status="needs_attention", reason=f"{type(error).__name__}: {error}"
            )
    if all(item["status"] == "complete" for item in state["arms"].values()):
        report = state["report"]
        if not report["jobs"]:
            submit(state, path, "comparison", "report")
        else:
            status = rows.get(report["jobs"]["report"])
            if status == ("COMPLETED", "0:0"):
                assert (
                    ROOT / f"extension_{state['target_step']}_results.json"
                ).is_file()
                report["status"] = state["status"] = "complete"
            elif status and status[0] not in ACTIVE:
                if retryable(status[0], log_text(report, "report")):
                    retry(state, path, "comparison", "report", 0)
                else:
                    raise ValueError(f"Report failed: {status}")
    elif all(
        item["status"] in {"complete", "needs_attention"}
        for item in state["arms"].values()
    ):
        state["status"] = "needs_attention"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--initialize", metavar="PREFLIGHT_JOB")
    args = parser.parse_args()
    with args.state.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = (
            initialize(args.state, args.initialize)
            if args.initialize
            else read_json(args.state)
        )
        validate_baseline()
        while state["status"] == "watching":
            try:
                tick(state, args.state)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                if any(
                    item["status"] == "submitting"
                    for item in [*state["arms"].values(), state["report"]]
                ):
                    state.update(
                        status="needs_attention",
                        reason=f"Submission uncertain: {error}",
                    )
                else:
                    print("Slurm query failed; retrying:", error, flush=True)
            except (AssertionError, ValueError, KeyError, OSError) as error:
                state.update(
                    status="needs_attention", reason=f"{type(error).__name__}: {error}"
                )
            save(args.state, state)
            print(
                state["heartbeat_utc"],
                state["status"],
                {
                    arm: {
                        k: item.get(k)
                        for k in (
                            "stage",
                            "status",
                            "slurm_state",
                            "last_progress",
                            "reason",
                        )
                    }
                    for arm, item in state["arms"].items()
                },
                state.get("reason", ""),
                flush=True,
            )
            if state["status"] == "watching":
                time.sleep(30)
        if state["status"] != "complete":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
