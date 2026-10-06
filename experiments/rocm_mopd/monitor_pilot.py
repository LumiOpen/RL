# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Advance audited native preflight to the two matched learning arms."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
ACTIVE = {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "SUSPENDED"}
PHASES = [
    "preflight",
    "routed",
    "control",
    "export_routed",
    "evaluate_routed",
    "grade_routed",
    "export_control",
    "evaluate_control",
    "grade_control",
    "report",
]


def save(path, state):
    state["heartbeat_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(path)


def submit(path, state, phase, *, resume=False):
    env = dict(os.environ)
    env.pop("NRL_MOPD_RUN_ID", None)
    if phase in {"routed", "control"}:
        if resume:
            env["NRL_MOPD_RUN_ID"] = state["jobs"][phase]
        config = "pilot.yaml" if phase == "routed" else "pilot_control.yaml"
        gpus = 8 if phase == "routed" else 6
        env.update(
            NRL_MOPD_CONFIG=str(ROOT / config),
            NRL_MOPD_ENTRYPOINT=str(ROOT / "audit_native.py"),
            NRL_MOPD_RUN_LABEL=f"mopd-{phase}",
        )
        arguments = [
            f"--gpus-per-node={gpus}",
            "--cpus-per-task=64",
            "--mem=512G",
            "--time=12:00:00",
            f"--job-name=mopd-{phase}",
            str(ROOT / "native_smoke.sbatch"),
        ]
    elif phase == "report":
        arguments = [str(ROOT / "report.sbatch"), str(path.resolve())]
    else:
        action, arm = phase.split("_")
        run = ROOT / f"run_{state['jobs'][arm]}"
        if action == "export":
            arguments = [
                str(ROOT / "export.sbatch"),
                "--run",
                str(run),
                "--step",
                "20",
                "--require-probe",
                "--validate-merge",
            ]
        elif action == "evaluate":
            arguments = [str(ROOT / "evaluate.sbatch"), str(run)]
        elif action == "grade":
            arguments = [
                str(REPO / "experiments/coding_opd/grade.sbatch"),
                "--input",
                str(run / "eval_lcb"),
                "--models",
                "opd",
                "--baseline",
                str(REPO / "experiments/coding_opd/run_eval_50305/graded.jsonl"),
            ]
        else:
            raise ValueError(f"Unknown phase {phase}")
    state.update(status="submitting", next_phase=phase)
    save(path, state)
    result = subprocess.run(
        [
            "sbatch",
            "--parsable",
            "--exclude=tus1-p15-g5,tus1-p15-g62",
            *arguments,
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=45,
    )
    job = result.stdout.strip().splitlines()[-1].split(";")[0]
    if not job.isdecimal():
        raise ValueError(f"Ambiguous Slurm submission: {result.stdout!r}")
    state.update(status="watching", phase=phase, job=job)
    if not resume:
        state["jobs"][phase] = job
    state.setdefault("attempts", {}).setdefault(phase, []).append(job)
    state.pop("next_phase", None)
    state.pop("reason", None)


def launch_evaluations(path, state, export_jobs):
    assert "evaluation_chain" not in state, "Evaluation chain already submitted"
    chain = state["evaluation_chain"] = {}

    def enqueue(phase, dependencies, arguments):
        state.update(status="submitting", next_phase=phase)
        save(path, state)
        result = subprocess.run(
            [
                "sbatch",
                "--parsable",
                "--kill-on-invalid-dep=yes",
                "--exclude=tus1-p15-g5,tus1-p15-g62",
                "--dependency=afterok:" + ":".join(dependencies),
                *map(str, arguments),
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
            timeout=45,
        )
        job = result.stdout.strip().splitlines()[-1].split(";")[0]
        assert job.isdecimal(), result.stdout
        chain[phase] = state["jobs"][phase] = job
        state.setdefault("attempts", {}).setdefault(phase, []).append(job)
        save(path, state)
        return job

    grades = []
    for arm, export in zip(("routed", "control"), export_jobs, strict=True):
        chain[f"export_{arm}"] = state["jobs"][f"export_{arm}"] = export
        run = ROOT / f"run_{state['jobs'][arm]}"
        evaluation = enqueue(
            f"evaluate_{arm}", [export], [ROOT / "evaluate.sbatch", run]
        )
        grades.append(
            enqueue(
                f"grade_{arm}",
                [evaluation],
                [
                    REPO / "experiments/coding_opd/grade.sbatch",
                    "--input",
                    run / "eval_lcb",
                    "--models",
                    "opd",
                    "--baseline",
                    REPO / "experiments/coding_opd/run_eval_50305/graded.jsonl",
                ],
            )
        )
    report = enqueue("report", grades, [ROOT / "report.sbatch", path.resolve()])
    state.update(status="watching", phase="parallel_evaluation", job=report)
    state.pop("reason", None)
    state.pop("next_phase", None)
    save(path, state)


def advance_evaluations(state):
    chain = state["evaluation_chain"]
    result = subprocess.run(
        [
            "sacct",
            "-X",
            "-j",
            ",".join(chain.values()),
            "-n",
            "-P",
            "--format=JobID,State,ExitCode",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    rows = {
        row[0]: row[1:]
        for line in result.stdout.splitlines()
        if (row := line.split("|"))[0] in chain.values()
    }
    state["evaluation_states"] = {
        phase: rows.get(job, ["UNKNOWN"])[0] for phase, job in chain.items()
    }
    failures = {
        phase: rows[job]
        for phase, job in chain.items()
        if job in rows
        and rows[job][0] not in ACTIVE
        and rows[job] != ["COMPLETED", "0:0"]
    }
    if failures:
        state.update(
            status="needs_attention", reason=f"Evaluation failures: {failures}"
        )
    elif all(rows.get(job) == ["COMPLETED", "0:0"] for job in chain.values()):
        assert (ROOT / "pilot_results.json").is_file()
        state.update(status="complete")


def advance(path, state):
    if state["phase"] == "parallel_evaluation":
        advance_evaluations(state)
        return
    job = state["job"]
    result = subprocess.run(
        ["sacct", "-X", "-j", job, "-n", "-P", "--format=JobID,State,ExitCode"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    row = next(
        (r.split("|") for r in result.stdout.splitlines() if r.split("|")[0] == job),
        None,
    )
    if row is None:
        return
    state["slurm_state"] = row[1]
    if row[1] in ACTIVE:
        return
    phase = state["phase"]
    recoverable_timeout = row[1] == "TIMEOUT" and phase in {"routed", "control"}
    if not recoverable_timeout and (row[1] != "COMPLETED" or row[2] != "0:0"):
        state.update(status="needs_attention", reason=f"{job}: {row[1]} / {row[2]}")
        return
    if phase in {"preflight", "routed", "control"}:
        run = ROOT / f"run_{state['jobs'][phase]}"
        checkpoint_root = run / "checkpoints"
        latest = json.loads(
            (checkpoint_root / "latest_checkpoint_status.json").read_text()
        )
        saved_step = latest["last_checkpoint_step"]
        info = json.loads(
            (checkpoint_root / f"step_{saved_step}/training_info.json").read_text()
        )
        assert info["current_step"] == saved_step
        step = 2 if phase == "preflight" else 20
        if saved_step < step:
            log = (ROOT / f"native-{job}.log").read_text()
            graceful_timeout = (
                "Timeout has been reached, stopping training early" in log
            )
            if phase == "preflight" or not (recoverable_timeout or graceful_timeout):
                raise ValueError(
                    f"Unexpected early completion at step {saved_step}/{step}"
                )
            previous_step = state.setdefault("resume_from_step", {}).get(phase, 0)
            retries = state.setdefault("resume_count", {}).get(phase, 0)
            if saved_step <= previous_step or retries >= 3:
                raise ValueError(
                    f"Resume made no checkpoint progress or exhausted retries: step {saved_step}"
                )
            if graceful_timeout:
                audit = json.loads((run / "logs/correctness_audit.json").read_text())
                assert audit["passed"] and audit["teacher_probes_unchanged"]
            state["resume_from_step"][phase] = saved_step
            state["resume_count"][phase] = retries + 1
            submit(path, state, phase, resume=True)
            return
        audit = json.loads((run / "logs/correctness_audit.json").read_text())
        assert audit["passed"] and audit["teacher_probes_unchanged"]
        assert audit["http_prompt_tokens_match_training_template"]
        assert info["current_step"] == step
        state.setdefault("results", {})[phase] = {
            "audit": audit,
            "training_info": info,
            "run": str(run),
        }
    elif phase == "report":
        assert (ROOT / "pilot_results.json").is_file()
        state.update(status="complete")
        return
    else:
        action, arm = phase.split("_")
        run = ROOT / f"run_{state['jobs'][arm]}"
        if action == "export":
            report = json.loads((run / "hf_step_20/export_validation.json").read_text())
            assert report["passed"] and report["native_probe_checked"]
        elif action == "evaluate":
            report = json.loads(
                (run / f"aime_{run.name}/aime_summary.json").read_text()
            )
            assert report["samples"] == 120
        elif action == "grade":
            report = json.loads((run / "eval_lcb/summary.json").read_text())
            assert report["problems"] == 175 and report["samples_per_problem"] == 4
    submit(path, state, PHASES[PHASES.index(phase) + 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--export-jobs", nargs=2, metavar=("ROUTED", "CONTROL"))
    args = parser.parse_args()
    with args.state.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(args.state.read_text())
        if args.export_jobs:
            launch_evaluations(args.state, state, args.export_jobs)
        while state["status"] == "watching":
            try:
                advance(args.state, state)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                if state["status"] == "submitting":
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
                state["phase"],
                state["status"],
                state.get("reason", ""),
                flush=True,
            )
            if state["status"] == "watching":
                time.sleep(30)


if __name__ == "__main__":
    main()
