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

"""Grade generated Python only inside a separate network/filesystem/PID namespace."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import time
import zlib

ROOT = Path(__file__).resolve().parent


class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError("Executable pickle content is not accepted")


def load_tests():
    result = {}
    for line in (ROOT / "data/lcb/test6.jsonl").open():
        row = json.loads(line)
        private = row["private_test_cases"]
        if private.lstrip().startswith("["):
            private = json.loads(private)
        else:
            decoded = zlib.decompress(base64.b64decode(private))
            private = json.loads(DataOnlyUnpickler(io.BytesIO(decoded)).load())
        tests = json.loads(row["public_test_cases"]) + private
        metadata = json.loads(row["metadata"])
        result[f"{row['platform']}:{row['question_id']}"] = {
            "inputs": [t["input"] for t in tests],
            "outputs": [t["output"] for t in tests],
            "fn_name": metadata.get("func_name"),
        }
    return result


def extract_code(text):
    lines = text.split("\n")
    fences = [i for i, line in enumerate(lines) if "```" in line]
    return "\n".join(lines[fences[-2] + 1 : fences[-1]]) if len(fences) >= 2 else ""


def run_sandbox(code, tests):
    timeout = 7 * len(tests["inputs"]) + 5
    command = [
        "prlimit",
        "--as=4294967296",
        "--fsize=16777216",
        "--",
        "singularity",
        "exec",
        "--userns",
        "--containall",
        "--cleanenv",
        "--net",
        "--network",
        "none",
        "--no-mount",
        "hostfs,bind-paths",
        "--no-home",
        "--pwd",
        "/app",
        str(ROOT / "data/grader.sif"),
        "/usr/local/bin/uv",
        "run",
        "--no-sync",
        "--project",
        "/app",
        "python",
        "/app/grade_one.py",
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
    )
    try:
        stdout, stderr = process.communicate(
            json.dumps({"code": code, "tests": tests}), timeout=timeout
        )
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        return {"passed": False, "results": [], "error_code": "global_timeout"}
    if "FATAL:" in stderr or "Traceback (most recent call last)" in stderr:
        raise RuntimeError(
            f"Sandbox/checker failed ({process.returncode}): {stderr[-4000:]}"
        )
    if process.returncode or not stdout.strip():
        return {
            "passed": False,
            "results": [],
            "error_code": "candidate_process_exit",
            "exit_code": process.returncode,
        }
    return json.loads(stdout)


def self_test():
    stdio = {"inputs": ["3\n", "5\n"], "outputs": ["6\n", "10\n"], "fn_name": None}
    cases = [
        ("stdio_correct", "print(int(input()) * 2)", stdio, True),
        ("wrong", "print(7)", stdio, False),
        ("syntax", "def broken(:", stdio, False),
        ("timeout", "while True: pass", stdio, False),
        (
            "function_correct",
            "class Solution:\n def twice(self, n): return n*2",
            {"inputs": ["3", "5"], "outputs": ["6", "10"], "fn_name": "twice"},
            True,
        ),
    ]
    for name, code, tests, expected in cases:
        result = run_sandbox(code, tests)
        print(name, result, flush=True)
        assert result["passed"] is expected, (name, result)
    print("GRADER_SELF_TEST_PASSED", flush=True)


def grade_one(model, row, tests):
    code = extract_code(row["text"])
    result = (
        run_sandbox(code, tests[row["problem_id"]])
        if code.strip()
        else {"passed": False, "results": [], "error_code": "no_code"}
    )
    return {
        "model": model,
        **{key: value for key, value in row.items() if key != "text"},
        **result,
    }


def summarize(records, models, repeats):
    import numpy as np

    clean = [
        json.loads(line)["id"]
        for line in (ROOT / "data/eval_clean.jsonl").read_text().splitlines()
    ]
    rng = np.random.default_rng(20260922)
    bootstrap = rng.integers(0, len(clean), size=(10000, len(clean)))
    means, report = (
        {},
        {"problems": len(clean), "samples_per_problem": repeats, "models": {}},
    )
    for model in models:
        selected = {
            (r["problem_id"], r["repeat"]): r for r in records if r["model"] == model
        }
        values = np.array(
            [
                [selected[(key, repeat)]["passed"] for repeat in range(repeats)]
                for key in clean
            ],
            dtype=float,
        )
        means[model] = values.mean(axis=1)
        rows = [selected[(key, repeat)] for key in clean for repeat in range(repeats)]
        report["models"][model] = {
            "pass_at_1": float(values.mean()),
            "passed": int(values.sum()),
            "samples": int(values.size),
            "problem_bootstrap_95_ci": np.quantile(
                means[model][bootstrap].mean(axis=1), [0.025, 0.975]
            ).tolist(),
            "truncation_rate": sum(r["finish_reason"] == "length" for r in rows)
            / len(rows),
            "mean_generated_tokens": float(
                np.mean([r["generated_tokens"] for r in rows])
            ),
            "error_counts": {
                str(k): sum(r.get("error_code") == k for r in rows)
                for k in {r.get("error_code") for r in rows}
            },
            "per_problem_pass_at_1": dict(
                zip(clean, means[model].tolist(), strict=True)
            ),
        }
    if {"student", "teacher"} <= set(models):
        delta = means["teacher"] - means["student"]
        ci = np.quantile(delta[bootstrap].mean(axis=1), [0.025, 0.975]).tolist()
        report["teacher_gap"] = {
            "pass_at_1_difference": float(delta.mean()),
            "paired_problem_bootstrap_95_ci": ci,
            "qualifies_for_pilot": bool(delta.mean() >= 0.05 and ci[0] > 0),
        }
    if {"student", "opd"} <= set(models):
        delta = means["opd"] - means["student"]
        ci = np.quantile(delta[bootstrap].mean(axis=1), [0.025, 0.975]).tolist()
        report["opd_gain"] = {
            "pass_at_1_difference": float(delta.mean()),
            "paired_problem_bootstrap_95_ci": ci,
            "qualifies_for_extension": bool(delta.mean() >= 0.02 and ci[0] > 0),
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--models", nargs="+", default=["student", "teacher"])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--comparison-summary", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.input:
        parser.error("--input is required for grading")
    tests = load_tests()
    protocol = json.loads((ROOT / "protocol.json").read_text())
    repeats = protocol["evaluation_repeats"]
    baseline_records = []
    if args.baseline:
        if "student" in args.models:
            raise ValueError(
                "A baseline cannot be added when student is already graded"
            )
        baseline_records = [
            row
            for row in map(json.loads, args.baseline.read_text().splitlines())
            if row["model"] == "student"
        ]
        baseline_keys = {(row["problem_id"], row["repeat"]) for row in baseline_records}
        required_keys = {(key, repeat) for key in tests for repeat in range(repeats)}
        if baseline_keys != required_keys or len(baseline_keys) != len(
            baseline_records
        ):
            raise ValueError(
                f"Baseline {args.baseline} must contain exactly {len(required_keys)} "
                f"matched student records; found {len(baseline_records)}, "
                f"missing {len(required_keys - baseline_keys)}, "
                f"unexpected {len(baseline_keys - required_keys)}"
            )
    expected = {
        (model, f"{repeat}:{key}")
        for model in args.models
        for key in tests
        for repeat in range(repeats)
    }
    output = args.input / "graded.jsonl"
    records = (
        [json.loads(line) for line in output.read_text().splitlines()]
        if output.exists()
        else []
    )
    done = {(r["model"], r["id"]) for r in records}
    if len(done) != len(records) or not done <= expected:
        raise ValueError("Unexpected or duplicate saved grading records")
    deadline = time.monotonic() + 4 * 3600
    with (
        ThreadPoolExecutor(max_workers=args.workers) as pool,
        output.open("a", buffering=1) as stream,
    ):
        while done != expected:
            pending = {}
            for model in args.models:
                for path in args.input.glob(f"{model}-rank*.jsonl"):
                    for line in path.read_text().splitlines(keepends=True):
                        if not line.endswith("\n"):
                            continue
                        row = json.loads(line)
                        key = (model, row["id"])
                        if key not in expected:
                            raise ValueError(f"Unexpected sample {key}")
                        if key not in done:
                            if key in pending:
                                raise ValueError(f"Duplicate generated sample {key}")
                            pending[key] = row
            for future in as_completed(
                [
                    pool.submit(grade_one, key[0], row, tests)
                    for key, row in pending.items()
                ]
            ):
                row = future.result()
                stream.write(json.dumps(row) + "\n")
                records.append(row)
                done.add((row["model"], row["id"]))
                if len(done) % 20 == 0:
                    print("GRADED", len(done), "/", len(expected), flush=True)
            status = {
                "graded": len(done),
                "expected": len(expected),
                "updated_at": time.time(),
            }
            (args.input / "grading-status.json").write_text(json.dumps(status) + "\n")
            if time.monotonic() > deadline:
                raise TimeoutError("Generation/grading did not finish in four hours")
            if not pending and done != expected:
                time.sleep(15)
    models = list(args.models)
    if args.baseline:
        records += baseline_records
        models.append("student")
    report = summarize(records, models, repeats)
    if args.comparison_summary:
        import numpy as np

        reference = json.loads(args.comparison_summary.read_text())
        before = reference["models"]["opd"]["per_problem_pass_at_1"]
        after = report["models"]["opd"]["per_problem_pass_at_1"]
        if before.keys() != after.keys() or reference["samples_per_problem"] != repeats:
            raise ValueError("Comparison requires identical problems and sample counts")
        delta = np.array([after[key] - before[key] for key in after])
        rng = np.random.default_rng(20260922)
        ci = np.quantile(
            delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(axis=1),
            [0.025, 0.975],
        ).tolist()
        report["comparison_to_4k"] = {
            "reference_summary": str(args.comparison_summary),
            "pass_at_1_difference": float(delta.mean()),
            "paired_problem_bootstrap_95_ci": ci,
        }
    temporary = args.input / "summary.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.input / "summary.json")
    print(
        "GRADING_COMPLETE",
        {k: v for k, v in report.items() if k != "models"},
        flush=True,
    )
    for model, summary in report["models"].items():
        print(
            model,
            {k: v for k, v in summary.items() if k != "per_problem_pass_at_1"},
            flush=True,
        )


if __name__ == "__main__":
    main()
