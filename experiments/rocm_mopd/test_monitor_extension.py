# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checks for automatic recovery boundaries and continuation artifact selection."""

import json

import pytest

from monitor_extension import arguments, retryable, validate_stage


@pytest.mark.parametrize(
    "status,text,expected",
    [
        ("NODE_FAIL", "", True),
        ("TIMEOUT", "", True),
        ("FAILED", "ActorDiedError: node disconnected", True),
        ("FAILED", "ActorDiedError: OutOfMemoryError", False),
        ("FAILED", "AssertionError: non-finite loss", False),
        ("CANCELLED", "", False),
        ("FAILED", "KeyError: missing baseline", False),
    ],
)
def test_retry_boundaries(status, text, expected):
    assert retryable(status, text) is expected


def test_continuation_requires_target_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoints"
    checkpoint.mkdir()
    (checkpoint / "latest_checkpoint_status.json").write_text(
        json.dumps({"last_checkpoint_step": 20})
    )
    (checkpoint / "step_20").mkdir()
    (checkpoint / "step_20/training_info.json").write_text(
        json.dumps({"current_step": 20, "total_steps": 20, "consumed_samples": 640})
    )
    with pytest.raises(AssertionError):
        validate_stage({"run": str(tmp_path)}, "train", 50)


def test_step_50_commands_use_separate_evaluation(tmp_path):
    state = {"target_step": 50, "arms": {"routed": {"run": str(tmp_path)}}}
    export, _ = arguments(state, "routed", "export", tmp_path / "state.json")
    evaluate, _ = arguments(state, "routed", "evaluate", tmp_path / "state.json")
    grade, _ = arguments(state, "routed", "grade", tmp_path / "state.json")
    assert export[export.index("--step") + 1] == "50"
    assert evaluate[-1] == "50"
    assert grade[grade.index("--input") + 1] == str(
        tmp_path / "evaluation_step_50/eval_lcb"
    )
    assert grade[grade.index("--baseline") + 1].endswith("run_eval_50305/graded.jsonl")
