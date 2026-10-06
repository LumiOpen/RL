# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit real teacher routing and model probes around a native MOPD smoke."""

import asyncio
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import uuid

import requests
import torch
from tensordict import TensorDict

from nemo_rl.algorithms.opd import TQTeacherLogprobCoordinator
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.interfaces import PromptGroupRecord

entry_path = (
    Path(__file__).resolve().parents[2] / "examples/run_grpo_single_controller.py"
)
entry_spec = importlib.util.spec_from_file_location("native_sc_entry", entry_path)
entry = importlib.util.module_from_spec(entry_spec)
entry_spec.loader.exec_module(entry)


class NativeAudit:
    def __init__(self, config, tokenizer, actors):
        self.actors = actors
        self.tokenizer = tokenizer
        self.model_name = config.policy["model_name"]
        self.output = Path(config.logger["log_dir"]).parent / "correctness_audit.json"
        texts = [
            "17 plus 24 equals 41.",
            "def larger(a, b): return max(a, b)",
            "Three times five plus five equals twenty.",
            "def reverse_copy(items): return items[::-1]",
        ]
        encoded = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
        lengths = torch.tensor([len(row) for row in encoded], dtype=torch.int32)
        tokens = torch.full(
            (4, int(lengths.max())), tokenizer.pad_token_id, dtype=torch.long
        )
        for i, row in enumerate(encoded):
            tokens[i, : len(row)] = torch.tensor(row)
        self.data = BatchedDataDict(input_ids=tokens, input_lengths=lengths)
        self.mask = torch.arange(tokens.shape[1])[None, :] < lengths[:, None]
        self.mask[:, 0] = False
        self.teachers = actors.teacher_worker_groups
        self.coordinator = TQTeacherLogprobCoordinator(
            dp_client=actors.dp_client,
            teacher_worker_groups=self.teachers,
            alias_to_group_alias=actors.alias_to_group_alias,
            on_policy_distillation_cfg=config.on_policy_distillation.model_dump(),
        )
        self.report = {"resumed_step": actors.save_state.total_steps}

    def scores(self):
        student = self.actors.trainer_handle.get_logprobs(self.data)["logprobs"]
        teachers = {
            alias: teacher.get_logprobs(self.data)["reference_logprobs"]
            for alias, teacher in self.teachers.items()
        }
        for scores in [student, *teachers.values()]:
            assert torch.isfinite(scores[self.mask]).all()
        return student, teachers

    async def check_routing(self, direct):
        client = self.actors.dp_client
        ids = [f"audit-{uuid.uuid4().hex}" for _ in range(4)]
        meta = client.put_samples(
            sample_ids=ids,
            partition_id=self.actors.partition_id,
            fields=TensorDict(dict(self.data), batch_size=[4]),
        )
        meta = replace(meta, sequence_lengths=self.data["input_lengths"].tolist())
        maximum_error = 0.0
        batch_shape_error = 0.0
        for i, alias in enumerate(["math_agent", "coding_agent"] * 2):
            record = PromptGroupRecord(
                prompt_idx=i,
                prompt=[],
                extra_env_info={"agent_ref": {"name": alias}},
                metadata={},
                completions=[],
                rollout_metrics={},
            )
            await self.coordinator.enrich(meta.slice(i, i + 1), record)
            row = client.get_samples(
                sample_ids=[ids[i]],
                partition_id=meta.partition_id,
                select_fields=[self.coordinator.teacher_logprobs_field],
            )[self.coordinator.teacher_logprobs_field][0]
            length = int(self.data["input_lengths"][i])
            single = BatchedDataDict(
                input_ids=self.data["input_ids"][i : i + 1, :length],
                input_lengths=self.data["input_lengths"][i : i + 1],
            )
            group = self.actors.alias_to_group_alias[alias]
            same_shape = self.teachers[group].get_logprobs(single)[
                "reference_logprobs"
            ][0]
            error = (row[1:length] - same_shape[1:length]).abs().max().item()
            maximum_error = max(maximum_error, error)
            batch_shape_error = max(
                batch_shape_error,
                (same_shape[1:length] - direct[group][i, 1:length]).abs().max().item(),
            )
            torch.testing.assert_close(
                row[1:length], same_shape[1:length], atol=0.001, rtol=0
            )
        client.clear_samples(sample_ids=ids, partition_id=meta.partition_id)
        self.report["routed_vs_direct_max_abs_logprob_error"] = maximum_error
        self.report["different_batch_shape_max_abs_logprob_difference"] = (
            batch_shape_error
        )

    def before(self):
        self.student_before, self.teachers_before = self.scores()
        for alias, teacher in self.teachers.items():
            if (
                teacher.model_name == self.model_name
                and self.report["resumed_step"] == 0
            ):
                gap = (self.student_before - self.teachers_before[alias])[self.mask]
                self.report["self_teacher_max_abs_logprob_gap"] = gap.abs().max().item()
                torch.testing.assert_close(
                    gap, torch.zeros_like(gap), atol=0.001, rtol=0
                )
        asyncio.run(self.check_routing(self.teachers_before))
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(self.report, indent=2) + "\n")
        print("MOPD_PRETRAIN_AUDIT_PASSED", self.report, flush=True)

    def after(self):
        student, teachers = self.scores()
        delta = (student - self.student_before)[self.mask].abs().max().item()
        assert delta > 0, "Student probe did not change after training"
        for alias in teachers:
            torch.testing.assert_close(
                teachers[alias], self.teachers_before[alias], atol=0, rtol=0
            )
        messages = [{"role": "user", "content": "What is 17 plus 24?"}]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        expected = self.tokenizer.encode(rendered, add_special_tokens=False)
        response = requests.post(
            self.actors.gen_handle.dp_openai_server_base_urls[0] + "/chat/completions",
            json={
                "model": self.model_name,
                "messages": messages,
                "max_tokens": 1,
                "temperature": 0,
                "return_tokenized_data": True,
            },
            timeout=60,
        )
        response.raise_for_status()
        actual = response.json()["choices"][0]["message"]["prompt_token_ids"]
        assert actual == expected, (
            "HTTP generation prompt differs from training template"
        )
        self.report.update(
            student_probe_max_abs_change=delta,
            teacher_probes_unchanged=True,
            http_prompt_tokens_match_training_template=True,
            student_probe={
                "input_ids": self.data["input_ids"].tolist(),
                "input_lengths": self.data["input_lengths"].tolist(),
                "logprobs": student.tolist(),
            },
            passed=True,
        )
        self.output.write_text(json.dumps(self.report, indent=2) + "\n")
        print("MOPD_POSTTRAIN_AUDIT_PASSED", self.report, flush=True)


def main():
    original_setup = entry.setup_single_controller
    original_run = entry._run_with_controller_liveness_watch
    audit = None

    def setup(config, tokenizer, **kwargs):
        nonlocal audit
        actors, timing = original_setup(config, tokenizer, **kwargs)
        audit = NativeAudit(config, tokenizer, actors)
        audit.before()
        return actors, timing

    def run(*args, **kwargs):
        result = original_run(*args, **kwargs)
        audit.after()
        return result

    entry.setup_single_controller = setup
    entry._run_with_controller_liveness_watch = run
    entry.main()


if __name__ == "__main__":
    main()
