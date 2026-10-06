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

"""Run historical sampled-token OPD with adapter and checkpoint audits."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import ray
from omegaconf import OmegaConf

from experiments.tml_opd_replication.opd_data import (
    HistoricalPromptDataset,
    ZeroRewardEnvironment,
)
from nemo_rl.algorithms.distillation import MasterConfig, distillation_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--verify-restore-audit", type=Path)
    args, overrides = parser.parse_known_args()
    register_omegaconf_resolvers()
    config = MasterConfig(
        **OmegaConf.to_container(
            parse_hydra_overrides(load_config(args.config), overrides), resolve=True
        )
    )
    assert config.loss_fn.kl_type == "sampled_reverse"
    assert config.loss_fn.sampled_token_reduction == "sum"
    assert (
        config.policy["train_global_batch_size"]
        == config.distillation.num_prompts_per_step
        * config.distillation.num_generations_per_prompt
    )
    worker = "experiments.tml_opd_replication.audit_worker.AuditedSFTWorker"
    ACTOR_ENVIRONMENT_REGISTRY[worker] = sys.executable
    config.policy["worker_extension_cls_fqn"] = worker
    init_ray()
    if "TML_NODE_COUNT" in os.environ:
        expected = int(os.environ["TML_NODE_COUNT"])
        for _ in range(120):
            nodes = [node for node in ray.nodes() if node["Alive"]]
            if len(nodes) == expected:
                assert all(
                    node["Resources"].get("GPU", 0) == int(os.environ["TML_GPU_COUNT"])
                    for node in nodes
                )
                break
            time.sleep(2)
        else:
            raise TimeoutError(f"Expected {expected} Ray nodes, got {nodes}")
    tokenizer = get_tokenizer(config.policy["tokenizer"])
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"], tokenizer
    )
    task_name = config.data.get("task_name", "deepmath")
    dataset = HistoricalPromptDataset(
        config.data["tokenized_path"], task_name=task_name
    )
    environment = ZeroRewardEnvironment.options(
        runtime_env={"py_executable": sys.executable}
    ).remote()
    (
        student,
        teacher,
        generation,
        _,
        loader,
        val_loader,
        loss,
        logger,
        checkpointer,
        state,
        config,
    ) = setup(config, tokenizer, dataset, None)
    if state.total_steps == 0:
        verification = student.run_all_workers_single_data("verify_initial_adapter")
        print("INITIAL_ADAPTER_VERIFIED", verification, flush=True)
    if args.audit or args.verify_restore_audit:
        before = student.run_all_workers_single_data("parameter_audit")
    if args.verify_restore_audit:
        expected = json.loads(args.verify_restore_audit.read_text())["after"]
        if before != expected:
            raise ValueError("Restored OPD parameters differ from the checkpoint audit")
        print("OPD_RESTORE_AUDIT_PASSED", flush=True)
    with checkpointer:
        distillation_train(
            student,
            teacher,
            generation,
            loader,
            val_loader,
            tokenizer,
            loss,
            {task_name: environment},
            {},
            logger,
            checkpointer,
            state,
            config,
        )
    if args.audit:
        after = student.run_all_workers_single_data("parameter_audit")
        for first, last in zip(before, after, strict=True):
            assert first["frozen_sha256"] == last["frozen_sha256"]
            assert first["adapter_sha256"] != last["adapter_sha256"]
        (Path(config.logger["log_dir"]) / "parameter_audit.json").write_text(
            json.dumps({"before": before, "after": after}, indent=2)
        )
        print("OPD_PARAMETER_AUDIT_PASSED", flush=True)


if __name__ == "__main__":
    main()
