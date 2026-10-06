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

"""Historical Qwen3 rendering and pretokenized NeMo SFT data."""

import json
from pathlib import Path

import ray
import torch

from nemo_rl.environments.interfaces import EnvironmentReturn


class HistoricalPromptDataset(torch.utils.data.Dataset):
    def __init__(self, path, task_name="deepmath"):
        self.rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
        self.task_name = task_name

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        tokens = torch.tensor(row["input_ids"], dtype=torch.long)
        return {
            "message_log": [
                {"role": "user", "content": row["question"], "token_ids": tokens}
            ],
            "length": len(tokens),
            "loss_multiplier": 1.0,
            "extra_env_info": {},
            "idx": index,
            "task_name": self.task_name,
        }


@ray.remote
class ZeroRewardEnvironment:  # pragma: no cover
    def step(self, message_log_batch, metadata):
        size = len(message_log_batch)
        return EnvironmentReturn(
            observations=[{"role": "environment", "content": ""} for _ in range(size)],
            metadata=metadata,
            next_stop_strings=[None] * size,
            rewards=torch.zeros(size),
            terminateds=torch.ones(size),
            answers=None,
        )

    def shutdown(self):
        return None
