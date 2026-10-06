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

"""Fail-fast native MOPD backend import and GPU probe."""

import json
import sys

import torch

print(
    json.dumps(
        {"python": sys.version, "torch": torch.__version__, "hip": torch.version.hip}
    ),
    flush=True,
)
assert torch.cuda.is_available()
print("GPU", torch.cuda.get_device_name(), flush=True)
x = torch.ones(32, 32, device="cuda", requires_grad=True)
(x @ x).sum().backward()
assert torch.isfinite(x.grad).all()
print("ROCM_BACKWARD_PASSED", flush=True)

import transformer_engine.pytorch  # noqa: E402, F401

print("TRANSFORMER_ENGINE_IMPORTED", flush=True)
from nemo_rl.models.policy.workers.megatron_policy_worker import MegatronPolicyWorker  # noqa: E402

print("MEGATRON_POLICY_IMPORTED", type(MegatronPolicyWorker).__name__, flush=True)
from nemo_rl.models.policy.teacher_worker_group import TeacherWorkerGroup  # noqa: E402

print("NATIVE_TEACHER_IMPORTED", TeacherWorkerGroup.__name__, flush=True)
from nemo_rl.environments.nemo_gym import setup_nemo_gym_config  # noqa: E402

print("GYM_INTEGRATION_IMPORTED", setup_nemo_gym_config.__name__, flush=True)
from nemo_gym.cli.env import RunHelper  # noqa: E402
from nemo_gym.base_resources_server import SimpleResourcesServer  # noqa: E402
from nemo_rl.data_plane.adapters.transfer_queue import TQDataPlaneClient  # noqa: E402
from hypercorn.asyncio import serve  # noqa: E402
from quart import Quart  # noqa: E402

print(
    "NATIVE_RUNTIME_DEPENDENCIES_IMPORTED",
    RunHelper.__name__,
    SimpleResourcesServer.__name__,
    TQDataPlaneClient.__name__,
    serve.__name__,
    Quart.__name__,
    flush=True,
)
