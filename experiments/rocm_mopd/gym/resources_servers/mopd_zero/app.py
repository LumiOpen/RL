# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reward-free single-turn environment for teacher-logprob distillation."""

from nemo_gym.base_resources_server import (
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class DistillationResourcesServer(SimpleResourcesServer):
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        return BaseVerifyResponse(**body.model_dump(), reward=0.0)


if __name__ == "__main__":
    DistillationResourcesServer.run_webserver()
