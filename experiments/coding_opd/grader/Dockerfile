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

FROM python:3.12-slim@sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9
COPY uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock grade_one.py testing_util.py /app/
RUN uv sync --locked --all-extras && chmod -R a+rX /app
ENV UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
ENTRYPOINT ["uv", "run", "--no-sync", "--project", "/app", "python", "/app/grade_one.py"]
