#!/bin/bash
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
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
root=experiments/coding_opd
revision=$(uv run --no-project --python /usr/bin/python3 -c 'import json; print(json.load(open("experiments/coding_opd/protocol.json"))["lcb_evaluator_revision"])')
if [ ! -d "$root/reference_lcb/.git" ]; then
  git clone https://github.com/LiveCodeBench/LiveCodeBench "$root/reference_lcb"
fi
git -C "$root/reference_lcb" checkout "$revision"
mkdir -p "$root/data/grader_build"
cp "$root/grader/pyproject.toml" "$root/grader/uv.lock" "$root/grader/grade_one.py" "$root/data/grader_build/"
cp "$root/reference_lcb/lcb_runner/evaluation/testing_util.py" "$root/data/grader_build/"
cp "$(command -v uv)" "$root/data/grader_build/uv"
docker build -t coding-opd-grader:lcb-28fef95 -f "$root/grader/Dockerfile" "$root/data/grader_build"
docker save coding-opd-grader:lcb-28fef95 -o "$root/data/grader.tar"
singularity build "$root/data/grader.sif" "docker-archive://$root/data/grader.tar"
sha256sum "$root/data/grader.sif" > "$root/data/grader.sha256"
