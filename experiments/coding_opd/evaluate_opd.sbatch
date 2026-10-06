#!/bin/bash
#SBATCH --partition=amd-tw-verification
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=8
#SBATCH --mem=512G
#SBATCH --time=03:00:00
#SBATCH --job-name=coding-opd-eval
#SBATCH --output=experiments/coding_opd/eval-%j.log
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
cd "${NRL_REPO_ROOT:-$SLURM_SUBMIT_DIR}"
: "${CODING_ADAPTER:?Set CODING_ADAPTER to a complete checkpoint adapter directory}"
output="experiments/coding_opd/run_eval_${SLURM_JOB_ID}"
mkdir -p "$output"
UV=(bash experiments/tml_opd_replication/container.sh env
 CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((SLURM_GPUS_ON_NODE - 1))")"
 XDG_CACHE_HOME="/tmp/coding-${SLURM_JOB_ID}/cache" TRITON_CACHE_DIR="/tmp/coding-${SLURM_JOB_ID}/triton"
 uv run --no-project --python /shared_silo/scratch/rahul.aralikatte@amd.com/prime-rl/.venv/bin/python python)
"${UV[@]}" experiments/tml_opd_replication/eval_aime.py --mode merge --model opd --output "$output" --adapter "$CODING_ADAPTER"
run_name=$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$CODING_ADAPTER")")")")")")
step_name=$(basename "$(dirname "$(dirname "$(dirname "$CODING_ADAPTER")")")")
merged="experiments/tml_opd_replication/checkpoints/inference_${run_name}_${step_name}"
test -f "$merged/merge_validation.json"
"${UV[@]}" experiments/coding_opd/generate.py --model opd --model-path "$merged" --output "$output" --workers "$SLURM_GPUS_ON_NODE"
