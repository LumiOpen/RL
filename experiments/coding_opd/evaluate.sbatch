#!/bin/bash
#SBATCH --partition=amd-tw-verification
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=8
#SBATCH --mem=512G
#SBATCH --time=03:00:00
#SBATCH --job-name=coding-baseline
#SBATCH --array=0-1
#SBATCH --output=experiments/coding_opd/eval-%A_%a.log
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
models=(student teacher)
model=${models[$SLURM_ARRAY_TASK_ID]}
output=experiments/coding_opd/run_eval_${SLURM_ARRAY_JOB_ID}
exec bash experiments/tml_opd_replication/container.sh env \
 CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((SLURM_GPUS_ON_NODE - 1))")" \
 XDG_CACHE_HOME="/tmp/coding-${SLURM_JOB_ID}/cache" TRITON_CACHE_DIR="/tmp/coding-${SLURM_JOB_ID}/triton" \
 uv run --no-project --python /shared_silo/scratch/rahul.aralikatte@amd.com/prime-rl/.venv/bin/python python \
 experiments/coding_opd/generate.py --model "$model" --output "$output" --workers "$SLURM_GPUS_ON_NODE" "$@"
