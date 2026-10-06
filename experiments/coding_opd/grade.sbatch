#!/bin/bash
#SBATCH --partition=amd-tw-verification
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=04:30:00
#SBATCH --job-name=coding-grade
#SBATCH --output=experiments/coding_opd/grade-%j.log
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
uv run --no-project --python /usr/bin/python3 experiments/coding_opd/grade.py --self-test
exec uv --no-config run --no-sync --project experiments/coding_opd/grader \
 python experiments/coding_opd/grade.py --workers "$SLURM_CPUS_PER_TASK" "$@"
