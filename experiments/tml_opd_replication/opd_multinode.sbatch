#!/bin/bash
#SBATCH --partition=amd-tw-verification
#SBATCH --qos=normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --exclude=tus1-p15-g5,tus1-p15-g62
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --job-name=tml-opd-32gpu
#SBATCH --output=experiments/tml_opd_replication/opd-%j.log
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
export TML_HEAD_NODE
TML_HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export TML_HEAD_IP
TML_HEAD_IP=$(getent ahostsv4 "${TML_HEAD_NODE}.amd.tus.tensorwave.lan" | awk 'NR==1 {print $1}')
export TML_RAY_PORT=$((25000 + SLURM_JOB_ID % 10000))
export TML_OPD_RUN_ID=${TML_OPD_RUN_ID:-$SLURM_JOB_ID}
export TML_OPD_CONFIG=${TML_OPD_CONFIG:-experiments/tml_opd_replication/opd.yaml}
mkdir -p "experiments/tml_opd_replication/run_opd_${TML_OPD_RUN_ID}"
exec 9>"experiments/tml_opd_replication/run_opd_${TML_OPD_RUN_ID}/training.lock"
echo "Waiting for run ${TML_OPD_RUN_ID} writer lock"
flock 9
echo "Acquired run ${TML_OPD_RUN_ID} writer lock"
echo "Starting $SLURM_NNODES nodes on $SLURM_JOB_NODELIST; head=$TML_HEAD_IP:$TML_RAY_PORT"
srun --kill-on-bad-exit=1 --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  bash experiments/tml_opd_replication/opd_multinode_node.sh "$@"
