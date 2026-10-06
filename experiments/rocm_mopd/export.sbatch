#!/bin/bash
#SBATCH --partition=amd-tw-verification
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=00:45:00
#SBATCH --job-name=mopd-export
#SBATCH --output=experiments/rocm_mopd/export-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export NRL_ROCM_RL_PACKAGES=/shared_silo/scratch/rahul.aralikatte@amd.com/RL/experiments/rocm_rl/packages
export NRL_ROCM_SFT_PACKAGES=/shared_silo/scratch/rahul.aralikatte@amd.com/RL/experiments/rocm_sft/packages
TASK_TMP=/tmp/nrl_mopd_export_$SLURM_JOB_ID
mkdir -p "$TASK_TMP"
exec bash experiments/rocm_mopd/container.sh env \
  CUDA_VISIBLE_DEVICES=0 TMPDIR="$TASK_TMP" AITER_JIT_DIR="$TASK_TMP/aiter" \
  USE_ROCM_AITER_ROPE_BACKEND=0 TRITON_CACHE_DIR="$TASK_TMP/triton" \
  uv run --no-project --python /opt/venv/bin/python python \
  experiments/rocm_mopd/export_native.py "$@"
