#!/bin/bash
#SBATCH --partition=amd-tw-verification
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --job-name=mopd-native-probe
#SBATCH --output=experiments/rocm_mopd/probe-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export NRL_ROCM_RL_PACKAGES=/shared_silo/scratch/rahul.aralikatte@amd.com/RL/experiments/rocm_rl/packages
export NRL_ROCM_SFT_PACKAGES=/shared_silo/scratch/rahul.aralikatte@amd.com/RL/experiments/rocm_sft/packages
TASK_TMP=/tmp/nrl_mopd_$SLURM_JOB_ID
mkdir -p "$TASK_TMP"
exec bash experiments/rocm_mopd/container.sh env \
  TMPDIR="$TASK_TMP" XDG_CACHE_HOME="$TASK_TMP/cache" \
  AITER_JIT_DIR="$TASK_TMP/aiter" USE_ROCM_AITER_ROPE_BACKEND=0 \
  TRITON_CACHE_DIR="$TASK_TMP/triton" CUDA_VISIBLE_DEVICES=0 \
  uv run --no-project --python /opt/venv/bin/python \
  python experiments/rocm_mopd/probe.py
