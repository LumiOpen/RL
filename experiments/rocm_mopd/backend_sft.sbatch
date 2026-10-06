#!/bin/bash
#SBATCH --partition=amd-tw-verification
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --job-name=mopd-backend-sft
#SBATCH --output=experiments/rocm_mopd/backend-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export NRL_ROCM_RL_PACKAGES=/shared_silo/scratch/rahul.aralikatte@amd.com/RL/experiments/rocm_rl/packages
export NRL_ROCM_SFT_PACKAGES=/shared_silo/scratch/rahul.aralikatte@amd.com/RL/experiments/rocm_sft/packages
TASK_TMP=/tmp/nrl_mopd_$SLURM_JOB_ID
mkdir -p "$TASK_TMP"
bash experiments/rocm_mopd/container.sh env \
  TMPDIR="$TASK_TMP" XDG_CACHE_HOME="$TASK_TMP/cache" \
  AITER_JIT_DIR="$TASK_TMP/aiter" USE_ROCM_AITER_ROPE_BACKEND=0 \
  MPLCONFIGDIR="$TASK_TMP/matplotlib" CUDA_VISIBLE_DEVICES=0 \
  uv run --no-project --python /opt/venv/bin/python \
  python experiments/rocm_mopd/probe.py
exec bash experiments/rocm_mopd/container.sh env \
  TMPDIR="$TASK_TMP" RAY_TMPDIR="$TASK_TMP" XDG_CACHE_HOME="$TASK_TMP/cache" \
  AITER_JIT_DIR="$TASK_TMP/aiter" USE_ROCM_AITER_ROPE_BACKEND=0 \
  MPLCONFIGDIR="$TASK_TMP/matplotlib" \
  TRITON_CACHE_DIR="$TASK_TMP/triton" CUDA_VISIBLE_DEVICES=0 \
  WANDB_DATA_DIR="$TASK_TMP/wandb-data" WANDB_CACHE_DIR="$TASK_TMP/wandb-cache" \
  WANDB_CONFIG_DIR="$TASK_TMP/wandb-config" \
  RAY_OVERRIDE_RESOURCES='{"CPU":8,"GPU":1,"memory":68719476736,"object_store_memory":2147483648}' \
  NCCL_SOCKET_IFNAME=eno0 GLOO_SOCKET_IFNAME=eno0 \
  uv run --no-project --python /opt/venv/bin/python \
  python examples/run_sft.py --config experiments/rocm_mopd/backend_sft.yaml \
  "logger.log_dir=experiments/rocm_mopd/run_${SLURM_JOB_ID}/logs" \
  "checkpointing.checkpoint_dir=experiments/rocm_mopd/run_${SLURM_JOB_ID}/checkpoints" \
  "logger.wandb.name=mopd-native-backend-${SLURM_JOB_ID}" \
  "logger.wandb.id=mopd-native-backend-${SLURM_JOB_ID}"
