#!/bin/bash
#SBATCH --partition=amd-tw-verification
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --job-name=mopd-native-smoke
#SBATCH --output=experiments/rocm_mopd/native-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export NRL_ROCM_RL_PACKAGES=/shared_silo/scratch/rahul.aralikatte@amd.com/RL/experiments/rocm_rl/packages
export NRL_ROCM_SFT_PACKAGES=/shared_silo/scratch/rahul.aralikatte@amd.com/RL/experiments/rocm_sft/packages
TASK_TMP=/tmp/nrl_mopd_$SLURM_JOB_ID
mkdir -p "$TASK_TMP"
exec bash experiments/rocm_mopd/container.sh env \
  SLURM_JOB_ID="$SLURM_JOB_ID" \
  NRL_MOPD_RUN_ID="${NRL_MOPD_RUN_ID:-$SLURM_JOB_ID}" \
  NRL_MOPD_CONFIG="${NRL_MOPD_CONFIG:-experiments/rocm_mopd/native_smoke.yaml}" \
  NRL_MOPD_ENTRYPOINT="${NRL_MOPD_ENTRYPOINT:-examples/run_grpo_single_controller.py}" \
  NRL_MOPD_RUN_LABEL="${NRL_MOPD_RUN_LABEL:-mopd-native-smoke}" \
  TMPDIR="$TASK_TMP" RAY_TMPDIR="$TASK_TMP" XDG_CACHE_HOME="$TASK_TMP/cache" \
  AITER_JIT_DIR="$TASK_TMP/aiter" USE_ROCM_AITER_ROPE_BACKEND=0 \
  MPLCONFIGDIR="$TASK_TMP/matplotlib" TRITON_CACHE_DIR="$TASK_TMP/triton" \
  WANDB_DATA_DIR="$TASK_TMP/wandb-data" WANDB_CACHE_DIR="$TASK_TMP/wandb-cache" \
  WANDB_CONFIG_DIR="$TASK_TMP/wandb-config" \
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((SLURM_GPUS_ON_NODE - 1))")" \
  RAY_OVERRIDE_RESOURCES="{\"CPU\":$SLURM_CPUS_PER_TASK,\"GPU\":$SLURM_GPUS_ON_NODE,\"memory\":$((SLURM_MEM_PER_NODE * 524288)),\"object_store_memory\":4294967296}" \
  NCCL_SOCKET_IFNAME=eno0 GLOO_SOCKET_IFNAME=eno0 \
  NEMO_GYM_EXTRA_ROOTS="$PWD/experiments/rocm_mopd/gym" \
  NRL_MOPD_GYM_VENVS="$TASK_TMP/gym-venvs" \
  bash experiments/rocm_mopd/run_native_smoke.sh
