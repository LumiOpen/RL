#!/bin/bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BRIDGE="$REPO/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
export NRL_ROCM_IMAGE=${NRL_ROCM_IMAGE:-/shared_silo/scratch/containers/primus_v26.5-pytorch2.12-te2.15.sif}
exec "$REPO/experiments/rocm_rl/container.sh" env \
  -u NVTE_FLASH_ATTN -u NVTE_FUSED_ATTN -u NVTE_UNFUSED_ATTN \
  PYTHONPATH="$REPO/experiments/rocm_mopd/deps/.venv/lib/python3.12/site-packages:$BRIDGE/src:$BRIDGE/3rdparty/Megatron-LM:$REPO/3rdparty/Gym-workspace/Gym:${NRL_ROCM_RL_PACKAGES:-$REPO/experiments/rocm_rl/packages}:${NRL_ROCM_SFT_PACKAGES:-$REPO/experiments/rocm_sft/packages}:/opt/venv/lib/python3.12/site-packages:$REPO" \
  "$@"
