#!/bin/bash
#SBATCH --partition=amd-tw-verification
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=48
#SBATCH --mem=512G
#SBATCH --time=04:00:00
#SBATCH --job-name=mopd-eval
#SBATCH --output=experiments/rocm_mopd/eval-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
RUN_ROOT=$(realpath "${1:?Pass the completed native training run}")
STEP=${2:-20}
MODEL="$RUN_ROOT/hf_step_$STEP"
EVAL_ROOT="$RUN_ROOT"
if [[ "$STEP" != 20 ]]; then
  EVAL_ROOT="$RUN_ROOT/evaluation_step_$STEP"
fi
test -f "$MODEL/export_validation.json"
export NRL_ROCM_RL_PACKAGES=/shared_silo/scratch/rahul.aralikatte@amd.com/RL/experiments/rocm_rl/packages
export NRL_ROCM_SFT_PACKAGES=/shared_silo/scratch/rahul.aralikatte@amd.com/RL/experiments/rocm_sft/packages
UV=(bash experiments/tml_opd_replication/container.sh env
 CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((SLURM_GPUS_ON_NODE - 1))")"
 XDG_CACHE_HOME="/tmp/mopd-eval-$SLURM_JOB_ID/cache"
 TRITON_CACHE_DIR="/tmp/mopd-eval-$SLURM_JOB_ID/triton"
 uv run --no-project --python /shared_silo/scratch/rahul.aralikatte@amd.com/prime-rl/.venv/bin/python python)
"${UV[@]}" experiments/coding_opd/generate.py --model opd --model-path "$MODEL" \
  --data experiments/coding_opd/data/eval_clean.jsonl --output "$EVAL_ROOT/eval_lcb" \
  --workers "$SLURM_GPUS_ON_NODE"
AIME="$EVAL_ROOT/aime_${RUN_ROOT##*/}"
mkdir -p "$AIME"
"${UV[@]}" experiments/coding_opd/aime_regression.py --mode prepare --output "$AIME"
"${UV[@]}" experiments/coding_opd/generate.py --model opd --model-path "$MODEL" \
  --data "$AIME/aime_prompts.jsonl" --output "$AIME" --workers "$SLURM_GPUS_ON_NODE"
"${UV[@]}" experiments/coding_opd/aime_regression.py --mode score --output "$AIME"
