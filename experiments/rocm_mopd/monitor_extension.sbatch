#!/bin/bash
#SBATCH --partition=amd-tw-verification
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --job-name=mopd50-monitor
#SBATCH --output=experiments/rocm_mopd/extension-monitor-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
exec uv run --no-project --python /usr/bin/python3 \
  experiments/rocm_mopd/monitor_extension.py --state experiments/rocm_mopd/extension_state.json
