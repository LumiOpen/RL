#!/bin/bash
set -euo pipefail
uv run --no-project --python /opt/venv/bin/python python experiments/rocm_mopd/probe.py
for server in resources_servers/mopd_zero responses_api_agents/simple_agent responses_api_models/vllm_model; do
  uv venv --system-site-packages --python /opt/venv/bin/python "$NRL_MOPD_GYM_VENVS/$server/.venv"
done
RUN_ROOT="$PWD/experiments/rocm_mopd/run_${NRL_MOPD_RUN_ID}"
exec uv run --no-project --python /opt/venv/bin/python \
  python "$NRL_MOPD_ENTRYPOINT" --config "$NRL_MOPD_CONFIG" \
  "logger.log_dir=$RUN_ROOT/logs" \
  "checkpointing.checkpoint_dir=$RUN_ROOT/checkpoints" \
  "logger.wandb.name=${NRL_MOPD_RUN_LABEL}-${NRL_MOPD_RUN_ID}" \
  "logger.wandb.id=${NRL_MOPD_RUN_LABEL}-${NRL_MOPD_RUN_ID}"
