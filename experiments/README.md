# ROCm experiment recipes

**Project complete, October 6, 2026.** These recipes record the AMD MI325X work
behind [TM_OPD_RECIPE.md](../TM_OPD_RECIPE.md) and [MOPD_RECIPE.md](../MOPD_RECIPE.md).
All training, evaluation and grading jobs are finished; checkpoints and raw
outputs are retained locally. No further run is scheduled.

| Directory | Purpose |
| --- | --- |
| [rocm_sft](rocm_sft/README.md) | Native SFT smoke on one/two GPUs |
| [rocm_rl](rocm_rl/README.md) | Native GRPO and conditional top-k OPD smoke |
| [rocm_opd_dsr](rocm_opd_dsr/README.md) | Prime-RL dataset comparison, packing and dedicated inference |
| [tml_opd_replication](tml_opd_replication/README.md) | Historical LoRA SFT, sampled-token OPD, audits, monitoring and AIME24 evaluation |
| [coding_opd](coding_opd/README.md) | Completed TACO → LiveCodeBench single-teacher experiment and isolated grader |
| [rocm_mopd](rocm_mopd/README.md) | Completed native Megatron MOPD, matched control, 20/50-update results and recovery controller |

## Final conclusions

| Experiment | Result |
|---|---|
| Single-teacher math OPD, 100 updates | AIME24 54.58% → 61.25%, 960 samples/model |
| Single-teacher coding OPD, 100 updates, 16K rollouts | LiveCodeBench 33.71% → 39.57%, 700 samples/model |
| Native MOPD, 50 updates | Coding 33.71%; math 55.00%; no clear gain over SFT or matched native control |

The MOPD AIME screen uses only 120 samples (matched SFT 59.17%), so its baseline
must not be substituted for the 960-sample math reproduction baseline. Native
MOPD also changes adapter continuation, objective/reduction, optimizer and batch
size relative to the historical recipe. Its completion is evidence of correct
execution, not of an accuracy benefit or a general verdict on multiple teachers.

Current-branch GPU training, checkpoint and resume checks: [September 22 sanity results](gpu-sanity.md).

## Tested environment

The full reproduction used NeMo-RL `feff70f5b6239d90dc703ae79b973ea809550732` with the included runtime changes and Automodel `1814c6c93a66b9d59d254960ef6a99a64249b671`. This PR ports those changes onto the fork's newer main; it retains main's Automodel pin. Historical GPU results in the recipe/JSON files describe the original checkout, not a rerun on the newer base.

The RL/TML wrappers use Singularity/Apptainer, `primus_v26.2_moefix.sif`, and an existing Python 3.12 environment with Torch 2.11 / ROCm 7.2 and vLLM 0.25.1 ROCm wheels. The standalone SFT smoke uses `primus_v26.5-pytorch2.12-te2.15.sif` and its interpreter instead. These cluster images and ROCm wheels are prerequisites, not repository artifacts. The frozen overlays supply the remaining imports without replacing ROCm Torch/vLLM:

```bash
uv pip install --target experiments/rocm_sft/packages --python-version 3.12 \
  --no-deps -r experiments/rocm_sft/requirements-overlay.txt
uv pip install --target experiments/rocm_rl/packages --python-version 3.12 \
  --no-deps -r experiments/rocm_rl/requirements-overlay.txt
git submodule update --init 3rdparty/Automodel-workspace/Automodel
uv run --no-project --python /usr/bin/python3 experiments/prepare_smoke_data.py
```

Native MOPD also uses the Primus v26.5 image (Torch 2.12/ROCm 7.15, AMD TE 2.15),
with its own locked overlay and source-pinned Bridge/Core/Gym. Setup is in
[rocm_mopd/README.md](rocm_mopd/README.md). Keep the two runtime stacks separate.

| Dependency file | Purpose |
|---|---|
| `rocm_sft/requirements-overlay.txt`, `rocm_rl/requirements-overlay.txt` | Frozen original SFT/RL overlays |
| `rocm_mopd/deps/pyproject.toml`, `rocm_mopd/deps/uv.lock` | Canonical added native MOPD dependencies |
| [rocm_mopd/requirements.txt](rocm_mopd/requirements.txt) | Generated export of the native lockfile; container Torch/NumPy/Ray excluded by markers |
| `coding_opd/grader/pyproject.toml`, `coding_opd/grader/uv.lock`, [requirements.txt](coding_opd/grader/requirements.txt) | Isolated CPU grader, NumPy 2.2.6 |

The root project uses `pyproject.toml`/`uv.lock`, not a root `requirements.txt`.
Regenerate requirements exports with `uv export --locked`; do not resolve newer
dependencies over the recorded experiment environment as part of reproduction.

Do not resolve the root CUDA lockfile over an existing ROCm environment. The wrappers use `NEMO_RL_PY_EXECUTABLES_SYSTEM=1` so Ray actors inherit the prepared interpreter and `PYTHONPATH`. They use the checkout containing the wrapper, and accept `NRL_ROCM_IMAGE`, `NRL_ROCM_SFT_PACKAGES`, `NRL_ROCM_RL_PACKAGES`, and, for TML, `NRL_AUTOMODEL_PATH` overrides. For the historical Automodel version, create a separate source checkout and point `NRL_AUTOMODEL_PATH` at it.

## Cluster paths and launch settings

Submit from the repository root. The `.sbatch` files are the tested TensorWave cluster recipes: `amd-tw-verification`, normal QoS, `eno0` networking, site-specific node exclusions, container bind mounts, `/shared_silo/scratch/hf_cache`, and the existing Prime-RL Python interpreter. Edit these settings and the YAML/JSON model, data, adapter and restore paths for a different installation. Relative Slurm log paths resolve from the submission directory. `NRL_REPO_ROOT` can override that directory inside the job.

The wrappers run with Hugging Face offline mode; populate the pinned model/dataset snapshots first. W&B-enabled recipes use the environment credential or the local `$HOME/.wandb_key` file and report to the configured entity/project. Change that entity/project for your account. No credentials are stored here.

Use the small SFT/GRPO/OPD recipes and `rocm_opd_dsr/verify_rccl_group.py` to validate a new installation before allocating a full training run. The RCCL probe runs via `uv run ... python -m torch.distributed.run --nproc_per_node=2 experiments/rocm_opd_dsr/verify_rccl_group.py` inside an allocated ROCm container. Generated datasets, dependencies and run artifacts are ignored by Git; small JSON files retain historical configuration, checksums and results.
