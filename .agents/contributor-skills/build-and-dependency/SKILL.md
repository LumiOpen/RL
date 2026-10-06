---
name: build-and-dependency
description: Build and dependency management for NeMo-RL. Covers Docker image building and running, uv usage, venv setup, and adding dependencies.
when_to_use: Setting up a dev environment; building or running the Docker container; adding or removing a dependency; uv errors; 'how do I install', 'ModuleNotFoundError', 'build the image', 'run in Docker', 'uv sync fails'.
---

# Build and Dependency Guide

---

## Docker Images

Build the release image (includes all dependencies and pre-fetched venvs):

```bash
# Build from local source
docker buildx build -f docker/Dockerfile --tag nemo-rl:latest .

# Build from a specific git ref (no local clone needed)
docker buildx build -f docker/Dockerfile \
    --build-arg NRL_GIT_REF=main \
    --tag nemo-rl:latest \
    https://github.com/NVIDIA-NeMo/RL.git
```

Skip optional backends to reduce build time:

```bash
# Skip vLLM, SGLang, and TRT-LLM
docker buildx build -f docker/Dockerfile \
    --build-arg SKIP_VLLM_BUILD=1 \
    --build-arg SKIP_SGLANG_BUILD=1 \
    --build-arg SKIP_TRTLLM_BUILD=1 \
    --tag nemo-rl:latest .
```

See @docs/docker.md for full options.

---

## Always Use uv

**Never use `pip install` directly** — always go through `uv`.

```bash
# Run a script
uv run examples/run_grpo.py

# Run tests
uv run --group test bash tests/run_unit.sh

# Install all deps from lockfile
uv sync --locked
```

Exception: `Dockerfile.ngc_pytorch` is exempt from this rule.

---

## Adding Dependencies

```bash
# Add a runtime dependency
uv add <package>

# Add an optional dependency
uv add --optional --extra <group> <package>

# Regenerate the lockfile after changes
uv lock
```

Commit both `pyproject.toml` and `uv.lock` together:

```bash
git add pyproject.toml uv.lock
git commit -s -m "build: add <package> dependency"
```

---

## Bumping Megatron-Bridge (and Megatron-LM)

`megatron-bridge` is installed directly from the submodule's own `pyproject.toml`
(`[tool.uv.sources]` points at `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge`),
and `megatron-core` at the Megatron-LM checkout nested inside it. Their dependency
lists — including `megatron-core[dev,mlm]` — flow transitively, so there is **no
mirrored dependency list to maintain** in NeMo RL.

The bump procedure is:

```bash
# 1. Check out the new Megatron-Bridge commit (brings its pinned Megatron-LM along)
cd 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
git fetch origin && git checkout <new-commit>
git submodule update --init --recursive
cd -

# 2. Relock. uv detects the submodule pyproject.toml change automatically and
#    rebuilds megatron-bridge's metadata (~1 min warm cache, ~3 min cold).
uv lock

# 3. Review `git diff uv.lock`, then commit the submodule pointer and uv.lock together.
```

### When `uv lock` errors after a bump

- **`Package X was included as a URL dependency. URL dependencies must be expressed
  as direct requirements or constraints`** — upstream changed or added a git/URL dep
  in Megatron-Bridge's or Megatron-LM's `[tool.uv.sources]` (e.g., Megatron-LM bumping
  its `emerging-optimizers` rev). uv requires such URLs to also appear as a direct
  requirement or constraint of the root project: update the matching pin in
  `constraint-dependencies` (where `emerging-optimizers` and `fast-hadamard-transform`
  live today) / the `mcore` extra / `[tool.uv.sources]` / `override-dependencies` in
  `pyproject.toml` to the same URL/rev.
- **Version conflicts** — resolve via `[tool.uv] override-dependencies` in
  `pyproject.toml`, same as any other conflict.

### Known fragilities

- uv currently does **not** enforce `requires-python` for path dependencies
  (Megatron-Bridge caps `<3.13` while NeMo RL runs 3.13). If a future uv upgrade
  starts enforcing it, `uv lock` will fail loudly: ask upstream to relax the cap,
  or reintroduce a thin proxy package with its own `pyproject.toml`.
- Megatron-Bridge's and Megatron-LM's own `[tool.uv.sources]` are honored for their
  dependencies. Root-level pins (e.g., `nvidia-modelopt` from git) win only because
  they are direct requirements of NeMo RL — don't remove those direct deps without
  re-checking where the transitive resolution lands.

---

## Common Pitfalls

| Problem | Cause | Fix |
|---------|-------|-----|
| `uv sync --locked` fails | Dependency conflict or stale lockfile | Re-run `uv lock` and commit updated lock |
| `ModuleNotFoundError` after pip install | pip installed outside uv-managed venv | Use `uv add` + `uv sync`, never bare `pip install` |
| Docker build fails at vLLM | vLLM build time overhead | Pass `--build-arg SKIP_VLLM_BUILD=1` |


## ROCm experiment environment

The MI325X recipes and frozen package overlays live in `experiments/`; start with
`experiments/README.md`. They reuse a ROCm container/interpreter rather than the
root CUDA lockfile. Do not install CUDA Torch/vLLM over that environment or edit
an existing `.venv`. Run commands with `uv run --no-project --python <interpreter>`
inside the wrapper and retain `NEMO_RL_PY_EXECUTABLES_SYSTEM=1` for Ray workers.
The wrapper derives its repository root; package/image/Automodel paths have
`NRL_*` overrides documented in the setup guide. Verify imports resolve to the
checkout being tested when reusing another checkout's dependency overlays.

Generate smoke inputs with `experiments/prepare_smoke_data.py`. The historical
renderer parity test additionally needs `prepare_reference.py`, which fetches a
pinned, checksummed source into an ignored directory. Never commit dependencies,
model/dataset caches, run outputs, checkpoints or credentials with these recipes.

Submit Slurm jobs from the repo root on verification/normal, and use the actual
allocated GPU count with cgroup-local device indices. Keep Ray/Triton/W&B scratch
paths job-specific. Multi-node launchers require routable head/node addresses.
The TML worker uses synchronous checkpoints; recovery requires complete model,
optimizer and dataloader state plus the shared run writer lock. Dedicated vLLM
ranks avoid the observed ROCm sleep/wake allocation failure. Validate transfers
with `verify_rccl_group.py` before launching distributed OPD on a new stack.

ROCm recipes using `_v2: false` must set
`policy.dtensor_cfg.checkpoint.model_save_format: null`; the exemplar's
`safetensors` setting applies to DTensor V2. Do not put `model_save_format` in
the top-level `checkpointing` block. Experiment worker overrides of
`_init_checkpoint_manager` must accept only `config_updates` and forward it by
keyword. The TML override sets `is_async=False` there. Verify both saving and
resuming in a GPU smoke run when adapting recipes to a newer NeMo/Automodel base.
For optional W&B fields absent from a smoke config, use Hydra's `++` override
syntax in launchers (for example `++logger.wandb.id=...`).

For coding evaluations, run model-produced programs only in the isolated grader
from `experiments/coding_opd/README.md`, never directly in the training container
or on the login host. These compute nodes support unprivileged Singularity
network/PID/filesystem isolation but do not provide Docker. Build the small
pinned grader image on the login node and run grading in a CPU Slurm allocation.
Disable host/home/configured bind mounts and clear inherited credentials. Run the
known-answer grader self-test first. The LCB checker needs stderr with a real
file descriptor for faulthandler: redirect output to `/dev/null`, not StringIO.
Use `uv --no-config` for the independent grader project so parent CUDA dependency
metadata does not leak into its lockfile. Keep dataset/checker revisions, image
hash, prompt formats, overlap thresholds and evaluation seeds in the run record.

Keep the explicit multi-node Ray head port outside the default worker range
10002–19999. The TML launchers use `25000 + SLURM_JOB_ID % 10000`; assigning a
head port inside the worker range fails startup before model initialization.
Check the full modulo range when deriving job-specific ports, not just the
current job's value.

When increasing an OPD rollout budget, validate the resolved policy/teacher
context limits, generation `max_model_len`, and packing budgets together. A
packed microbatch must fit the longest complete prompt plus its allowed rollout.
Run a long-sequence backward/checkpoint preflight before the full batch. For a
controlled comparison, start from the same SFT adapter with a fresh optimizer,
keep the old run separate, and persist the variant config path in monitor state
so automated retries/extensions cannot fall back to the shorter-rollout recipe.

Native MOPD uses Megatron teacher workers and NeMo Gym; a passing DTensor OPD
recipe does not validate that stack. Start with the GPU import probe in
`experiments/rocm_mopd`, then a tiny native training/checkpoint run. Preserve the
parent-pinned Bridge/Core/Gym revisions. ModelOpt needs installed distribution
metadata, so merely adding its source checkout to PYTHONPATH is insufficient.
The isolated `deps/pyproject.toml` pins ModelOpt and excludes container-provided
Torch/NumPy from resolution; sync it with `uv --no-config sync --all-extras`
inside the ROCm container. Never overlay CUDA Torch onto the working ROCm stack.
Use a writable job-local `AITER_JIT_DIR` when the home directory is read-only.
Imports through Mamba can query GPU properties, so a login-node import failure
is not a substitute for the allocated GPU probe. For selected pure OPD routing
unit tests, `pytest --confcutdir=tests/unit/algorithms` avoids the parent suite's
automatic Ray startup and whole-checkout upload; do not use that shortcut for
tests that need parent fixtures.

For ROCm Megatron, the CUDA-only `TORCH_CUDA_ARCH_LIST` guard must not run when
`torch.version.hip` is set. Before Hugging Face conversion, unset inherited
`NVTE_FLASH_ATTN`, `NVTE_FUSED_ATTN`, and `NVTE_UNFUSED_ATTN`: conversion constructs
an `auto` attention provider before applying policy overrides. Set
`policy.megatron_cfg.attention_backend: fused` for the tested AMD TE backend.
The one-GPU native SFT probe completed two forward/backward/optimizer updates
and synchronous checkpoints with this combination. This does not validate
Megatron generation or MOPD routing.

The pinned Gym source can use `NEMO_GYM_EXTRA_ROOTS` for experiment resource
servers. Use a zero-reward verifier for OPD smoke prompts; never execute model
coding responses in the training container. Validate Gym's complete server
config offline (including mandatory resource `domain`) before GPU launch.
`skip_venv_if_present` needs both `bin/python` and `bin/activate`; create fresh
job-local server environments with `uv venv --system-site-packages` and inherit
the pinned source/dependency overlay. The current Gym pin declares Python 3.13;
source-import success under the Primus image's 3.12 is only a compatibility
probe, not upstream-supported installation. SingleController checkpointing
with the in-order sampler requires `checkpointing.save_data_plane: true`.
Use a top-level `_override_` block when replacing the inherited teacher map;
otherwise OmegaConf retains the recipe's default teacher alias.

Ray's AMD accelerator manager honors `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES`
(even when it writes CUDA_VISIBLE_DEVICES). Megatron workers index devices by
`ray.get_gpu_ids()[0]`, so their `configure_worker` must set the HIP no-mask flag
on ROCm as well as the CUDA flag. Otherwise ranks on GPU 2+ see one device and
fail with invalid device ordinal. Keep this worker-specific; CPU Gym actors
should retain Ray's normal masking. Fresh Gym venvs resolve to the container's
base interpreter, not `/opt/venv` as their site-package parent; explicitly carry
`/opt/venv/lib/python3.12/site-packages` in PYTHONPATH along with the overlays.
Without it, Gym's resource server may import an incomplete wandb namespace and
fail to find `Histogram`, even though the driver's W&B logger works.

Native Megatron generation additionally needs TransferQueue at the root-pinned
revision and Quart/Hypercorn for its HTTP frontend. The isolated dependency
project records these while preserving the tested Ray overlay. Extend the GPU
probe to import actual Gym resource/launcher, TQ adapter, and HTTP server
packages, not just the lazy NeMo-RL integration module. Use synchronous
Megatron checkpoints to avoid requiring NVRx during the smoke. On ROCm, select
MCore's Torch sampler and unfused RoPE instead of FlashInfer. Eager generation
(`cuda_graph_impl: none`) does not need static KV pointers when recomputing the
cache. Report memory with PyTorch's HIP runtime API when AMD SMI is absent.
Gym's endpoint-readiness wait can hide an earlier setup exception inside the
parallel initialization executor; disable that redundant wait in the smoke
while retaining the trainer's own server startup/refit checks.

Use absolute checkpoint and log paths with `uv run` and Ray: automatic working
directory packaging can put a relative output path under node-local `/tmp`.
Override inherited `checkpointing.checkpoint_must_save_by` for the actual Slurm
allocation. SingleController can save a partial checkpoint and exit successfully
when this internal deadline is reached. Slurm `COMPLETED` does not prove the
requested update count finished. Inspect `latest_checkpoint_status.json` and
the saved `training_info.json`; resume known timeout exits only when checkpoint
progress was made. Keep the original output/W&B run ID separate from each new
Slurm job ID, and bound automatic retries.
Megatron resume validates the saved optimizer schedule, including weight decay
duration. Keep the scheduler horizon fixed across planned interruptions; when
deliberately extending a completed smoke, explicitly use the saved scheduler
configuration. Restore the dataloader and native data plane together.

For native MOPD routing audits, compare direct and routed teacher scores on the
same token batch shape. Packed BF16 forwards with different batch shapes can
differ numerically even for identical weights. Record that difference separately
from routing correctness. Check the student against an identical teacher before
training and repeat fixed probes after training to verify the teachers stayed
unchanged. Native smoke and resume passed on four MI325X GPUs with this stack.

The native generation HTTP server must receive the configured training
tokenizer's chat template; the checkpoint tokenizer may have a different
default. Check the returned prompt token IDs against training tokenization.
Refit-fed LoRA generation workers must attach adapters without trying to load a
pretrained checkpoint: their base and adapter weights arrive through refit.
Training/resume workers still use the normal PEFT checkpoint-loading hook.
Bridge's tensor-parallel LoRA path also reads NVIDIA TE distribution metadata.
On HIP, use MCore's module-aware TE version lookup for that Bridge check; AMD's
installed TE module can be present without a `transformer-engine` distribution.


Validate native LoRA exports separately from BF16 backend parity. A native
unmerged packed forward and an HF merged SDPA forward can exceed a mean
log-probability tolerance even for a correct export. The MOPD export validator
reconstructs all HF weights as base + (B @ A) * alpha/r in FP32 then rounds to
export dtype, requires exact tensor equality, and checks merged/unmerged FP32
probe equivalence. Retain native probe checks and record BF16 differences;
do not just increase tolerances after a failure. `--validate-merge` runs these
checks on an allocated GPU; `--reuse-export` avoids repeating conversion.
For independent completed training arms, submit evaluations concurrently and
queue grading/reporting with Slurm `afterok` dependencies. Use
`--kill-on-invalid-dep=yes` so failed prerequisites cannot strand pending jobs.
Persist job IDs immediately after each submission and refuse duplicate chains.
A tmux polling script records failures but does not wake an assistant; do not
represent it as unattended diagnosis of unknown errors.

Coding grading baselines must contain `model: student` records for every held-out
problem/repeat. An OPD evaluation directory's summary can include a reused SFT
baseline while its `graded.jsonl` contains only newly graded OPD records. Use the
original baseline grades (`coding_opd/run_eval_50305/graded.jsonl` for this pilot),
validate completeness before submitting work, and rebuild summaries from saved
grades after a reporting-only failure rather than regenerating responses.

For a completed native OPD run, extending `max_num_steps` alone is insufficient:
raise `max_num_epochs` to permit further dataloader passes and configure a scheduler
horizon at least as large as the new training budget. With a constant LR/WD,
`use_checkpoint_opt_param_scheduler: true` preserves saved optimizer/scheduler
state without changing the schedule. Copy the complete checkpoint, including
optimizer, dataloader, replay and data-plane files, into a separate continuation
root before training; keep original evaluated runs immutable. Verify optimizer
presence through `CheckpointManager.get_resume_paths` and exercise a GPU resume
past the original final epoch. Self-teacher identity audits apply only at step
zero: matching model names do not mean the resumed student's weights are still
the frozen teacher's weights. Check teacher invariance around resumed training.
Parameterize export/evaluation/report checkpoint steps together and keep their
outputs separate so generation-resume logic cannot reuse older responses.
The MOPD extension controller runs under Slurm, uses dependencies for downstream
stages, retries known infrastructure failures with bounds, and flags unknown
failures. Test its retry exclusions, target-checkpoint gate, baseline completeness,
and step-specific artifact paths before entrusting it with long runs.

At experiment closeout, export `experiments/rocm_mopd/requirements.txt` from its
own `deps/uv.lock` with `uv --no-config export --project experiments/rocm_mopd/deps
--locked --all-extras --no-dev --no-hashes --no-emit-project --output-file ...`.
Export the isolated coding grader lockfile separately. Keep the false-platform
Torch/NumPy/Ray overrides: the container and existing overlays supply those
packages. Do not replace the frozen original SFT/RL overlay manifests with a
combined environment freeze. Record resolved optimizer, LoRA initialization,
loss reduction and importance-correction settings in experiment recipes; shared
learning rates or algorithm names do not establish an identical training recipe.
