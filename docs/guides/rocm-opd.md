# OPD experiment recipes on AMD ROCm

The repository includes MI325X recipes for SFT, single-teacher OPD and native
multi-teacher OPD. These are tested cluster configurations, with explicit image,
model, dataset and source prerequisites. They do not replace the standard CUDA
installation instructions or establish support for every ROCm configuration.

## Runtime paths

| Path | Training stack | Recipe |
|---|---|---|
| Historical sampled-token OPD | DTensor V2, Torch 2.11/ROCm 7.2, dedicated vLLM generation | [Math recipe](../../TM_OPD_RECIPE.md), [coding recipe](../../experiments/coding_opd/README.md) |
| Native routed MOPD | Megatron/SingleController, Torch 2.12/ROCm 7.15, AMD TE 2.15, NeMo Gym | [MOPD recipe](../../MOPD_RECIPE.md) |

Use the [experiment setup guide](../../experiments/README.md) for container
wrappers, pinned overlays and Slurm launchers. The native overlay is locked in
`experiments/rocm_mopd/deps/uv.lock`; its `requirements.txt` export suppresses
Torch, NumPy and Ray so their tested container/overlay versions remain in use.
The coding grader has a separate NumPy 2.2.6 environment. The Gym source pin is
tested locally under Python 3.12 despite declaring Python 3.13.

## Native routing and checks

Each prompt selects one frozen teacher by agent alias. Shared teacher
checkpoints can be deduplicated; teacher distributions are not averaged.
The mixed math/coding configurations allocate two GPUs per role, using eight
GPUs for two distinct teachers or six for the shared-teacher control.

The ROCm worker paths preserve allocation-local GPU visibility, use the Torch
generation sampler/unfused RoPE, and obtain memory statistics from the HIP
runtime. LoRA refit workers receive base and adapter weights from the trainer.
The HTTP generation frontend receives the configured training chat template.

Recipe audits check teacher routing at identical batch shapes, finite changed
student probes, unchanged teacher probes, checkpoint restoration, template
parity and native-to-HF export correctness. Export validation reconstructs all
merged weights and compares merged/unmerged FP32 probes. BF16 backend differences
are recorded separately. None of these checks substitutes for downstream scoring.

Generated coding solutions execute only inside the isolated grader container,
not in the training process. The continuation controller verifies saved update
counts, retries known infrastructure failures with bounds, and advances export,
evaluation, grading and reporting. Unknown failures require investigation.

## Recorded results and interpretation

The single-teacher math recipe achieved **61.25% AIME24**, up from **54.58% SFT**
on 960 samples/model. The coding recipe achieved **39.57% LiveCodeBench**, up from
**33.71% SFT** on 700 samples/model.

The native 50-update MOPD experiment scored **33.71% coding and 55.00% math**;
its matched single-teacher control scored **35.14% and 54.17%**. That math screen
uses 120 samples and a matched SFT baseline of 59.17%. Paired confidence
intervals include zero for MOPD versus control on both domains. Native execution
is validated; a learning benefit is not established in this experiment.

Teacher checkpoints differ between the native arms. Relative to the historical
DTensor recipe, the native experiment also changes the adapter continuation,
loss reduction, optimizer and batch size. The records do not isolate teacher
count or establish that MOPD generally fails. Full counts, confidence intervals,
truncation measurements and artifact locations are in the linked recipes.
