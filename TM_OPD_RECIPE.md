# Thinking Machines OPD reproduction on ROCm

Completed September 21, 2026. We trained a local Qwen3-8B-Base LoRA SFT student,
then continued its adapter with on-policy distillation from Qwen3-8B using
NeMo-RL. AIME24 mean sample accuracy rose from **54.58% to 61.25%** after
100 OPD updates; the teacher scored **76.25%**.

The target was the historical **rank-128 LoRA reasoning recipe** in
[Tinker Cookbook commit 330b73d](https://github.com/thinking-machines-lab/tinker-cookbook/tree/330b73d8d763b0f65bcf04e643cc6f082e002554/tinker_cookbook/recipes/distillation),
associated with the Thinking Machines on-policy distillation experiment.
The pinned source is also saved locally under
`experiments/tml_opd_replication/reference_330b73d/`.
This run used local NeMo-RL training, not the Tinker training service.

Code, experiment configurations, dependency overlays, and launch instructions are
in [experiments/README.md](experiments/README.md). Historical artifact paths below
refer to the original run workspace; generated data and checkpoints stay local.

## Models and datasets

Experiment and artifact paths below refer to the local NeMo-RL run workspace,
under `experiments/tml_opd_replication/` (abbreviated `E/` below).

| Use | Hugging Face resource | Pinned revision |
|---|---|---|
| Student base | `Qwen/Qwen3-8B-Base` | `49e3418fbbbca6ecbdf9608b4d22e5a407081db4` |
| Frozen teacher | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |
| SFT data | `open-thoughts/OpenThoughts3-1.2M` | `61bcf9d4eb38b30295efc2021227a63cc5bb34c8` |
| OPD prompts | `zwhe99/DeepMath-103K` | `5cf055d1fe3d7a2eb19719ac020211469736ae44` |
| Evaluation | `HuggingFaceH4/aime_2024` | `2fe88a2f1091d5048c0f36abc874fb997b3dd99a` |

Both training stages used a frozen base and LoRA on `q_proj`, `k_proj`,
`v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, and `lm_head`:
**rank 128, alpha 32, dropout 0**, covering 253 modules / 506 adapter tensors.
Adapter A used Kaiming uniform initialization and B started at zero.
Local initialization/training seed was 42. OPD continued the trained SFT
adapter directly, with a fresh optimizer; it did not merge and initialize
a second adapter. LoRA constrains update rank, not distance from the base.

## 1. Local SFT

The published SFT checkpoint could not be downloaded (archive returned 404;
the attempted storage copy required billing), so we recreated SFT locally.

- Read the 120 OpenThoughts3 parquet shards in sorted order; apply the historical
  streaming shuffle with **seed 0, buffer 384,000**, and materialize the first
  **384,000 conversations**. No second shuffle during training.
- Reproduce historical Qwen3 rendering and loss masks: mask message headers,
  supervise assistant bodies including the end token, and preserve the
  renderer's final-message-body rule. Add the thinking prefill when needed.
  Right-truncate each example at **16,384 tokens**.
- Train **3,000 updates**, global batch **128**. Materialized data contained
  5,691,525,930 tokens, including **5,584,671,396 supervised next-token targets**.
- Use AdamW with **LR 1e-3**, linear decay to zero, no warmup,
  betas **(0.9, 0.95)**, epsilon **1e-8**, weight decay **0**, no gradient clipping.
- Backpropagate **summed token cross-entropy**, matching the documented Tinker
  reduction; log mean token NLL. Large raw gradient norms consequently should
  not be compared directly with norms from token-averaged losses.
- Enable FlashAttention2 sequence packing with a **16,384-token microbatch
  budget**; disable activation checkpointing after memory/throughput validation.

Configuration: `E/sft.yaml`.
Data preparation and assumptions:
`E/sft_recipe.json`,
`E/prepare_sft_data.py`, `E/sft_data.py`, and `E/run_sft.py`.
Final logged training NLL was **0.90577**.

## 2. On-policy distillation

- Prepare **103,022 DeepMath prompts in original shard/row order**, without
  shuffling. Truncate the raw question to 1,024 tokens, decode it, then apply
  the historical Qwen3 template with assistant `<think>\n` prefill. Eight
  prompts required truncation; the longest rendered prompt was 1,034 tokens.
  No answer-format instruction was added to training prompts.
- Each update samples **512 prompts × 4 responses = 2,048 sequences** from
  the current student. Stop after **100 updates / 51,200 prompts / 204,800
  sampled responses**, rather than completing the entire dataset.
- Generate up to **4,096 new tokens**, temperature **1.0**, top-p **1.0**,
  no top-k filtering. The total training sequence limit is **5,184 tokens**.
- Take exactly **one synchronous optimizer update per rollout batch** and
  refresh inference weights before the next batch. No stale rollout reuse,
  correctness reward, format reward, group centering, or reward normalization.
- Use fresh AdamW with constant **LR 1e-4**, betas **(0.9, 0.95)**,
  epsilon **1e-8**, weight decay **0**, and no gradient clipping.
- Enable sequence packing for student training and teacher scoring, with
  **16,384-token microbatch budgets**. The separate `dynamic_batching.enabled`
  flag is **false** in this recipe; packing performs the batching.

We added `SampledReverseKLLossFn` to NeMo-RL and selected it using
`loss_fn.kl_type=sampled_reverse`, `sampled_token_reduction=sum`:

```text
advantage_t = stop_gradient(log p_teacher(token_t) - log p_rollout(token_t))
ratio_t     = exp(log p_train(token_t) - stop_gradient(log p_rollout(token_t)))
loss        = -sum_over_response_tokens(ratio_t * advantage_t)
```

Prompt/padding tokens are masked. The ratio is unclipped; teacher and rollout
probabilities are detached. This is the sampled-token reverse-KL policy-gradient
objective, with KL coefficient 1 and discount 0. Teacher probabilities are
normalized over the full vocabulary and gathered at sampled tokens.
**There is no teacher top-k approximation**; the earlier `k=20/64` experiments
are not this recipe. Logged loss/KL values are token means even though the
optimization loss is summed.

Configuration: `E/opd.yaml`.
Entry points: `E/prepare_opd_data.py`, `E/opd_data.py`, `E/run_opd.py`,
and `E/opd_multinode.sbatch`.

## 3. ROCm execution and validation

We used AMD Instinct **MI325X**, Slurm **amd-tw-verification / normal QoS**, Ray,
NeMo DTensor V2 / FSDP2, the pinned Automodel source (`1814c6c`), PyTorch 2.11
with ROCm 7.2, vLLM 0.25.1+rocm722, Transformers 5.12.1, and PEFT 0.19.1.
`E/container.sh` uses `primus_v26.2_moefix.sif` and the existing package overlays;
commands run through `uv run`. Training precision configuration was not changed.

SFT completed on **32 GPUs** (four replicas of eight shards). Its final recovery
job, **49502**, ran 17h46m54s; this excludes earlier failed jobs and preparation.
OPD used **32 GPUs**: 16 for student/teacher computation (two replicas of eight
shards) and 16 dedicated single-GPU vLLM replicas. Preflight **49581** completed
the first two full updates; **49582** restored that checkpoint and completed
updates 3–100 in **7h29m46s**. Typical OPD updates took approximately 4–5 minutes.

The material implementation fixes/checks were:

- **Logprob alignment:** explicitly insert zero prompt logprobs before message
  flattening; otherwise generated-token logprobs shift into prompt positions.
- **Adapter loading/refit:** use absolute shared checkpoint paths for Ray
  workers; form LoRA merge products in FP32 before casting inference weights.
  Verify all 506 loaded tensors, unchanged frozen weights, finite adapter
  updates, and exact checkpoint restoration.
- **Checkpoint reliability:** replace the experiment's asynchronous optimizer
  checkpointing after an `EADDRINUSE` failure with synchronous saves every five
  updates, including optimizer and dataloader state. Use a shared writer lock.
- **Evaluation startup:** explicitly request token-ID lists from Transformers 5,
  isolate Triton caches per worker, and use the GPUs actually allocated by
  Slurm. In the container, allocated GPUs use local contiguous indices; global
  Slurm indices must not be applied as a second device mask.
- **Verification:** five SFT parity/coverage tests, four sampled-loss/alignment
  tests, two configuration regression tests, GPU update audits, and checkpoint
  resume checks passed. Final training/rollout importance ratio was **0.999964**;
  mean absolute logprob difference was **0.02291**. These check implementation
  consistency; downstream evaluation measures task improvement.

`E/monitor_opd.py` tracked training and automatically submitted evaluations at
updates 20 and 100. The initial final evaluation failed when Slurm supplied
7 GPUs for an 8-GPU request. The monitor flagged it but did not retry evaluation
failures. After fixing allocation handling and device mapping, replacement
**50155** completed on four GPUs on September 21 in **1h04m59s**.

## 4. AIME24 protocol and results

Evaluate all **30 problems with 32 independent samples each (960 total)**.
Use temperature **0.6**, top-p **0.95**, top-k **20**, min-p **0**, a **32,768-token
total context limit** (generation budget = limit minus prompt length), and
seed `1234 + repeat * 30 + problem_index`. Use the historical thinking template
and append: “Please reason step by step, and put your final answer within
`\boxed{}`.” Stop on end-of-text/end-of-message tokens.

Score the last complete boxed expression with `math_verify`; missing boxes
count as incorrect. Report **mean sample accuracy / pass@1**, not pass@32.
For student evaluation, verify the adapter and pre/post-merge FP32 logits,
then export a separate BF16 inference model. Keep the original adapter intact.
Changing evaluation GPU count preserves the repeats, prompts and sample seeds.

| Model/checkpoint | Correct / 960 | Accuracy | Change from SFT | Truncated responses |
|---|---:|---:|---:|---:|
| Local SFT, step 3,000 | 524 | 54.58% | — | 18.54% |
| OPD, step 20 | 582 | 60.63% | +6.04 pp | 23.96% |
| OPD, step 100 | 588 | **61.25%** | **+6.67 pp** | 13.02% |
| Qwen3-8B teacher | 732 | 76.25% | +21.67 pp | 5.52% |

Most of the observed improvement appeared by step 20. The additional 0.625 pp
at step 100 is small; this single run does not establish a reliable late-stage
gain. The final problem-bootstrap 95% interval is **47.60–74.27%**, reflecting
the small 30-problem benchmark. Final training sampled KL was **0.05687**.

The historical recipe reported approximately 55% after SFT and 65% after OPD.
Our SFT baseline was close, and OPD improved it, but **we did not reach 65%**.
This is not an exact numerical reproduction: the original adapter export was
unavailable; alpha/initialization, dropout, clipping, weight decay and local
seeds include documented assumptions; historical server internals and exact
AIME decoding settings were not published in the pinned recipe. See
`E/sft_recipe.json` and `E/manifest.json` for provenance and assumptions.

## Saved artifacts

- SFT adapter: `E/run_49416/checkpoints/step_3000/policy/weights/model/`.
- OPD adapters: `E/run_opd_49581/checkpoints/step_{20,100}/policy/weights/model/`;
  each checkpoint also contains optimizer, dataloader and resolved configuration.
- Evaluation outputs: `E/run_aime_49575/` (SFT/teacher), `E/run_aime_49595/`
  (step 20), `E/run_aime_50155/` (step 100). Raw responses remain local.
- Combined results:
  `E/run_opd_49581/final_comparison.json`.
  Code/config snapshot and hashes: `E/run_opd_49581/source_snapshot/`.
- W&B: [SFT training](https://wandb.ai/rahular/opd-tests/runs/tml-sft-r128-49416),
  [OPD training](https://wandb.ai/rahular/opd-tests/runs/tml-opd-r128-49581),
  [step-20 evaluation](https://wandb.ai/rahular/opd-tests/runs/tml-aime24-opd-run_aime_49595),
  [final evaluation](https://wandb.ai/rahular/opd-tests/runs/tml-aime24-opd-run_aime_50155).
  Evaluation uploads contain summaries, not the full generated responses.

All training and evaluation jobs are complete.


## Completed follow-up experiments

The [coding-domain single-teacher experiment](experiments/coding_opd/README.md)
reached **39.57% LiveCodeBench pass@1**, versus **33.71% SFT**, after 100 updates
with 16K rollouts. The [native mixed-domain MOPD recipe](MOPD_RECIPE.md) completed
a matched 20/50-update comparison but found no reliable accuracy benefit. Its
adapter, optimizer, batch size and objective differ from this historical recipe;
these experiments should not be treated as an isolated teacher-count comparison.
