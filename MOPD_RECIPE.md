# Multi-teacher OPD on ROCm: math/coding experiment

Completed October 3, 2026; project closed October 6. We ran native NeMo-RL
multi-teacher on-policy distillation on AMD MI325X and compared it with a
single-teacher control. Both reached 50 updates and passed routing, checkpoint,
frozen-teacher, and export checks. **Neither established a reliable improvement
over SFT in this mixed-domain experiment.** No 100-update extension was run.

This is a native Megatron/SingleController experiment, separate from the
successful DTensor math [single-teacher recipe](TM_OPD_RECIPE.md) and
[coding follow-up](experiments/coding_opd/README.md). It does not reproduce their
training objective or isolate teacher count from teacher quality.

Code, configs, manifests and results are in `experiments/rocm_mopd/` (`M/` below).
The [runtime README](experiments/rocm_mopd/README.md) covers setup and launch
commands. Cluster-local datasets, model weights, responses and credentials are
not included in Git.

## Models and datasets

| Role | Model or checkpoint |
|---|---|
| Student initialization | Local OpenThoughts3 Qwen3-8B-Base SFT, step 3000, merged HF weights |
| Math teacher | Math single-teacher OPD step 100, `inference_run_opd_49581_step_100` |
| Coding teacher | Coding 16K-rollout OPD step 100, `inference_run_opd_coding-16k-seed42-50507_step_100` |
| Single-teacher control | `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218` |
| Training prompts | 320 DeepMath + 320 TACO, interleaved 50/50 |
| Evaluation | 175 LiveCodeBench problems × 4 samples; 30 AIME24 problems × 4 samples |

The SFT base is `Qwen/Qwen3-8B-Base` at
`49e3418fbbbca6ecbdf9608b4d22e5a407081db4`, trained on 384,000 conversations from
`open-thoughts/OpenThoughts3-1.2M` at `61bcf9d4eb38b30295efc2021227a63cc5bb34c8`.
Its construction is documented in the single-teacher recipe. The math teacher
previously scored 61.25% on the 960-sample AIME evaluation; the coding teacher
scored 39.57% on the 700-sample coding evaluation. These results showed headroom
over SFT, not superiority to the Qwen3-8B control teacher.

Training sources are `zwhe99/DeepMath-103K` revision
`5cf055d1fe3d7a2eb19719ac020211469736ae44` and TACO `ALL/train` revision
`d593ed0a2becbbc952230bb89be09189bf1056dc`. `M/prepare_data.py` reads the prepared
historical math and coding sources, filters inputs above 1,024 tokens, reservoir
samples 320 per domain with seed 42, shuffles each reservoir, then alternates
math and coding. Every 32-prompt batch therefore has 16 of each.

The overlap screen checks all 205 evaluation questions: exact normalized match,
or at least four shared unique 13-word n-grams covering at least half of an
evaluation question's n-grams. No selected prompt matched. This cannot exclude
paraphrases or unknown pretraining exposure. Source indices and hashes are in
the generated `M/data/mixed_pilot_manifest.json`. Dataset SHA-256:
`aea1c32b0275898150e98fd8c858ca46f36954e43e8a2fdcbcbe8846d8ae9c9b`.

## 1. Student and teacher setup

Attach **fresh LoRA adapters, rank 128, alpha 32, dropout 0**, to the merged SFT
student. Native target modules are `linear_qkv`, `linear_proj`, `linear_fc1`, and
`linear_fc2`: 252 HF attention/MLP projections, excluding `lm_head`. A uses Xavier
initialization and B starts at zero. The base and teachers remain frozen.

Route `math_agent` prompts to the math teacher and `coding_agent` prompts to
the coding teacher. Each prompt uses **one** teacher; teacher distributions are
not averaged. Strict alias matching is enabled. The control maps both aliases
to Qwen3-8B, with shared-checkpoint deduplication producing one teacher group.

Each role uses TP=2. The routed arm uses **8 GPUs**: two each for student training,
generation, math teacher and coding teacher. The control uses **6 GPUs** because
its teacher is shared. Both use the same prompt stream and training settings.

## 2. Native OPD training

| Setting | Value used in both arms |
|---|---|
| Batch | 32 prompts × 2 student responses = 64 sequences/update |
| Budgets | 20 updates, then resume each checkpoint to 50 |
| Data exposure | 640 prompts at step 20; 1,600 at step 50, repeating the same 640 unique prompts |
| Generation | 16,384 new tokens; temperature 1; top-p 1; no top-k filtering |
| Context / packed microbatch budget | 17,472 tokens |
| Template | Historical Qwen3 thinking template from the single-teacher experiment |
| Optimizer | Native Adam, LR 1e-4, betas (0.9, 0.999), epsilon 1e-8, weight decay 0 |
| Schedule / clipping | Constant LR, no warmup, gradient clipping 1.0 |
| Batching | Sequence packing enabled; separate dynamic-batching flag disabled |
| Memory | Sequence parallelism and activation checkpointing enabled |
| Rollout freshness | In-order sampler, `max_lookahead_versions: 0` |
| Checkpointing | Synchronous every 5 updates, optimizer + dataloader + native data plane saved |

Training uses `OPDAdvantageEstimator` and native `ClippedPGLossFn` configured for
REINFORCE (`disable_ppo_ratio: true`), token-mean reduction and ICE-POP correction:

```text
A_t = stop_gradient(log p_teacher(token_t) - log p_student_preupdate(token_t))
w_t = exp(log p_student_preupdate(token_t) - log p_generation(token_t))
w_t = 0 when outside [0.2, 5.0]
loss = mean_over_valid_response_tokens(-w_t * A_t * log p_student_train(token_t))
```

Teacher scores are gathered at sampled tokens from full-vocabulary probabilities;
there is no top-k teacher approximation. The OPD estimator does not use the
inherited GRPO reward-normalization/group-baseline fields. The Gym resource
returns zero reward and never executes generated programs. Correctness tests
are used only during isolated downstream evaluation. PPO ratio clipping and a
separate reference-policy KL penalty are disabled in this configuration.

Configs: `M/pilot.yaml`, `M/pilot_control.yaml`, `M/extend_routed_50.yaml`, and
`M/extend_control_50.yaml`. Resolved checkpoint configs are the authoritative
record of inherited settings.

### Differences from the successful single-teacher recipes

| Setting | Historical DTensor OPD | This native experiment |
|---|---|---|
| Adapter | Continue SFT adapter, including `lm_head` | Fresh attention/MLP adapter on merged SFT |
| Gradient reduction | Sum over response tokens | Mean over valid response tokens |
| Loss | Unclipped sampled reverse-KL ratio objective | REINFORCE with ICE-POP correction |
| Adam beta2 / clipping | 0.95 / none | 0.999 / 1.0 |
| Batch | 512 prompts × 4 | 32 prompts × 2 |
| Prompt stream | Full domain-specific source | Repeated 640-prompt mixed-domain subset |

The native control makes the two native arms comparable. It does not make their
absolute learning behavior directly comparable to the successful DTensor runs.
The teacher checkpoints also differ between the two native arms, so this is not
an isolated test of teacher count. LoRA limits update rank, not weight distance.

## 3. ROCm execution, recovery and verification

Native training used `primus_v26.5-pytorch2.12-te2.15.sif`, Python 3.12, PyTorch
2.12 / ROCm 7.15, AMD Transformer Engine 2.15, Megatron and NeMo Gym. Evaluation
used the preserved Torch 2.11 / ROCm 7.2 and vLLM 0.25.1 ROCm environment.
Slurm partition/QoS: **amd-tw-verification / normal**. Training numerical dtype
settings were preserved.

Source revisions are recorded in `M/protocol.json`: Bridge `5ed97996`,
MCore `1e7598cb`, Gym `267305e2`, ModelOpt `43fd41a5`, and TransferQueue `c5161430`.
`M/deps/uv.lock` pins the added runtime dependencies;
`M/requirements.txt` is its generated export, excluding active Torch/NumPy/Ray
dependencies because the container/overlays provide them. The Gym pin declares
Python 3.13; use of its source under this Python 3.12 image is a tested local
compatibility configuration, not a claim of upstream installation support.

ROCm changes cover the CUDA-only architecture guard, Ray AMD device visibility,
Torch sampling and unfused RoPE, HIP memory diagnostics, AMD TE version lookup,
LoRA refit initialization, and the generation HTTP frontend's chat template.
Job-local caches, allocation-local GPU indices and absolute output paths avoid
cache permission and Ray working-directory problems.

GPU checks established:

- Native SFT, MOPD forward/backward, checkpoint save and optimizer/data restoration.
- Exact routed/direct teacher score equality on the same token batch shape.
- An initial identical-teacher check; finite changed student probes and exactly
  unchanged teacher probes after training. Identity is not expected after resume.
- HTTP prompt token IDs matching the training template.
- All 8,190,735,360 exported weights matching independent FP32 LoRA reconstruction
  followed by BF16 rounding; merged/unmerged FP32 probe errors below 0.0001.

BF16 comparisons across packed native and HF inference backends are recorded
separately. A mean-error threshold initially blocked a correct export; exact
weight checks and FP32 merge checks established conversion correctness. Maximum
native/HF BF16 probe error must remain below 0.3. These checks do not establish
accuracy gains or rule out every algorithmic issue.

The 20-update arms required timeout recovery. The 50-update continuations copied
complete step-20 checkpoints into separate directories, raised the epoch limit
to three, and restored the constant checkpoint scheduler. They preserved the
original evaluated runs. Training jobs **53206/53207** completed the additional
30 updates in **17h39m/17h41m**, concurrently on 8/6 GPUs.

CPU Slurm controller **53205** then advanced exports, evaluations, isolated
grading and W&B reporting without retries or manual intervention. Completion:
**October 3, 09:33:37 UTC**. The controller polls every 30 seconds, retries known
infrastructure failures with bounds, and detects three hours without a completed
training update. Unknown correctness failures stop the affected arm; a script
cannot wake an interactive assistant to investigate them.

## 4. Evaluation and results

LiveCodeBench uses `code_generation_lite` revision
`0fe84c3912ea0c4d4a78037083943e8f0c4dd505`, the 175 `test6.jsonl` problems, and
four samples per problem. AIME uses `HuggingFaceH4/aime_2024` revision
`2fe88a2f1091d5048c0f36abc874fb997b3dd99a`, 30 problems and four samples each.
The AIME SFT reference is the matching first four draws from the original
32-draw evaluation: **71/120**, not the full-run **524/960** baseline.

Both domains use temperature 0.6, top-p 0.95, top-k 20, min-p 0, a 32,768-token
total context budget, and seed `1234 + repeat * problem_count + problem_index`.
Training and evaluation share the historical template. Coding uses the pinned
LCB prompt formatter and last fenced-code-block extraction. AIME adds the
boxed-answer instruction and scores the last complete box with `math_verify`.
Report mean sample accuracy/pass@1, not pass@4.

Coding candidates run only in the isolated grader container: no network, home,
shared repository or credentials, with resource limits. The pinned checker is
`28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`; grader NumPy is 2.2.6. The SFT grading
reference is `experiments/coding_opd/run_eval_50305/graded.jsonl`. Baseline
completeness is checked before grading; OPD-only raw grades are not a baseline.

| Model | Coding correct / 700 | Coding accuracy | Math correct / 120 | Math accuracy | Coding truncation |
|---|---:|---:|---:|---:|---:|
| SFT baseline | 236 | 33.71% | 71 | 59.17% | 31.14% |
| Single teacher, 20 updates | 243 | 34.71% | 67 | 55.83% | 28.71% |
| MOPD, 20 updates | 239 | 34.14% | 72 | 60.00% | 34.71% |
| Single teacher, 50 updates | 246 | 35.14% | 65 | 54.17% | 38.00% |
| MOPD, 50 updates | 236 | 33.71% | 66 | 55.00% | 42.71% |

At 50 updates, MOPD minus control is **−1.43 percentage points** on coding
(paired problem-bootstrap 95% interval **[−3.57, +0.86]**) and **+0.83 points** on
math (**[−7.50, +9.17]**). MOPD minus SFT is **0.00 points** on coding
(**[−2.14, +2.00]**) and **−4.17 points** on math (**[−10.00, +0.83]**).
All these intervals include zero. Bootstrap comparisons resample problems,
not individual responses, with 10,000 draws.

MOPD's coding truncation rose by eight points between steps 20 and 50; 297/700
final responses had no extracted code. This identifies completion behavior as
an investigation target, not a demonstrated cause. Neither native arm met the
coding extension gate (at least two points over SFT with a positive lower CI).
**The project stops at 50 updates.** A future experiment should investigate
completion behavior and align the native optimizer/objective with the successful
recipe before treating more updates or more teachers as the solution.

## Saved artifacts

- Step-20 runs: `M/run_52376` (routed), `M/run_52732` (control).
- Step-50 runs: `M/run_extend50_routed`, `M/run_extend50_control`.
- Native checkpoints: `checkpoints/step_50/`; merged exports: `hf_step_50/`;
  PEFT adapter exports: `hf_adapter_step_50/` under each continuation run.
- Step-50 responses, grades and domain summaries: `evaluation_step_50/`.
- Tracked reports: `M/pilot_results.json`, `M/extension_50_results.json`,
  `M/protocol.json`, `M/extension_protocol.json`, and `M/final_results.json`.
- W&B: [20-step comparison](https://wandb.ai/rahular/opd-tests/runs/mopd-comparison-52376-52732),
  [50-step comparison](https://wandb.ai/rahular/opd-tests/runs/mopd-comparison-extend50_routed-extend50_control),
  [routed continuation](https://wandb.ai/rahular/opd-tests/runs/mopd-extend50_routed),
  [control continuation](https://wandb.ai/rahular/opd-tests/runs/mopd-extend50_control).

All project training and evaluation jobs are complete. Large local artifacts
are retained; no further experiment is scheduled.
