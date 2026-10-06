# Coding-domain single-teacher OPD

**Complete.** The 100-update, 16K-rollout run improved coding accuracy from
33.71% to 39.57%. All training and evaluations are finished. The mixed-domain
follow-up is documented separately in [MOPD_RECIPE.md](../../MOPD_RECIPE.md).

Test whether the successful math SFT → sampled-token OPD recipe transfers to
Python code generation. Start from the **pre-OPD SFT student**, not the math OPD
checkpoint. This is a local controlled experiment, not a LiveCodeBench leaderboard
reproduction or an exact replication of a published coding experiment.

## Models and objective

`protocol.json` pins the Qwen3-8B-Base student, its existing rank-128 OpenThoughts3
SFT adapter/inference export, and the Qwen3-8B teacher. The student saw 384,000 SFT
examples. The coding pilot restores that same adapter and uses a fresh AdamW
optimizer: LR 1e-4, constant schedule, betas 0.9/0.95, eps 1e-8, no weight decay
or gradient clipping. All original precision settings are inherited unchanged.

Keep 512 prompts × 4 student samples/update, 4096 generated tokens, temperature 1,
unrestricted training sampling, historical Qwen3 thinking renderer, summed
sampled-token reverse KL, and sequence packing. This objective does not use a
top-k teacher distribution (`topk_logits_k=0`). Coding unit tests are used only
for evaluation, not training rewards. Training prompts are shuffled with seed 42.
The initial run requests 16 GPUs on two nodes: eight training/teacher GPUs and
eight dedicated generation GPUs, verification partition/normal QoS.

## Data and overlap audit

- TACO `ALL/train`, revision `d593ed0a2becbbc952230bb89be09189bf1056dc`:
  25,443 source rows; 663 duplicate normalized questions removed; 1,288 prompts
  excluded for exceeding 1024 raw tokens; **23,492 retained**. Long specifications
  are filtered, never truncated. No detected evaluation overlap. The rendered
  maximum is 1,034 tokens, fitting the 5,184-token training context with 4,096
  generated tokens. Training and evaluation use the same generic coding format.
- LiveCodeBench `code_generation_lite`, revision
  `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`, `test6.jsonl`: **175 problems**, dated
  January 4–April 6, 2025. All remain after the SFT overlap audit.
- Audit the actual 384,000 tokenized SFT inputs, rather than assuming the SFT
  corpus is disjoint. A near match shares at least four unique normalized
  13-word n-grams covering at least half of the evaluation question's n-grams.
  **Zero matches** were found. Apply the same screen to TACO and deduplicate
  normalized training questions. This is a text-overlap screen; it cannot exclude
  paraphrases or unknown base-model pretraining exposure.

## Evaluation and decisions

Four independently seeded samples/problem, temperature 0.6, top-p 0.95, top-k 20,
32,768 total tokens. Student and teacher have identical prompts and budgets.
Extract the last fenced code block, matching the pinned LCB generic extractor.
Grade with the public and private tests using the official checker at
`28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`, Python 3.12, NumPy 2.2.6,
six seconds/test, and the checker's `(6+1)*test_count+5` outer time limit.
Missing code, truncation without a correct solution, and runtime failures count
as incorrect. Report mean pass@1 over all four samples, not pass@4.

The baseline gate, fixed before scoring, requires a teacher advantage of at least
five percentage points and a paired problem-bootstrap 95% CI lower bound above
zero (10,000 resamples, seed 20260922). Otherwise inspect a 32B teacher before
training. The 20-update pilot extends to 100 only if its gain over the matched SFT
baseline is at least two points and its paired CI excludes zero. Retain every
five-update checkpoint; evaluate the preselected endpoints, not a retrospectively
chosen best checkpoint. A positive pilot is provisional: subsequent work should
include a second training seed, an SFT control, and secondary coding benchmarks
before claiming broad transfer. The monitor automatically runs a 120-sample AIME
regression screen after the final coding endpoint, paired against the matching
first four original SFT draws (71/120 correct). This small check has limited power.

## Executable-code isolation

`grader/build.sh` builds a small Docker image on the login node and exports a
Singularity image. The compute nodes do not need Docker. Each candidate runs in
its own unprivileged user/network/PID/filesystem namespaces with a clean
environment, no external network, no home or shared repository mounts, and a 4GB
address-space limit. Only the candidate and its tests are passed on stdin. The
upstream reliability guard is additional protection, not the sandbox itself.
The CPU Slurm allocation owns all grading work. Known correct/incorrect/syntax/
timeout/function-call examples must pass the grading self-test before evaluation.

The pinned upstream checker remains in ignored build inputs, with its original
source available in `reference_lcb`; it is not copied into tracked project code.
`data/grader.sha256` records the built image. The image contains no credentials.

## Running

From the repository root, set the ROCm overlay paths as documented in the parent
experiment setup. Dataset downloads and image construction run on the login
node; preprocessing, generation, and grading run under Slurm.

```bash
uv run --no-project --python /usr/bin/python3 experiments/coding_opd/prepare_sources.py
bash experiments/coding_opd/grader/build.sh
uv --no-config sync --project experiments/coding_opd/grader --all-extras \
  --python /shared_silo/scratch/rahul.aralikatte@amd.com/prime-rl/.venv/bin/python
sbatch experiments/coding_opd/prepare.sbatch
# Wait for the data audit to finish before interpreting any scores.
sbatch experiments/coding_opd/evaluate.sbatch
sbatch experiments/coding_opd/grade.sbatch --input experiments/coding_opd/run_eval_JOB_ID
```

The baseline array uses one eight-GPU allocation/model. Grading can start during
generation and resumes existing complete records. `track.py` logs counts and
final scalar scores to W&B group `coding-opd`, without code/checkpoint artifacts.

`monitor.py` runs persistently via `uv run --no-project --python /usr/bin/python3
-m experiments.coding_opd.monitor --state <state.json>`. It holds a single-writer
lock, records a heartbeat, applies the gates above, submits training/evaluation,
and resumes known infrastructure failures from complete checkpoints (up to three
retries). Unknown failures, missing audits, nonfinite failures, or ambiguous
submission responses require inspection. The shared training launcher also holds
a writer lock for the run. It never changes or cancels unrelated jobs.

The new `data.task_name` setting routes coding examples to their zero-reward
environment; omitting it retains the math recipe's `deepmath` behavior.

## Initial execution

- Baseline array: **50305**, eight GPUs/model, 700 samples/model.
- Final prompt preparation/audit: **50386**, passed (first audit: 50344).
- Isolated grading: **50373**, five known-answer checks passed.
- W&B: https://wandb.ai/rahular/opd-tests/runs/coding-baseline-50305
- Validation: four existing OPD loss tests, paired-scoring/decision-gate checks,
  all 175 prompts matched against the pinned upstream formatter, config/dataset
  checks, and the 120-sample AIME reference check passed.
- Baseline complete: **SFT 236/700 = 33.71%; teacher 346/700 = 49.43%**.
  Paired improvement: **15.71 percentage points**, 95% interval **11.71–20.00**.
  This passes the predeclared pilot gate. See `baseline_results.json`.
- Student token-limit fraction **31.14%**, teacher **3.29%**. Mean generated
  lengths 18,676 and 14,696 tokens. Completion/formatting is an important part of
  the gap; these scores alone do not establish a difference in reasoning ability
  conditional on producing a finished answer.
- Pilot **50450** requests 16 GPUs and 20 updates. The initial launch (50449)
  failed before training because its Ray head port overlapped the default worker
  port range; both multi-node launchers now choose ports in 25000–34999.
  Run ID `coding-seed42-50305`. Completed 20 updates in 2h45m42s; all adapter/base
  audits passed. Mean update time 481 seconds. No extension to 100 was triggered.
- Local run directories hold generated answers, grades, provenance, and
  summaries and are excluded from Git.

## Pilot result

The 20-update pilot did **not** demonstrate transfer gains: LiveCodeBench
232/700 = **33.14%**, compared with SFT 236/700 = **33.71%**. Paired delta
**−0.57 percentage points**, 95% interval **[−2.71, +1.57]**. Evaluation
token-limit failures rose from **31.14% to 48.57%**, and mean output length
rose from 18,676 to 21,020 tokens. All jobs in the 4K pilot completed.

The 120-sample AIME screen was 64/120 = **53.33%**, versus the matched SFT
71/120 = **59.17%**. Delta −5.83 points, paired 95% interval [−15.00, +3.33];
this small check does not establish a regression.

Training loss declined from 0.4378 to 0.1312, with frozen-base/changed-adapter
audits passing on all eight training ranks. However, 1,638/2,048 first-update
responses and 1,675/2,048 final-update responses never closed the thinking block
within the 4,096-token training rollout. Only 412 and 368 respectively contained
a code fence. The working hypothesis is insufficient exposure to completed
solutions during OPD; this has not been isolated from learning-rate, objective,
or task-distribution effects. The controlled longer-rollout experiment below tested that hypothesis.

See `pilot_results.json` and the
[checkpoint-20 W&B evaluation](https://wandb.ai/rahular/opd-tests/runs/coding-seed42-50305-eval-20).

## Longer-rollout follow-up

`opd_16k.yaml` starts again from the original SFT adapter and a fresh optimizer,
with **16,384 generated tokens**. It preserves the 512×4 batch, seed 42, prompts,
rank-128 adapter, LR 1e-4, summed loss, teacher and precision settings. Context and
packing budgets increase to **17,472**, enough for the longest 1,034-token prompt
plus the rollout. The training data SHA-256 matches the original run exactly.
The resolved config diff contains only sequence-length/budget changes (including
inherited inactive dynamic-batching settings).

The four-GPU one-update preflight, job **50507**, passed long-sequence generation,
backpropagation, checkpointing and frozen-base/changed-adapter audits. Its eight
responses averaged 13,512 generated tokens; loss was finite (0.5272). Its weights
are not reused. The persistent monitor launched a fresh **16-GPU, 20-update**
pilot, job **50509**, run ID `coding-16k-seed42-50507`. It started on nodes g49/g50
at 14:26 UTC on September 22, 2026, in `amd-tw-verification` with normal QoS. State and heartbeat live under
`run_coding-16k-seed42-50507/`, tmux window `Coding-16K-Watch`. The monitor persists
the config path so retries and an approved-by-metrics extension keep the 16K
settings. Evaluation also reports a paired comparison against the 4K checkpoint,
in addition to the original SFT baseline. The same extension gate and AIME screen
apply. No new evaluation prompts or decoding settings are introduced.

See `long_rollout_protocol.json`. This intervention also increases the number
of tokens contributing to the summed gradient; it is not a token-budget-matched
comparison. The 20-update result is recorded below.

Preflight evidence: `long_rollout_preflight.json`. Training metrics: [W&B](https://wandb.ai/rahular/opd-tests/runs/coding-16k-seed42-50507).

### 16K pilot result (20 updates)

LiveCodeBench: **263/700 = 37.57%**, versus SFT **236/700 = 33.71%**.
Paired gain **+3.86 percentage points**, 95% problem-bootstrap interval
**[+1.43, +6.57]**. Compared with 4K OPD, the gain is **+4.43 points**,
95% interval **[+1.71, +7.29]**. Evaluation truncation fell to **23.86%**
(SFT 31.14%; 4K OPD 48.57%). This supports the completion-coverage hypothesis,
although the summed-gradient token-count difference remains a confound.

Job 50509 completed all 20 updates and passed the parameter audit; loss fell
from 0.4800 to 0.1092. Mean update time was 35.5 minutes. Evaluation 50531 and
isolated grading 50532 completed successfully. The predeclared extension gate
passed, and the monitor resumed toward 100 updates in job **50549** on 16 GPUs.
The extension and its coding/AIME evaluations completed; the final results are below.

See `long_rollout_results.json` and the
[20-step evaluation](https://wandb.ai/rahular/opd-tests/runs/coding-16k-seed42-50507-eval-20).

### Final 16K result (100 updates)

Training and both final evaluations completed on September 25, 2026. Coding
accuracy reached **277/700 = 39.57%**, compared with **33.71% SFT** and **37.57%
at step 20**. The paired gain over SFT is **+5.86 percentage points**, with a
95% problem-bootstrap interval **[+3.14, +8.71]**. Final truncation was **17.71%**,
and mean generated length was 17,473 tokens. The final loss was 0.0714; the
frozen-base/changed-adapter audit passed.

The matched 120-sample AIME screen scored **79/120 = 65.83%**, versus SFT
**71/120 = 59.17%**. Difference **+6.67 points**, paired 95% interval
**[0.00, +13.33]**. This small screen shows no observed regression; it does not
establish a robust math improvement.

Three 12-hour allocation timeouts were automatically recovered from complete
checkpoints. Final training job 51600, coding evaluation 51668, grading 51669,
and AIME job 51701 all completed successfully. The monitor is complete and no
jobs remain active. This is a single-seed coding result; the longer-rollout
comparison also changes the number of tokens contributing to the summed loss.
See `long_rollout_results.json` and the
[final coding evaluation](https://wandb.ai/rahular/opd-tests/runs/coding-16k-seed42-50507-eval-100).
