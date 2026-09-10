> Heading-flow configuration cleanup: historical experiment names below are preserved for provenance. Current supported configs are the six standalone variants in [Heading flow policies](heading_policies.md). Retired YAML files are in the cleanup backup.

# Heading-conditioned flow policy: verified result

Objective: one general flow-matching policy checkpoint with approximately
70–80% LIBERO-10 success. Predicted heading remains both explicit conditioning
and a noise-prior parameter. No task-specific geometry, phase rules, oracle
future actions, or per-task checkpoint selection is allowed.

## Starting point

`HeadingZeroFlowPolicy` wraps `SharedHeadingFlowPolicy`: the shared observation
encoder predicts a unit XY heading and a motion-validity probability. Those three
numbers are appended to each observation token. A Gaussian action chunk is
unnormalized, rotated so its XY resultant matches the predicted heading, then
normalized. The source heading is detached; conditioning and auxiliary heading
supervision train jointly. Training and inference use the same source rule.

The rectified-flow objective is unchanged:

```
x0 ~ p(source | predicted_heading(observations))
t ~ Uniform(0,1)
xt = (1-t) x0 + t action_normalized
loss = MSE(v(xt,t,observations,predicted_heading), action_normalized-x0)
       + 0.1 heading_cosine_loss + 0.1 validity_BCE
```

Inference integrates the learned velocity with ten Euler steps, predicts sixteen
actions and executes eight. Future actions are used only for offline supervision.

The old source is supported on an exact-angle surface. Its validity probability
does not quantify heading prediction accuracy. The original 76×76 image crop also
retains only 35% of a 128×128 image. These motivate the comparisons below, but are
not evidence of a success-rate improvement.

## Initial comparisons

All runs use split seed 42, 450 training/50 validation demonstrations, train-only
normalization, batch 64, seed 42, EMA, full training windows, 500-update warmup and
1e-4 learning rates. The initial budget is 61 full epochs. Validation and checkpoint
saving occur every 5 epochs; simulator evaluation is separate. No inference DNO
has been added because the existing DNO scorer needs task-specific subgoals.

| Run directory under `output/heading_goal/` | GPU | Velocity backbone | Vision | Heading source |
|---|---:|---|---|---|
| `transformer_wide` | 0 | original Transformer | learned ResNet18, 116px crop | original exact angle |
| `unet_wide` | 1 | FiLM temporal U-Net 128/256/512 | same ResNet18 | original exact angle |
| `unet_gaussian` | 2 | same U-Net | same ResNet18 | full-rank Gaussian |
| `dino_gaussian` | 3 | FiLM temporal U-Net | frozen DINOv2-S/14, 4×4 pooled spatial features | full-rank Gaussian |

The new Gaussian uses raw-coordinate unit direction `d` and its perpendicular
`d_perp`:

```
raw_xy = s * ((0.5 + epsilon_parallel) * d + 0.5 * epsilon_perp * d_perp)
s = RMS(reciprocal(action_normalizer.scale[:2]))
source_xy = action_normalizer(raw_xy)
```

Other action channels remain Gaussian. Positive parallel and perpendicular
standard deviations retain full support. This changes the noise distribution
through both its mean and covariance; rotating an isotropic Gaussian alone would
not do so. The checkpoint includes all frozen visual weights, normalizers and the
heading-label RMS. The DINO policy can load without its original pretraining file.

The fourth run draws motivation from Cocos, which studies frozen visual features
and conditional Gaussian sources for straight flow matching. It uses a different,
smaller observation compression here and is not a reproduction of its results.
Primary reference: https://papers.nips.cc/paper_files/paper/2025/file/ad6363efb7af02f7db13d087c7e649bd-Paper-Conference.pdf

## Evaluation protocol

`scripts/eval_heading_policy.py` snapshots one complete checkpoint, records its
SHA256 and uses the same EMA policy for every task. Episodes start from official
LIBERO states, observations are refreshed after ten settling steps, and success
is checked after every executed action. Each episode has independent simulator
and policy-noise seeds. Missing/error episodes prevent a benchmark SR from being
reported. The manifest includes task/state mapping, software versions and source
hashes. No action candidates or checkpoints are selected inside an episode.

There are 50 official states per task. Development pilots use indices 0–9. Reserve
indices 10–49 for the final 400-episode held-out assessment after selecting one
checkpoint. A standard 500-episode report may additionally be produced for that
same checkpoint; it must disclose overlap with the development pilot states.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/fm_dno /venv/oat/bin/python \
  scripts/eval_heading_policy.py -c CHECKPOINT -o NEW_OUTPUT_DIRECTORY \
  --n-per-task 10 --init-start 0 --seed 20260910 --workers 4
```

The selected final assessment uses `--n-per-task 50 --init-start 0 --seed 20260911`
and eight workers. The reserved 400 episodes are extracted from that same run
using indices 10–49, with no reruns. Episodes allow 550 physical control steps
after ten settling steps. This horizon is stated explicitly because LIBERO
evaluation protocols differ. Training/validation action MSE is not task success.

## Runtime and integrity notes

Jobs are managed by supervisor as `fm-heading-transformer`, `fm-heading-unet`,
`fm-heading-gaussian`, and `fm-heading-dino`. Their foreground launcher is
`scripts/run_heading_experiment.sh`. All jobs are restricted to GPUs 0–3.

The shared Python environment has another editable package named `oat` from
`/workspace/past_action`. The launcher now explicitly sets this checkout first
in `PYTHONPATH`; `run_workspace.py` also inserts its root first. The initial
mixed-import Transformer run was stopped and moved to
`transformer_wide_INVALID_import_path`; its metrics are excluded. U-Net failed
starts before the import fix remain in console logs and produced no checkpoints.

Historical success rates in RUNLOG/RUN_SUMMARY refer to another machine and do not
establish performance of these new models. The completed assessment below is the evidence for the achieved target.

## Completed implementation checks

- 49 focused integration tests passed before the final positive-noise guard; the
  Gaussian-prior suite subsequently passed all 9 tests including that guard.
- U-Net/source/backbone regression suite: 95 tests passed.
- Real eight-step simulator checks completed all 10 tasks without errors for the
  Transformer, Gaussian U-Net and DINO Gaussian policies. These are smoke tests,
  not success-rate estimates.
- Repeating the Transformer smoke with 1 versus 2 workers gave identical initial
  state and executed-action hashes for every episode; see
  `output/heading_goal/eval_smoke/parallel_reproducibility.json`.
- DINO native patch features match the official cached model exactly on CPU;
  full-policy checkpoint restoration works with the pretraining path absent.

## First complete development pilots (checkpoint epoch 5)

Each is one EMA checkpoint evaluated on the same 100 full-horizon episodes:
10 official initial states per task, indices 0–9, policy seed 20260910.

| Model | Successes | SR | Episode Wilson 95% interval |
|---|---:|---:|---:|
| Transformer + exact heading prior | 30/100 | 30% | 21.9–39.6% |
| U-Net + exact heading prior | 58/100 | 58% | 48.2–67.2% |
| U-Net + Gaussian heading prior | 64/100 | 64% | 54.2–72.7% |
| DINO + Gaussian heading prior | 58/100 | 58% | 48.2–67.2% |

All four completed without errors. The Gaussian-vs-exact comparison is matched
for backbone and training recipe. These small development pilots do not establish
a statistically reliable 6-point prior benefit, and none establishes the final
target. DINO did not improve the first-milestone success rate over the ResNet U-Net.
Full manifests, checkpoint hashes, task results and episode records live in
`output/heading_goal/pilots/`; aggregate pointers are in
`output/heading_goal/epoch5_pilot_results.json`.

The fixed 640-window validation diagnostic gives prefix action MSE
0.06210/0.05182/0.05059/0.05176 for Transformer/U-Net/Gaussian/DINO respectively.
Predicted heading MAE is 29.18/29.78/29.95/28.24 degrees. Gripper MSE dominates
raw channel error, but simulator gripper commands use sign; raw MSE alone is
insufficient evidence for changing the gripper model.

GPU workspace continuation was also tested with two updates before and two
after checkpoint restoration: saved epoch 1, next epoch 2, optimizer/EMA update
counts both 4. Evidence: `output/heading_goal/resume_gpu_smoke/verification.json`.

## Checkpoint selection (epoch 15)

All four second-milestone pilots completed 100 episodes with zero errors, using
exactly the same development state indices 0–9 and seed 20260910.

| Model | Successes | SR |
|---|---:|---:|
| Transformer + exact heading prior | 62/100 | 62% |
| U-Net + exact heading prior | 82/100 | 82% |
| U-Net + Gaussian heading prior | 82/100 | 82% |
| DINO + Gaussian heading prior | 70/100 | 70% |

The epoch-15 Gaussian U-Net was selected for final assessment. Both U-Nets tie;
the full-rank source is a method preference, not evidence of superior SR. There
is no task-specific model selection. The final export contains the selected EMA
weights and all inference configuration/normalization data, with no optimizer
state or unused online model. All 548 exported tensors exactly match the source
EMA, and fixed-noise inference through the public policy loader matches exactly.
The export SHA-256 is
`cbf4a130f218b2f87a67931bf066ed26915bc77ea7770ab392d9b9bac6ce50f5`.

The consolidated implementation regression suite passed 158 tests. A separate
18-test result-integrity suite checks completed-run reporting and refusal of
incomplete, duplicate, mismatched or error episode records.

## Final verified result

The selected epoch-15 EMA achieved **76.4% (382/500)** across all ten LIBERO-10
tasks using one fixed checkpoint, 50 official initial states per task, a
550-action limit, and seed 20260911. All 500 episodes completed without errors.
On indices 10–49, excluded from development rollout pilots, the same final run
achieved **74.5% (298/400)**. These reserved states are reserved from development
evaluation; this is not a claim that their configurations never occur in training
demonstrations. The full 500 includes 100 initial states used in development
pilots, rerun here with the final seeds.

Files under `output/heading_goal/final/`:

- `policy.ckpt`: selected inference-only policy (approximately 230 MB), with
  embedded normalization and exact epoch-15 EMA tensors.
- `RESULTS.md`: per-task full/reserved counts, method description and reproduction
  commands. `report.json` includes recomputed integrity checks and intervals.
- `eval_500/manifest.json` and `eval_500/episodes.jsonl`: complete evaluation
  protocol, seeds, initial-state/checkpoint/code hashes and raw episode records.
- `eval_500/videos/`: ten videos, initial-state index 0 for each task selected
  before evaluation; both successes and failures are retained.
- `success_by_task.png`: per-task comparison of full and reserved results.

The report generator verified unique episode identities and state indices,
recorded initial-state hashes, both episode seeds, exact checkpoint hashes and
agreement with every saved aggregate. Implementation and result-integrity tests
passed (158 + 18 = 176). All experiment training and watcher jobs were stopped
after the target was verified; the final evaluator exited normally.

Training uses `oat/config/train_flowpolicy_heading_gaussian.yaml` with the launch
command below. The selected checkpoint is epoch 15 (zero-based), after sixteen
full epochs; later checkpoints were not used for the final assessment.

```bash
scripts/run_heading_experiment.sh 2 reproduce_gaussian train_flowpolicy_heading_gaussian
```

The launcher is a foreground command intended for supervisor. Use a fresh run
name; the recorded task and data configuration is included in the checkpoint.
The original `flow_policy_heading_zero.py` remains available for matched exact
heading-prior comparisons. No inference DNO, task-specific geometry, scripted
stages, privileged future actions or per-task policy routing is used.

This is a general flow-policy architecture verified on LIBERO-10. Transfer to
other benchmarks and training seeds has not been measured. The matched exact
prior and Gaussian prior tied in the selection pilot, so the result supports
the complete policy recipe, not a claim that the Gaussian prior alone caused
the improvement.

## Additional evaluation: epoch 20 and latest

At the user's request, epoch 20 and the latest saved Gaussian U-Net checkpoint
were evaluated using exactly the epoch-15 final protocol: 500 official initial
states, seed 20260911, 550 policy-action steps after ten settling steps, and one
EMA policy across all tasks. Both files contain epoch 20/global step 40718,
identical resolved inference configurations and all 548 exactly equal EMA
tensors. The job stopped during epoch 24, before the next scheduled save at 25;
there is no later saved checkpoint. One evaluation therefore covers both names.

| Checkpoint | Full 500 | Previously reserved indices 10–49 |
|---|---:|---:|
| Epoch 15 | 382/500 (76.4%) | 298/400 (74.5%) |
| Epoch 20 / latest | 380/500 (76.0%) | 301/400 (75.25%) |

The epoch-20 run has zero evaluation or video errors. Its full-suite result is
0.4 percentage points below epoch 15 (two fewer successes). With matched episode
seeds, 331 episodes succeed for both policies, 69 fail for both, 51 succeed only
at epoch 15, and 49 succeed only at epoch 20. These are descriptive results;
indices 10–49 now also participate in this later-checkpoint comparison and are
not a fresh holdout.

The complete comparison, per-task counts, episode records, checkpoint audit,
exported policy and ten preselected videos are in
`output/heading_goal/epoch20_latest/`. The report is `RESULTS.md`, with machine
readable details in `comparison.json` and `verification.json`. The new comparison
reporter passed 24 integrity tests, and the supervised evaluation exited normally.
The original epoch-15 final artifact and assessment remain the recorded selected
result from the initial goal.

## Matched 500-episode backbone comparison

At the user's request, the epoch-15 Transformer and U-Net with the **same original
exact-heading prior** were evaluated on 500 paired episodes each. This isolates
the backbone configuration from the Gaussian-prior change.

| Backbone | Successes | SR | Velocity parameters |
|---|---:|---:|---:|
| Transformer | 315/500 | 63.0% | 4,784,903 |
| U-Net | 376/500 | 75.2% | 12,674,439 |

U-Net achieved 61 additional successes, a **12.2 percentage-point gain**. It
improved on nine of ten tasks; the both-moka-pots task fell from 39/50 to 29/50.
Paired outcomes: both succeed 271, both fail 80, Transformer only succeeds 44,
U-Net only succeeds 105. All 1,000 episodes completed with zero evaluation or
video errors; ten preselected videos are retained per model.

Saved training and policy settings match except for backbone type and options:
same data split, training seed, learning rates, optimizer, batch size, EMA, image
encoder architecture and crops, heading predictor/conditioning, exact-heading
source, 16 completed epochs and 31,024 optimizer updates. All 127 stored
normalization/heading-scale state entries match exactly. The models were trained
separately, so their learned observation/head weights need not be equal.

Both evaluations use official initial-state indices 0–49 per task, paired seeds
derived from 20260911, ten open-gripper settling actions, fresh post-settling
observations, and a 550-action limit. This is the same protocol as our other
500-episode results, not the separately named Official matched five-step,
3000–3499 seed protocol. This is one training seed and is not a parameter-count-
matched experiment.

Artifacts: `output/heading_goal/backbone_500/RESULTS.md`, `comparison.json`,
`success_by_task.png`, `verification.json`, and each model's `eval_500/` and
inference-only `policy.ckpt`. The comparison reporter passed 18 integrity tests.
Both supervised evaluation jobs exited normally.


## StarVLA-DiT: matched training and 500-episode evaluation

The requested StarVLA-DiT variant is configured in
`oat/config/train_flowpolicy_heading_starvla_dit.yaml`. It inherits the same
`HeadingZeroFlowPolicy` recipe as the matched Transformer/U-Net experiment and
changes only the velocity backbone and its architecture-specific options. It
retains the jointly trained shared observation encoder and heading predictor,
heading conditioning, exact-heading XY source, and straight-path flow-matching
objective. This is the local StarVLA/GR00T action-head adaptation, trained from
scratch; it does not load a pretrained VLA or VLM.

The DiT uses four alternating cross/self-attention blocks, width 256, four heads,
time-adaptive layer normalization, and 32 register tokens. Training uses seed 42,
batch 64, AdamW with learning rates 1e-4, the same EMA and 500-update warmup,
116-pixel crops, and the same 450/50 demonstration split. The predeclared
checkpoint is epoch 15, after 16 complete epochs and 31,024 optimizer updates.
The saved configuration retains the same 61-epoch schedule budget as the
baselines; the trainer was stopped after securing the epoch-15 snapshot.
No simulator results were used to select this checkpoint.

| Backbone | Successes | SR | Velocity parameters |
|---|---:|---:|---:|
| Transformer | 315/500 | 63.0% | 4,784,903 |
| U-Net | 376/500 | 75.2% | 12,674,439 |
| StarVLA-DiT | 385/500 | 77.0% | 4,374,561 |

StarVLA-DiT gains 70 successes (+14.0 percentage points) over Transformer and
nine successes (+1.8 points) over U-Net. Against U-Net, 328 episodes succeed for
both policies, 67 fail for both, 48 succeed only for U-Net, and 57 succeed only
for StarVLA-DiT. These are descriptive results from one training seed and
unequal backbone parameter counts; they do not establish a general ranking.
The 500 states were used in earlier comparisons, so this is not a fresh holdout.

Evaluation uses one fixed EMA checkpoint for all ten tasks, the same official
initial-state indices 0–49 per task, paired seeds derived from 20260911, ten
open-gripper settling actions, and 550 policy actions. There are zero evaluation
or video errors. Evaluator source hashes, software versions, every task/state
assignment, both episode seeds, and settled initial-state hashes match the
baseline evaluations. This remains our ten-settling-step protocol, rather than
the separately named Official matched five-step/3000–3499 protocol.

All 465 selected StarVLA EMA tensors match the private training snapshot,
exported checkpoint, and strict reloaded policy exactly. All 127 saved
normalization and heading-scale entries match both baselines. The final policy
has 26,804,580 learned parameters; its learned observation encoder and heading
head have the same architecture and parameter counts as the baselines, but
were trained separately. Independent raw episode counts and paired outcomes
agree with the generated comparison report. The new reporter passed 30 tests.

Artifacts are in `output/heading_goal/starvla_dit_500/`: `RESULTS.md`,
`comparison.json`, `success_by_task.png`, `verification.json`,
`checkpoint_audit.json`, `model_sizes.json`, `normalization_equivalence.json`,
`selection.json`, and `eval_500/` with all episode records and ten videos.
The inference checkpoint is `policy.ckpt` with SHA256
`175cf4c5893dd640e6bdf38638928b76528cdf0dd5f40476abd99a2795002083`.
Its private training source is `source_epoch15.ckpt`. Training ran on GPU 0;
evaluation ran on GPU 1 and exited normally. The trainer is stopped.

To reproduce the training configuration with a fresh run name, run this
foreground command under supervisor:

```bash
scripts/run_heading_experiment.sh 0 reproduce_starvla_dit train_flowpolicy_heading_starvla_dit
```
