# QwenVL + StarVLA DiT Heading Gaussian

This experiment fully fine-tunes Qwen2.5-VL-3B (vision tower, visual merger and
language model) with StarVLA's native GR00T-style DiT and the fm_dno Heading
Gaussian source. It trains only this heading variant. Original LIBERO supplies
all demonstrations, statistics and checkpoint selection. LIBERO-Plus is held out
for zero-shot evaluation.

## Architecture and data

The native StarVLA revision is pinned in `oat/starvla_heading/__init__.py`.
Qwen emits the final hidden tokens without computing unused vocabulary logits.
The expert attends to all unpadded image/language tokens. A masked pooled summary
plus two normalized 9D robot states predicts unit XY heading and validity.
The direction and validity condition the DiT; detached copies construct its
heading-aligned, full-support Gaussian source in raw action coordinates.

The source retains the existing mean 0.5, parallel standard deviation 1.0,
perpendicular standard deviation 0.5, confidence threshold 0.5, and ordinary
Gaussian fallback. Flow time is continuous Uniform(0,1); inference takes ten
Euler steps. Direction cosine and validity BCE losses each have weight 0.1. Both reduce
masked losses per example, so regrouping samples into different microbatches
preserves the effective-batch objective; invalid headings contribute zero direction
loss.
The policy predicts 16 seven-dimensional actions and executes eight.

Original HDF5 demonstrations are read directly to preserve full instructions.
Inputs contain previous/current front and wrist images in temporal-major order,
plus position3, quaternion4 (xyzw), and gripper2. Images are rendered/stored at
128 pixels, flipped vertically, and bicubic-resized to 224 for Qwen. Dataset
episode starts repeat the initial frame. At later control cycles, history holds
the two most recent simulator steps. Padded action labels are masked from
supervision and cannot enter other action positions through the DiT's attention.

The frozen [manifest](../artifacts/starvla_heading/libero_manifest.json) contains
40 original tasks, 2,000 demonstrations, and deterministic per-task 45/5 splits:
1,800 train / 200 validation. Normalization and heading RMS use only unpadded
training frames. Source revisions, numerical hashes, and file identities are
recorded. Task IDs never enter the policy.

## Setup

Run `bash scripts/setup_starvla_heading.sh --download --with-plus`. It uses the separate
training and simulator environments under `/workspace/venvs`, pins StarVLA,
downloads the named base Qwen checkpoint and only the four original LIBERO
demonstration suites, and installs the official Plus assets. The preparation
command is:

```bash
/workspace/venvs/starvla-heading/bin/python scripts/prepare_starvla_libero.py \
  --root /workspace/shared_data/libero_hdf5 \
  --output artifacts/starvla_heading/libero_manifest.json
```

The manifest is immutable: an existing file is deliberately not overwritten.
The original simulator uses `/workspace/libero_config/original`; Plus uses
`/workspace/libero_config/plus`. Each runs in its own Python environment because
both packages own the `libero` import namespace.

## Training

The [configuration](../oat/config/starvla_heading_gaussian.yaml) uses six GPUs,
BF16, ZeRO-3, gradient checkpointing in Qwen, microbatch ten, and gradient
accumulation one (effective batch 60). This choice follows measured eight-update
profiles on six RTX 4090 cards: microbatch five took a median 18.37 seconds per
update, while microbatch ten took about 10 seconds. The latter peaked at
15.48 GiB allocated and 21.22 GiB reserved per GPU in PyTorch; an independent
device-memory observation was 22,371 MiB. These are short infrastructure
measurements, not completed training or evaluation results. It initializes the action expert from
scratch and fully fine-tunes every parameter. Learning rates are 1e-5 for
language, 5e-6 for vision/merger, and 1e-4 for the action expert, with a
500-update warmup. The configured training budget is 30,000 optimizer updates
per seed; seeds are 42, 43 and 44.

Launch training through supervisor using the wrapper in `scripts/supervisor`.
The underlying foreground command, useful inside a managed job, is:

```bash
bash scripts/run_starvla_heading.sh --seed 42 \
  --output-dir output/starvla_heading_gaussian/seed42
```

The launcher assigns physical GPUs 0–5. Override `STARVLA_TRAIN_GPUS` only with
six free GPUs. The experiment coordinator trains first, then reuses GPU 0 for
inference and GPU 1 for rendering. Six total GPUs are sufficient.

Before a full run, use the profile service to perform actual forward/backward,
component gradient audits, a full optimizer save, and a separate resumed step.
Its wrapper runs both fresh-process phases sequentially in a new
`checkpoint_profile` directory; set `STARVLA_PROFILE_OUTPUT` in its supervisor
environment to select another fresh directory.
The profile deliberately uses accumulation two and a tiny validation subset;
it is an infrastructure check, not a reported training result. Memory claims
must come from its per-rank measured peaks.

Validation computes generated normalized-action MSE on 600 fixed original
LIBERO holdout windows using deterministic source randomness and batches of ten
per GPU. It runs every 1,000 optimizer updates; the first production checkpoint
also receives the full validation pass before the long run resumes.
`best` and `latest` point to immutable inference export directories; only a
validated checkpoint can claim selection on original LIBERO. Serving resolves
these links once and verifies weight/configuration hashes.

A restart restores optimizer, scheduler, per-rank RNG, sampler epoch and batch
cursor. Only application-owned checkpoint fields are passed back to DeepSpeed.
If the final optimizer commit exists but completion status publication was
interrupted, resume verifies the final and selected exports and repairs the status
without taking an additional optimizer step. Its `INCOMPLETE` marker prevents loading an interrupted save.
Large optimizer state is retained once per active run to fit local disk.
Incomplete or failed jobs must not be described as trained checkpoints.

## Evaluation

The final evaluation of the checkpoint selected by original-LIBERO validation
runs after training exits, with inference on GPU 0 and simulator rendering on
GPU 1. Periodic success-rate evaluation runs alongside training on this
instance's two additional GPUs: inference on GPU 6 and rendering on GPU 7,
while training continues on GPUs 0–5. Both evaluation paths bind their inference
servers only to localhost and batch requests from simulator workers. Each server
loads one frozen export; workers do not instantiate their own Qwen models.
Requests carry independent noise seeds, language, images and raw state, and
return raw simulator actions.

Run original LIBERO first (50 saved states per task), then full Plus (one state
per variant). Plus comprises 2,402 Spatial, 2,518 Object, 2,591 Goal and 2,519
Long variants: 10,030 episodes. The evaluator uses the official initial-state
resolver and consumes observations returned by the Plus wrappers, including
the last settling step, so sensor corruption is preserved.

The instruction protocol is explicit. Non-language Plus variants use the exact
original instruction from the selected export's frozen training manifest.
Language Instructions variants retain the complete official rewritten instruction
from their language BDDL. Every Plus variant must map uniquely to one of the
40 original training tasks. The evaluator audits this mapping across all 10,030
variants and records the instruction protocol, training-manifest identity, and
prompt-audit hashes in the evaluation manifest.

This resolves an upstream discrepancy: the pinned StarVLA Plus helper forwards
`task.language` directly, while the pinned Plus task constructor can include
viewpoint, initial-state, noise, and other filename parameters in that string.
Using every variant's BDDL instruction also changes some non-language prompts
beyond those parameters. Canonical original instructions plus preserved language
rewrites are this experiment's declared protocol choice. It follows the
[official paper's separate perturbation factors and language-rewrite definitions](https://arxiv.org/html/2510.13626v1#S2.SS1),
but should not be described as identical to the pinned helper's prompt behavior.

Evaluation for either benchmark, including simulator smoke checks, requires
`--training-manifest /path/to/selected_export/dataset_manifest.json`. Use the
manifest shipped with the same immutable export loaded by the inference server;
its identity must match the checkpoint metadata. The experiment coordinator
passes this argument automatically. Direct evaluator invocations must include
it along with the selected benchmark's explicit simulator configuration.

Results include immutable evaluation manifests, episode JSONL, checkpoint
hashes, overall success, suite/category/difficulty breakdowns and infrastructure
errors. A partial evaluation reports no complete official success rate.
Resume rejects a changed checkpoint, task plan, state, instruction catalog, or
protocol. Failed infrastructure episodes can be retried with an audit trail.
Plus outcomes never select weights, normalize actions, change priors, or tune
hyperparameters.

## Checks

```bash
/workspace/venvs/starvla-heading/bin/python -m pytest -q \
  tests/test_starvla_heading_data.py tests/test_starvla_heading_model.py \
  tests/test_starvla_heading_eval.py tests/test_starvla_heading_training.py \
  tests/test_starvla_experiment.py
```

These cover raw Gaussian geometry and fallback, source detachment, padding and
attention masks, train-only statistics, real native-DiT gradients, checkpoint
contracts, image processing, wire batching, official state mapping, and resume.
They complement the actual six-GPU profile and simulator rollouts.

## Managed experiment

The production wrapper is
`scripts/supervisor/fm-starvla-heading-experiment.sh`, with a matching supervisor
configuration. It runs `scripts/run_starvla_experiment.py` in the foreground.
The coordinator runs each configured seed through 30,000 training updates,
original LIBERO evaluation, and the full LIBERO-Plus evaluation before advancing
to the next seed. It records per-seed results and the mean and sample standard
deviation across independently trained seeds.

The service uses a sixteen-minute shutdown window to allow a validation pass
and optimizer save to finish. The coordinator signals training
ranks first and waits for an optimizer-boundary save before cleaning up their
process groups. Training, inference, and simulation logs remain in each seed
directory. A failed stage stops the coordinator and retains its error evidence.

The wrapper enables `--prune-completed-optimizer` to fit this instance's disk.
This removes a seed's generated optimizer state and unselected latest export
only after all 30,000 updates, successful selected-checkpoint reload, and both
complete benchmark evaluations. It preserves selected weights, source/configuration
metadata, episode records, and result summaries.

The completed initial checks are recorded in
`output/starvla_heading_gaussian/profile`, `profile_micro5`, and `profile_micro10`.
Their disposable optimizer/weight files were pruned after verification; the
`profile_*_pruned.json` records identify exactly what was removed. These
directories provide infrastructure evidence and are not production checkpoints.
The separate `outputs/starvla_heading_gaussian/real_policy_smoke/step2_export`
snapshot remains available for reproducing short policy-integration checks.

## Periodic success-rate evaluation

For each training seed (42, 43 and 44), evaluate the exact committed checkpoints
at steps **10,000, 20,000 and 30,000**. Every milestone covers all 2,000 original
LIBERO episodes and all 10,030 LIBERO-Plus episodes. The existing 600-window
validation and checkpoint-save interval remains 1,000 optimizer updates.
Periodic success rates do not select checkpoints or change training settings;
the final selected-policy evaluation continues to use original-LIBERO
validation MSE. A step-30,000 policy and the final selected policy may differ.

The independent `fm-starvla-heading-periodic-eval` supervisor service watches all
seeds, including while another milestone's evaluation is running. It pins an
exact committed export outside the trainer's export directory using hardlinks,
so routine best/latest pruning cannot remove an evaluation's weights. Inference
uses GPU 6 at `http://127.0.0.1:18087`; four simulator workers render on GPU 7.
The training coordinator and its six ranks continue independently.

Periodic artifacts live under
`output/starvla_heading_gaussian/periodic_evaluation/seed42/step_00010000/`
(with matching directories for the other seeds and milestones). Each contains
checkpoint identity/configuration, the frozen training manifest, benchmark
manifests, episode results, and verified summaries. Global events are written to
`periodic_evaluation/events.jsonl`. Both benchmarks must pass the existing
complete-coverage and checkpoint-identity checks before a milestone is complete.
After both benchmarks are verified and its server stops, the coordinator
releases its own extra weights link, retaining metadata, hashes and results.
It never removes the trainer's best/latest exports or optimizer state.

The complete experiment requires both the original selected-policy workflow and
all periodic milestone evaluations. Results are reported separately by exact
training step, with the mean and sample standard deviation across completed
training seeds.

## Current execution and evidence

The implementation passed 73 combined tests. Seed 42 completed the initial ten
production updates, 600-sample original-LIBERO validation, and an optimizer
save; the coordinator then resumed that checkpoint. Its first scheduled
validation at step 1,000 also covered all 600 windows: normalized action MSE
improved from 1.2568308674 to 0.1403594262. That checkpoint committed and was
selected as both best and latest at the time of the audit. Training continued
beyond step 1,000 without a restart.

The step-1,000 checkpoint passed 51 acceptance checks covering export/config/
manifest hashes, checkpoint selection, all six model/RNG records, and sequential
CPU deserialization of all six optimizer shards. This was a file-integrity and
state-consistency audit; it did not perform another distributed restart or an
exhaustive finiteness scan of saved optimizer tensors. Evidence is in
`outputs/starvla_heading_gaussian/checkpoint_1000_audit/verification.json`.
W&B received separate training, validation, and checkpoint events at step 1,000;
its remote verification is recorded in
`output/starvla_heading_gaussian/wandb_tracking/verification_checkpoint_1000.json`.

The 30,000-update budget and seeds 42/43/44 remain active. No complete benchmark
success rates are available yet.

The live process is `fm-starvla-heading-experiment` in supervisor. Check it with
`supervisorctl status fm-starvla-heading-experiment`. Its event log is
`output/starvla_heading_gaussian/experiment_events.jsonl`; the active seed's
`metrics.jsonl` and `training.log` contain training progress. A stored pause status
from the initial bootstrap is historical; supervisor and current log events
establish whether training is running.

The consolidated evidence index is
`outputs/starvla_heading_gaussian/readiness.json`. It links the gradient, memory,
resume, simulator, prompt and token audits and records implementation file hashes.
The clean Plus prompt audit covers every variant, and all 1,549 unique policy
instructions fit the 512-token budget (313–364 tokens including four images).

A direct simulator replay audit also checked image alignment: one task per
original suite, three frames, and both cameras (24 pairs). Raw HDF5 images
matched raw simulator images without rotation, and the actual training and
evaluation preprocessing remained aligned (mean pixel MAE 1.34 on a 0–255
scale). All 40 original files declare the same OpenGL image convention.
See `outputs/starvla_heading_gaussian/camera_alignment_audit/REPORT.md`.

The replayed images matched the next stored simulator state, consistent with
the official collector recording an observation after applying each action.
The adapter preserves the reference loader's stored-row observation/action
pairing; this provenance detail does not introduce an action-label shift.

## Online W&B tracking

The private project is
[fm_dno_starvla](https://wandb.ai/andyliu7081-northeastern-university/fm_dno_starvla).
The current run is
[Heading Gaussian, seed 42](https://wandb.ai/andyliu7081-northeastern-university/fm_dno_starvla/runs/svhg42-a2230f2009ce).
Separate runs for seeds 43 and 44 are created when their training logs appear;
all three share one experiment group.

The `fm-starvla-heading-wandb` supervisor service runs a CPU-only logger,
independently of the training coordinator. It backfills each seed's existing
`metrics.jsonl` and polls for appended events every ten seconds. Training
currently emits metrics every ten optimizer updates, approximately every
105 seconds. Charts use `optimizer_step` as their horizontal axis. Each seed's
`wandb_history_ledger.json` durably orders training and verified periodic
evaluation events before upload. The W&B history step is the ledger row index;
legacy training history keeps its original indices. A delayed evaluation result
uses its checkpoint's optimizer step even when training has advanced further.

Logged fields include action and flow losses, heading and validity losses,
heading error, source activation, learning rates, elapsed training time,
per-rank peak GPU allocation/reservation, original-LIBERO validation metrics,
and committed checkpoint paths and steps. Verified benchmark success rates
are added after the corresponding full benchmark is verified. Periodic curves
use `evaluation/periodic/libero/success_rate` and
`evaluation/periodic/libero_plus/success_rate`. Periodic status and results have
separate summary keys from the final selected-policy results. W&B runs remain
open until both workflows finish for that seed.
Checkpoint weights and optimizer states remain in their existing local paths;
the logger does not upload model artifacts.

Inspect the three services with:

```bash
supervisorctl status fm-starvla-heading-experiment fm-starvla-heading-wandb fm-starvla-heading-periodic-eval
```

Each seed's `wandb_online.json` records its run URL, stable identity and source
fingerprint. Its local cursor records queued events, not confirmed uploads.
On restart, the logger resumes the same remote run using W&B's history position
and verifies the local source is append-only. A lock prevents concurrent logger
instances. Restart only `fm-starvla-heading-wandb` when needed; training continues
independently. Credentials come from the existing W&B login and are not stored
in the repository. The logger's offline contracts have 71 passing tests.
