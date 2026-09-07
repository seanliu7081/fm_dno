# Image and state reference headings

The learned reference predicts an action-chunk heading from the current observation
window. It can use both camera images and robot state, or state alone. The action flow
still receives its usual image and state conditions. The additional reference either
changes the initial noise orientation, enters the flow as an explicit condition, or does
both. This adds a geometric bias; it does not add observation information.

For source canonicalization, the policy samples fresh noise and applies

```text
theta_ref = heading_predictor(observations)
z = rotate(epsilon, theta_ref + angular_noise - heading(epsilon))
```

Subtracting the original noise heading is essential. Rotating an isotropic Gaussian by
an observation-dependent angle alone would leave its distribution unchanged. A finite
von Mises `kappa` retains angular diversity around the predicted reference. This is an
observation-dependent source distribution, not a guarantee of rotation equivariance.

## Train the reference

Run commands from the repository root, using the project's Python environment. The
trainer uses the existing LIBERO Zarr format and does not require simulator rollouts or
W&B. These examples use the default `data/libero/libero10_N500.zarr`; override
`task.policy.dataset.zarr_path` for another compatible dataset.

```bash
python scripts/train_heading_reference.py --config-name=train_heading_reference \
  reference_mode=image_state seed=42 split_seed=42 \
  hydra.run.dir=output/heading_reference/image_state/seed42

python scripts/train_heading_reference.py --config-name=train_heading_reference \
  reference_mode=state seed=42 split_seed=42 \
  hydra.run.dir=output/heading_reference/state/seed42
```

The state variant removes RGB keys before loading the dataset and constructing the
encoder. It retains state fields such as task identity when provided in `shape_meta`.
The image variant uses the existing fused observation encoder with fixed center crops
during evaluation. Random crops are allowed during reference pretraining; the frozen
reference uses evaluation preprocessing during both flow training and deployment.

Future actions are supervised labels only. The target is `chunk_heading` of the action
after SO(2)-compatible normalization, with translation-only heading weights by default.
Samples whose target heading magnitude is below `reference.min_target_confidence`
contribute no angular loss and receive an invalid-heading label. The predictor learns a
unit direction and the probability that this heading is valid. That probability is
**not** a calibrated probability that the predicted angle is correct. In particular, a
multimodal direction can have a valid target heading but remain hard to predict.

The trainer splits by episode, fits action/state normalization on training episodes
only, and uses a fixed `[0, 255]` RGB range. `split_seed` stays constant independently of
the optimization seed. The exact episode IDs and dataset settings are written to
`split.json`. Subsampling training episodes does not add the omitted episodes to
validation.

Each run writes:

- `checkpoints/best.pt`, selected by validation heading-plus-validity loss.
- `checkpoints/last.pt`, the final reference artifact.
- `metrics.json`, per-epoch training and validation metrics.
- `config.yaml` and `split.json`, resolved construction settings and split provenance.

Reference artifacts include the complete encoder, heading head, normalizers, action
layout, horizon, model configuration and metadata. They are used to initialize the
frozen reference in the action policy; they are not optimizer-resume snapshots.

Validation reports angular MAE, mean cosine, residual concentration `R1`, validity
coverage, Brier score and probability calibration bins. It also reports angular
metrics separately for stationary and moving observations when `robot0_eef_pos` is
available. Angle metrics exclude invalid target headings. `R1` alone is insufficient:
a consistently reversed predictor can have `R1=1` while its mean cosine is `-1`.

A short state-only trainer smoke run on a compatible local dataset is:

```bash
python scripts/train_heading_reference.py reference_mode=state training.device=cpu \
  training.num_epochs=1 training.max_train_batches=2 training.max_val_batches=2 \
  dataloader.batch_size=8 dataloader.num_workers=0 \
  hydra.run.dir=output/heading_reference/smoke
```

This still loads the selected dataset; use a small fixture for a memory-light smoke run.
It verifies execution and artifact writing, not prediction quality.

## Train the action policy

Initialize the learned canonical flow from a pretrained reference:

```bash
python scripts/run_workspace.py --config-name=train_flowpolicy_learned_canon \
  reference_checkpoint=output/heading_reference/image_state/seed42/checkpoints/best.pt \
  policy.reference_use=source policy.kappa=4.0 \
  seed=42 training.seed=42 task.policy.dataset.seed=42 \
  hydra.run.dir=output/learned_canon/source/seed42
```

The entire reference encoder and head are frozen, stay in evaluation mode when the flow
enters training mode, and use their saved normalization. The flow uses predicted
references on both training and inference paths; ground-truth action headings never
choose the source orientation. The pretrained reference is stored in the policy's
own checkpoint once initialized. A fresh workspace loads the pretrained artifact in
`policy.prepare_for_training()` after attempting workspace resume, so resumed and
deployed policies can use the embedded reference without the original external file.
If you change the reference architecture, observation
ports, crop settings, horizon, or heading target settings during pretraining, match those
settings in the policy config; checkpoint loading rejects incompatible references.

To use the learned state-only reference, set `reference_mode=state` and supply its
checkpoint through `reference_checkpoint`. The flow's main
observation encoder still receives images and state in both cases. `kappa` controls the
source's angular spread; compare several finite values, such as `2`, `4`, and `16`, and
retain `.inf` as the exact-alignment comparison.

For low-level diagnostics, `sample_chunk` expects a source that has already been
canonicalized; obtain it with `sample_prior_from_obs`. `predict_action_from_noise`
accepts Gaussian noise and performs the observation-dependent source transformation
internally, just as the standard prediction path does.

## Controlled comparisons

Use the same flow backbone to isolate canonicalization from differences between
diffusion and flow matching. Keep action normalization, dataset split, batch size,
action horizon, observation window and inference steps fixed.

| Experiment | Config name | Reference | Extra condition | Canonical source |
|---|---|---|---|---|
| A | `train_flowpolicy_canon_iid` | None | No | No |
| B | `train_flowpolicy_canon_soft` | Recent end-effector motion | No | Yes |
| C | `train_flowpolicy_learned_canon_state` | Learned state | No | Yes |
| D | `train_flowpolicy_learned_canon` | Learned image and state | No | Yes |
| E | `train_flowpolicy_heading_condition` | Same checkpoint as D | Yes | No |
| F | `train_flowpolicy_heading_both` | Same checkpoint as D | Yes | Yes |

C and D use `policy.reference_use=source`. E and F use the same reference checkpoint
with `policy.reference_use=condition` and `policy.reference_use=both`, respectively.
E versus F isolates the effect of changing the source when the network already receives
the predicted reference explicitly. C versus D measures the value of images to the
reference. A uses the same SO(2)-normalized backbone with source alignment disabled;
this avoids changing action normalization along with the source distribution. B uses
finite angular dither. Hold `policy.kappa` fixed across B, C, D and F for the first
comparison; the original `train_flowpolicy_canon` remains the exact-alignment control.

Repeat training with at least seeds 42, 43 and 44 while keeping `split_seed=42` for the
reference trainer and `task.policy.dataset.seed=42` for the action policy. Compare
paired E/F runs using the same pretrained reference per seed. Evaluate all conditions
on the same environment starts and report closed-loop success, seed variation, training
cost, inference latency and action diversity. Pay particular attention to stationary
starts, turns and reversals. Better offline heading metrics alone do not establish a
better action policy.

The initial implementation does not perform long training runs or claim a success-rate
improvement. Useful next experiments are direction-error uncertainty (beyond heading
validity), multimodal directional mixtures, execution-window heading labels, joint
reference/flow fine-tuning, and generation in a transformed action frame. Each changes
the hypothesis being tested and should follow the frozen-reference comparison.
