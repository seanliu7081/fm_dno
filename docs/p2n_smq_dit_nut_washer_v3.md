# SMQ-DiT with self-generated action history

Experiment name: `train_p2nDiTL_nut_wahser_v3` (the requested spelling).
Dataset: `/workspace/ysk/zarr/nut_washer_v3_N77.zarr`.
This experiment is implemented in new files; the existing SMQ-DiT, Past2Next,
datasets, and training workspace are unchanged.

## Launch

```bash
bash /workspace/fm_dno/train_p2nDiTL_nut_wahser_v3.sh --gpu 1 --batch-size 64
```

Use `--gpu 0,1` for two training processes. Batch size is per GPU. Optional
flags are `--epochs` (default 500), `--run-dir` (must be fresh), and `--dry-run`.
The dry run checks data/GPU selection and resolves configuration without
initializing W&B or creating a training run.

W&B is online in project `real_robot`, group `train_p2nDiTL_nut_wahser_v3`.
The inherited checkpoint interval is 50 zero-based epoch IDs: completed epochs
1, 51, 101, ... plus the final epoch. Every numbered archive is retained, and
`latest.ckpt` is updated at the same boundaries. Default output directories
start with `output/train_p2nDiTL_nut_wahser_v3_`.

## Model and training

The model has 56,359,818 trainable parameters before/after fitting the frozen
normalizers. It retains two independent ResNet18 encoders, DiT width 512,
8 attention blocks, 8 heads, 16 predicted actions, 8 executed actions, and
10 Euler sampling steps. OAT tokenization and autoregressive output are not used.

Seven normalized past action commands produce seven ordered history tokens.
Separate projections encode the latest first and second command differences.
Lag/type/validity embeddings distinguish ordering, feature kind, and missing
history. Invalid values are replaced by learned missing content, including
unavailable difference features. These are command differences, not calibrated
physical acceleration/jerk measurements for every action channel.

Both SMQ and DiT read history. SMQ receives 67 observation tokens plus 9 history
tokens, and DiT receives those plus 4 motion tokens (80 in total, width 256).
The segmented heading source, source detach, flow loss, heading/validity losses,
and training-only normalization/RMS remain as in the SMQ baseline.

After 500 successful optimizer updates with expert history, the probability of
using generated history increases linearly over 2000 updates to 1. The online
policy counter advances in an optimizer post-step hook; a separate EMA variant
copies that counter exactly. Both counters are checkpointed, and validation
never advances them. Settings live in `policy.self_past_*` in the new YAML.

For a current action anchor t, generation uses observations at t-9 and t-8
and expert history at t-15 through t-9. It samples 16 actions, then keeps
`previous_prediction[:, 1:8]`, corresponding to t-7 through t-1. The unexecuted
prediction tail is never used as history. Windows with t < 8 within their
episode cannot generate a real previous chunk and retain masked expert history.
Inner generation runs in eval/no-grad mode with the autocast weight cache
disabled, then restores all module modes. It does not retain a gradient graph.

## Evaluation and rollout

`SpatialMotionSelfPastDataset` preserves the original train/validation split
and current observation/action alignment. Missing history is zero-filled with
an explicit mask. Future action masks remain auxiliary-label inputs only.

The dataset also places target-free history context under `obs['_p2n_context']`.
This lets the unchanged workspace's observation-only reconstruction call use
explicit per-sample history without mixing independent windows in an online
buffer. That mapping is removed before the RGB/state encoder is called.

Both `val_loss` and `test_reconst_mse` use depth-one generated history for every
eligible sample, even during the expert-history training warmup. Therefore early
validation uses a harder history distribution than training. Generated history
is still paired with recorded expert observations; this is not a closed-loop
robot rollout or a success-rate measurement.

`sample_actions(obs, past_action, past_mask, noise=None)` is a stateless sampler.
`predict_action` is also stateless when supplied dataset context or explicit
`past_action` and `past_mask`. For ordinary rollout observations its default
buffer update assumes the entire returned prefix executes. For interrupted,
clipped, or otherwise modified execution, call
`predict_action(obs, update_history=False)`, then
`record_executed_actions(actual_commands)` with shape `[B, N, 7]`. Call `reset()`
at an episode boundary; keep independent buffers for independently reset streams.

## Validation performed

Focused dataset, policy, and EMA tests cover temporal alignment, episode
boundaries, history masks, lag ordering, stateless evaluation, detached sampling,
optimizer coverage, curriculum updates, EMA synchronization, and checkpoint
resume. A full-size BF16 smoke run on a temporary real-data slice exercised
self-past training, validation, reconstruction, and numbered/latest checkpoints.
The smoke test used disabled W&B only in its temporary overrides; the launch
script and experiment configuration enforce online W&B.
