# Shared-heading mini experiment

This experiment tests whether heading can be read from the ordinary flow policy's observation features, whether explicit heading supervision improves those features for action generation, and whether feeding the predicted heading back into the flow network adds value.

It is an **offline early-learning experiment**, not a LIBERO success-rate evaluation. Its results cannot establish whether the historical 39% versus 28% success-rate gap is recovered.

## Three matched variants

| Mode | Heading supervision updates the shared encoder | Heading enters flow conditioning |
| --- | --- | --- |
| `baseline` | No; the diagnostic head receives detached features | No |
| `auxiliary` | Yes | No |
| `condition` | Yes | Yes, predicted direction and validity probability |

All variants share the same observation encoder architecture: separate ResNet18 image branches for external and wrist cameras, state features, and two observation frames. A `276 → 128 → 3` heading head reads the shared observation features. There is no separate reference encoder, pretrained reference artifact, or frozen heading predictor.

The baseline's diagnostic head learns a readout without changing the policy's gradients. The runner clips heading-head and core-policy gradients separately, so the readout cannot indirectly change baseline updates through global clipping. The head uses its own optimizer group. The baseline and auxiliary variants reserve three zero conditioning columns; all models have identical shapes and the added projection columns start at zero. CPU checks verify equivalence to an ordinary FlowPolicy with copied matching weights.

The heading target is the raw XY resultant over the 16-action chunk. Heading validity uses its norm divided by a fixed training-only XY RMS and `sqrt(16)`, with threshold 0.05. The RMS only sets the validity-label scale: **policy actions continue to use ordinary per-dimension Min–Max normalization**. The source remains IID Gaussian in every variant. Future actions only supply training labels and never enter conditioning.

## Fixed protocol

- LIBERO-10, `libero10_N500.zarr`, all 10 task IDs 30–39.
- Original episode split seed 42: 450 training and 50 validation episodes.
- 4,000 fixed training windows (400/task), 1,000 validation windows (100/task), spread across eligible episodes. Episodes never cross the split.
- Action and state normalization statistics fitted only on the 450 training episodes; RGB uses fixed `[0,255]` bounds. This corrects the historical dataset helper's full-buffer statistics for **all three** experimental arms.
- Same initial weights, minibatch order, augmentation/dropout RNG, flow noise and time draws within each seed.
- Training seeds 42 and 43; from-scratch initialization; batch 32; 5,000 updates per variant in the formal mini comparison.
- Transformer: width 256, 4 layers, 4 heads; action horizon 16, execution prefix 8, 10 Euler steps.
- Learning rates: flow `5e-5`, observation encoder `1e-5`, heading head `1e-3`; 100-step warmup; heading cosine and validity BCE loss weights each 0.1; separate gradient clipping at 1.
- EMA evaluation every 1,000 updates, using fixed center crops and identical noise/time samples.
- Generated-action metrics use 200 validation windows (20/task), distributed across validation episodes, with one common latent sample per window.
- Report the fixed final update, with curves showing earlier checkpoints; do not select a different best checkpoint per metric.

The smaller `pilot_20260909` run used 600 updates to validate throughput and early-learning behavior. It is separate from the formal 5,000-update comparison. The initial two-update smoke run only checked execution.

## Run and inspect

From the repository root, with the `oat` environment:

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python scripts/mini_shared_heading.py \
  --output-dir output/mini_shared_heading/mini_5000_20260909 \
  --seeds 42 43 --steps 5000 --eval-every 1000
```

The output directory must not already exist. Use a new directory for another run.

```bash
python scripts/summarize_mini_shared_heading.py \
  --run-dir output/mini_shared_heading/mini_5000_20260909 --pdf
```

The run records `manifest.json`, `split.json`, fitted normalization, per-step metrics and per-example evaluation arrays, and final EMA weights for every seed/variant. Matching source snapshots are saved under the run's `source/` directory.

Useful metrics are the **flow loss alone**, valid-label heading MAE/cosine, raw-action MSE for the executed eight-step prefix, and generated-action heading error. Total loss is not a fair action-quality comparison because it includes auxiliary/readout supervision.

## Interpretation limits

A successful baseline readout shows that direction is accessible from the shared features. Since those features also contain robot state, this does **not** prove the visual encoder alone learned geometric reasoning, or that the online readout is optimal.

Lower heading error does not automatically imply better action generation. Lower offline flow/action error does not establish higher closed-loop success. Two training seeds and one auxiliary weight provide a preliminary comparison, not a robust ranking across hyperparameters. The smallest task validation split has only two episodes. Any paired episode bootstrap interval is conditional on the two trained models/seeds and does not capture general training-seed uncertainty.

## Completed result — 2026-09-09

The formal run completed all six models (two seeds × three variants), each at the fixed 5,000-update endpoint. Training/evaluation/checkpoint time after data preparation was 928.4 seconds on GPU 1. [Full report](../output/mini_shared_heading/mini_5000_20260909/results.md), [paired statistics](../output/mini_shared_heading/mini_5000_20260909/comparison.json), and [learning curves](../output/mini_shared_heading/mini_5000_20260909/learning_curves.png).

| Variant | Heading readout MAE | Flow MSE | Raw prefix-8 action MSE | Generated chunk heading MAE |
| --- | ---: | ---: | ---: | ---: |
| Basic Flow + detached probe | 39.93° | 0.181017 | 0.151808 | 69.43° |
| Shared auxiliary supervision | 33.56° | 0.181084 | 0.159220 | 65.52° |
| Shared auxiliary + heading condition | 34.85° | 0.174208 | 0.139015 | 50.78° |

Values are means over seeds 42 and 43. All errors are lower-is-better. Training-only global and task-specific constant-heading controls yield 82.93° and 78.19° validation MAE, respectively.

The baseline's features support a useful sample-dependent heading readout without heading supervision reaching the encoder. Auxiliary supervision improves that readout but does not improve the overall action objective here. Adding the predicted heading condition reduces flow MSE by 3.76% and generated-action heading error by 26.86% relative to baseline; both seeds show these improvements.

Mean raw prefix-action MSE improves by 8.43%, but the seed-specific reductions are **16.28% and 0.04%**. That average is not evidence of a stable improvement across training seeds. Condition-minus-baseline paired episode-bootstrap intervals are [-0.00838, -0.00522] for flow MSE and [-0.02288, -0.00169] for raw prefix MSE, conditional on these two models. They do not resolve the training-seed dependence or imply improved SR.

The evidence supports shared-feature heading predictability and a potential optimization benefit from explicit conditioning. It does not establish that a separate frozen predictor is necessary, that the vision branch alone learned direction, or that the historical 39% versus 28% closed-loop gap has been recovered.

Validation: 17 targeted CPU tests passed; all three smoke variants ran on GPU; CPU reconstruction on real observations reproduced predictions exactly. The new checkpoints were verified with `weights_only=True`. Their TorchVersion metadata was converted to a plain string without changing any tensor values; `checkpoint_serialization.json` records that metadata-only repair, and `artifact_hashes.json` records final checkpoint identities. When restoring, explicitly set the checkpoint's `mode`, which is configuration rather than a tensor in the state dict.
