# Shared heading condition + directional noise prior mini experiment

This is a matched extension of [the shared-heading mini experiment](shared_heading_mini.md). It tests whether a directional source adds useful action-generation bias once the flow model already receives the same jointly learned heading as a condition. It is an offline early-learning test, not a closed-loop success-rate benchmark.

## Three variants

| Mode | Shared heading condition | Source change |
| --- | --- | --- |
| `condition` | Yes | IID Gaussian |
| `condition_prior_xy` | Yes | Align the realized translation XY source heading |
| `condition_prior_both` | Yes | Same alignment; also rotate the rotation XY block by the same angle |

The head, encoder, and flow network train jointly in all three variants. There is no separately pretrained or frozen predictor. The head receives flow gradients through its conditioning features and auxiliary heading/validity supervision. **Only the prediction used to construct the source is detached**: this experiment does not test gradients through the learned interpolation endpoints or velocity targets.

Every variant uses exactly the same ordinary per-coordinate Min–Max action normalizer, fitted only on the 450 training episodes. For this dataset, translation XY happens to have equal scales `1.0666667` and zero offsets, since both coordinates range from `-0.9375` to `0.9375`. Rotation XY is different: scales are approximately `3.537587` and `2.962963`, with offsets `-0.163613` and `-0.095238`.

## Source definition

Let `N` be the fixed action normalizer, `z = prior_scale * epsilon` the ordinary Gaussian source in normalized coordinates, and `u = N^-1(z)`. Compute the realized source heading from the raw translation XY resultant over the 16-step chunk. Then

```text
phi = predicted_heading + delta - heading(u_translation_XY)
delta ~ VonMises(0, kappa=4)
x0 = N(R_phi u)
```

`R_phi` rotates the selected two-dimensional blocks at every action time. All unselected coordinates are preserved exactly in normalized space. Apply the transform when predicted validity probability is at least `0.5` and the realized source direction is defined; otherwise retain `z`. Validity probability is not an estimate of angular prediction accuracy.

Subtracting the realized source heading is essential: rotating isotropic Gaussian noise by an independent heading angle alone would leave its distribution unchanged. The source construction uses observations and randomness only, never future action targets. Training interpolation and Euler inference use the same transform.

Raw-coordinate conjugation preserves raw vector magnitudes. It need not preserve normalized magnitudes, offsets or covariance when normalization is anisotropic. Translation XY already commutes with rotation in this particular dataset, so `condition_prior_xy` isolates the directional-source change cleanly. The both-block variant also changes the rotation-block source distribution under ordinary Min–Max; it is a sensitivity comparison, **not an exact reproduction of the historical SO(2)-normalized F model**. Under the historical isotropic normalized rotation source, rotating that independent block by a translation-derived angle would leave its distribution unchanged.

## Fixed protocol

- Same LIBERO-10 dataset, episode split seed 42, 4,000 training windows, 1,000 validation windows, 200 generated-action evaluation windows and fitted normalization as the previous shared-heading mini.
- Training seeds 42 and 43; all three arms start from identical weights per seed and train from scratch for 5,000 updates each, batch size 32.
- Same shared observation encoder, transformer, optimizer groups, learning rates, clipping and EMA as the previous mini. Heading cosine and validity BCE weights remain `0.1` each.
- Same minibatches, crops, dropout, base Gaussian noise and flow times within each seed. Von Mises angles come from an explicit independent NumPy generator and do not advance global NumPy or torch RNG streams.
- EMA evaluation at initialization and every 1,000 updates. Use the final fixed endpoint, without selecting different best checkpoints per metric.
- One generated action sample per selected observation, with common base Gaussian draws and ten Euler steps. Evaluation uses fixed center crops.
- The IID condition control is rerun and compared against the prior experiment to detect implementation drift.

Primary measurements are raw action MSE over the executed eight-step prefix, its XYZ component, and generated 16-step heading error. Predictor heading error, rotation/gripper error, source activation, source magnitude and source-to-target distance provide diagnostics. **Native flow MSE is not directly comparable as action quality across source variants**, because both interpolation paths and velocity targets change.

Paired episode-bootstrap intervals, when reported, condition on the two trained seeds. They do not estimate general training-seed uncertainty. Offline imitation error with one sample does not establish success rate or cover the generated distribution fully. Kappa, confidence threshold, source-gradient convention and auxiliary weights are fixed in advance for this test; conclusions are limited to this setting.

## Secondary sampling check

After the first seed showed slightly lower generated heading error but higher action error with the prior, a secondary inference-only check was specified while the second seed was still training. The fixed primary results remain unchanged. All six final models are evaluated on the exact same 200 windows with four additional paired latent streams; all draws are included. This checks dependence on the single primary latent sample without retraining or selecting a best draw.

The extra Gaussian seeds are `1900000 + draw_index * 10000 + batch_start`, using a local torch generator; angular seeds are `2300000000 + draw_index * 10000 + batch_start`, using the private NumPy Von Mises sampler. Errors are calculated separately for each generated action, then averaged over draws; actions themselves are not averaged before scoring. Saved arrays retain every draw and window plus the actual Gaussian/source/jitter samples. The secondary episode-bootstrap intervals condition on these four streams and the two training seeds.

## Reproduce

From the repository root in the `oat` Python environment:

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python scripts/mini_heading_prior.py \
  --output-dir output/mini_heading_prior/mini_5000_20260909 \
  --seeds 42 43 --steps 5000 --eval-every 1000

python scripts/summarize_mini_heading_prior.py \
  --run-dir output/mini_heading_prior/mini_5000_20260909 --pdf
```

Use a new output directory for another run. Outputs include the manifest, exact executed source snapshots, split and frame IDs, normalizer, per-example metrics, learning curves and final EMA weights. Restore both `heading_mode` and source configuration (`source_mode`, `source_vector_blocks`, kappa, confidence threshold) from checkpoint metadata; these settings are not tensor state. Supplying explicit base noise and angular jitter reproduces sampling exactly, independently of the private fallback angular RNG state.

The two-update `smoke_v2_20260909` run checks execution only. The first smoke attempt stopped before training because a tiny evaluation split exposed a sample-allocation edge case; the allocator now redistributes counts across available episodes. The formal split and sampled windows are unchanged.

Secondary inference command, after the primary run completes:

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python scripts/evaluate_mini_heading_prior_sampling.py \
  --run-dir output/mini_heading_prior/mini_5000_20260909 \
  --output-dir output/mini_heading_prior/sampling_robustness_20260909
```

## Completed result — 2026-09-09

All six models completed the fixed 5,000-update budget. Training/evaluation/checkpoint time after data preparation was 946.7 seconds on GPU 1. The secondary four-stream evaluation took 6.5 seconds after loading the selected observations.

| Variant | Primary prefix-8 MSE | Primary generated heading MAE | Secondary 4-stream prefix-8 MSE | Secondary generated heading MAE |
| --- | ---: | ---: | ---: | ---: |
| Condition + IID | 0.139015 | 50.78° | 0.140088 | 51.42° |
| Condition + translation XY prior | 0.150203 | 49.00° | 0.141372 | 47.54° |
| Condition + both-block prior | 0.144900 | 49.16° | 0.140320 | 48.67° |

Values average the two fixed training seeds. The primary single-stream comparison gives 8.05% higher overall action MSE for XY-only and 4.23% higher for both-block. XY-only worsens both seeds; both-block changes sign between seeds. Its small heading-error reductions alone do not demonstrate better actions.

The secondary four-stream check substantially weakens a harm interpretation: mean overall action MSE is only 0.92% higher for XY-only and 0.17% higher for both-block, with opposite effects across training seeds. Paired episode intervals for their MSE differences are respectively [-0.00308, +0.00580] and [-0.00528, +0.00547]. Both include zero. Directional steering remains visible: generated heading error falls by 3.88° (XY-only) and 2.75° (both-block); these effects have the same sign across the two seeds.

**Conclusion: the prior changes generated direction, but this experiment provides no stable overall action-quality benefit from adding it to the shared heading condition. It also does not establish robust harm.** Keeping the shared trainable head with an IID source is the simpler supported default for the next comparison; the prior remains an optional ablation. These short offline results cannot determine SR or establish that all historical D/E/F variants should use the same source or normalizer.

The measured source gate is active on all 1,000 final validation windows in both prior variants. XY-only preserves normalized source mean-square magnitude (1.000850 for both prior and IID); both-block raises it to 1.019126. This confirms that the extra rotation-block transformation changes more than the translation direction distribution.

Validation: 31 targeted CPU tests passed. All six final checkpoints load safely and strictly, restore their source configuration, and produce finite, exactly reproducible actions on real observations. Both IID controls reproduce every shared evaluation array at every saved step and all 487 final checkpoint tensors from the previous experiment. Dataset split, normalizer, 36 evaluation identities and executed source snapshots were independently verified.

[Primary report](../output/mini_heading_prior/mini_5000_20260909/results.md), [learning curves](../output/mini_heading_prior/mini_5000_20260909/learning_curves.png), [secondary four-stream report](../output/mini_heading_prior/sampling_robustness_20260909/report.md), [independent verification and checkpoint hashes](../output/mini_heading_prior/mini_5000_20260909/verification.json).

## Visual comparison of the saved source noise

[PNG figure](../output/mini_heading_prior/noise_visualization_20260909/iid_vs_heading_prior.png), [PDF](../output/mini_heading_prior/noise_visualization_20260909/iid_vs_heading_prior.pdf), [SVG](../output/mini_heading_prior/noise_visualization_20260909/iid_vs_heading_prior.svg), and [offline interactive viewer](../output/mini_heading_prior/noise_visualization_20260909/interactive.html).

The figure uses actual paired source arrays from the secondary sampling check. Its example is the first stored observation and first draw of seed 42 (episode 59, frame 16622), selected by index. The upper panels are cumulative sums of 16 XY noise vectors, not robot trajectories or generated actions. A single rotation changes the chunk resultant from -164.01° to 48.00°, around the predicted heading 52.05° with recorded jitter -4.06°. Each step's XY magnitude and all five unselected dimensions remain unchanged to floating-point precision.

The angular histogram uses seed 42 only: 200 observations × 4 source draws, with angles relative to each observation's own predicted heading. Within ±30°, IID has 16.625% and the prior has 71.125% of the samples. Mean absolute source-angle deviations are 88.258° and 23.490°. These describe the initial noise relative to the predictor, not generated-action error against demonstrations. The figure's ±4.06° example is not a claim about typical jitter.

The interactive viewer includes both training seeds and the same 800 paired base draws per seed; these must not be interpreted as 1,600 independent random draws. Controls select seed, observation and noise draw, while a slider interpolates the single rotation. At the endpoints it displays the exact saved IID/prior arrays. It is standalone HTML and requires no network connection.

Reproduce with `python scripts/visualize_heading_prior_noise.py` and `python scripts/visualize_heading_prior_interactive.py`. The output directory contains source snapshots, input hashes and numerical validation. Actual generated JavaScript passed interaction checks in a mocked DOM/canvas; a full browser screenshot test was unavailable in this environment.

## Fixed-model source-angle diagnostic (2026-09-09)

This follow-up tests whether inaccurate heading leaves room for improvement in the existing translation-XY prior. It reuses the two 5,000-update `condition_prior_xy` EMA checkpoints and the exact 200 validation windows and four saved Gaussian/jitter streams from the secondary sampling check. The observation encoder runs once per batch, and its original predicted heading condition is passed unchanged to every intervention. No model is retrained.

Only the source-angle input changes: predicted heading, GT heading, or GT plus signed 15°, 30°, or 60°. The error sign is fixed per draw/window and shared across error levels and training seeds. The original normalizer, predicted-validity gate, κ=4 Von Mises angular jitter, and 10 Euler steps remain fixed. GT is the original 16-action raw XY resultant. Of 200 windows, 199 have valid heading targets; the remaining window retains the original source and generated actions exactly in every arm. The source gate is active for all saved samples.

| Source-angle input | Prefix-8 action MSE | Prefix-8 XYZ MSE | Generated 16-step heading MAE |
| --- | ---: | ---: | ---: |
| Original prediction | 0.141372 | 0.139278 | 47.54° |
| GT | 0.137352 | 0.129041 | 31.14° |
| GT ±15° | 0.138785 | 0.132548 | 34.03° |
| GT ±30° | 0.143387 | 0.142675 | 42.09° |
| GT ±60° | 0.159963 | 0.179812 | 66.53° |

Values average errors over all four draws and both training seeds. GT lowers overall action MSE by **2.84%** and XYZ MSE by **7.35%**. Both checkpoints improve: action-MSE differences (GT minus prediction) are −0.002952 for seed 42 and −0.005087 for seed 43. The mean paired action effect is −0.004019, with a 95% episode-bootstrap interval [−0.006516, −0.001295]. These intervals resample episodes within tasks and condition on the two fixed checkpoints and four fixed noise streams; they do not measure training-seed uncertainty. Action, XYZ, and generated-heading errors all rise monotonically with the 0/15/30/60° injected error in each checkpoint.

Descriptive strata show the largest action improvement in windows whose original predictor error exceeds 60°; the <15° group is essentially unchanged. Subgroup membership is fixed separately for each checkpoint, and sparse-bin intervals are suppressed. Gripper MSE has no clear improvement.

**The result supports investigating better source-heading prediction in this setting.** It does not establish that an improved, retrained prior will beat IID or raise SR. GT uses future demonstration labels and changes the source distribution of models trained on predicted angles. The original, possibly incorrect predicted heading remains in the condition. Also, GT means the *center* of the angular source distribution is correct: the retained κ=4 jitter still gives the actual initial noise a mean heading error of 23.56°. This is not a zero-jitter source or a strict performance upper bound.

The evaluation took about 6.2 seconds, excluding data preparation. All original predicted-source arrays and five saved action metrics reproduce the preceding sampling check bit for bit. Thirteen new diagnostic CPU tests and 31 existing shared-policy tests passed, covering condition isolation, original gate preservation, invalid-label fallback, circular-angle offsets, XY norm preservation, and unchanged non-XY source dimensions.

Reproduce with the `oat` environment and a new output directory (select an available GPU through `CUDA_VISIBLE_DEVICES` if needed):

```bash
python scripts/evaluate_heading_angle_sensitivity.py \
  --run-dir output/mini_heading_prior/mini_5000_20260909 \
  --paired-source-dir output/mini_heading_prior/sampling_robustness_20260909 \
  --output-dir output/mini_heading_prior/angle_sensitivity_20260909

python scripts/summarize_heading_angle_sensitivity.py \
  --run-dir output/mini_heading_prior/angle_sensitivity_20260909 --pdf
```

Evidence: [full report](../output/mini_heading_prior/angle_sensitivity_20260909/report.md), [angle-sensitivity figure](../output/mini_heading_prior/angle_sensitivity_20260909/angle_sensitivity.png), [predictor-error strata](../output/mini_heading_prior/angle_sensitivity_20260909/accuracy_strata.png), [machine-readable summary](../output/mini_heading_prior/angle_sensitivity_20260909/summary.json), and [independent artifact audit](../output/mini_heading_prior/angle_sensitivity_20260909/verification.json). The independent audit reconstructs GT labels from the dataset and recomputes all 8,000 generated-action errors. The output also saves exact source snapshots, input/checkpoint hashes, paired source/action arrays, and the completion record. Implementation: [evaluator](../scripts/evaluate_heading_angle_sensitivity.py), [reporter](../scripts/summarize_heading_angle_sensitivity.py), and [tests](../tests/test_heading_angle_sensitivity.py).

## GT-source condition × jitter factorial (2026-09-09)

This is the follow-up 2×2 diagnostic to separate the effect of retained source-angle jitter from the effect of the explicit heading condition. All four cells use the GT source center on valid labels. The factors are original predicted versus GT heading condition, and original saved κ=4 angular jitter versus explicit zero jitter. The predicted-condition/original-jitter cell reproduces the preceding oracle-source experiment; it is not the fully predicted deployment baseline.

GT condition replaces only the explicit cos/sin pair in each observation token. The shared observation features, predicted-validity channel, source gate, model weights, Min–Max normalizer, Gaussian samples, and Euler solver remain unchanged. Zero jitter removes only angular dither; the Gaussian noise still varies across draws. The same two checkpoints, 200 windows, and four noise streams are reused. For the single invalid GT target, all four cells retain the original predicted source angle, original jitter, and original condition.

| Heading condition | Source jitter | Prefix-8 action MSE | Prefix-8 XYZ MSE | Generated 16-step heading MAE |
| --- | --- | ---: | ---: | ---: |
| Predicted | Original κ=4 | 0.137352 | 0.129041 | 31.14° |
| Predicted | Zero | 0.131390 | 0.115072 | 15.60° |
| GT | Original κ=4 | 0.142568 | 0.128623 | 32.64° |
| GT | Zero | 0.136482 | 0.114723 | 15.94° |

With predicted condition, removing jitter lowers action MSE by **4.34%**, XYZ MSE by **10.83%**, and generated heading MAE from **31.14° to 15.60°**. Both training seeds improve, and removing jitter also improves action/XYZ/heading errors under GT condition. The paired action effect with predicted condition is −0.005962, 95% episode-bootstrap interval [−0.007285, −0.004527]; the heading effect is −15.533°, interval [−16.847°, −14.334°].

Replacing the explicit condition with GT does not improve mean full-action error in these checkpoints: action MSE rises by 0.005216 with jitter and by 0.005092 without it. Both seeds have the same sign, but the corresponding episode intervals include zero, so this is not evidence of established general harm. XYZ is almost unchanged; mean gripper MSE rises from 0.558121 to 0.596092 with jitter and from 0.558347 to 0.595220 without jitter. The full-action interaction, `(GT-zero − GT-jitter) − (predicted-zero − predicted-jitter)`, is −0.000124 with interval [−0.001020, +0.000723]. Conditional intervals average the fixed noise draws and checkpoints before resampling episodes within tasks; they do not quantify training-seed uncertainty.

The source center is exactly GT in all valid-label cells. Actual initial noise heading MAE is 23.56° with saved jitter and approximately 0.000006° with zero jitter, yet final generated heading MAE remains 15.60°/15.94° in the zero-jitter cells. This demonstrates that precise source alignment does not force the generated direction to match. It does not isolate a unique cause of the residual error, and source/final MAEs cannot be subtracted to obtain an additive causal error budget. The experiment did not support the hypothesis that replacing the explicit predicted heading condition with GT would remove the remaining error.

**Removing dither helps when the source center is supplied by GT in this fixed-model diagnostic.** It does not establish that deterministic alignment around an imperfect predicted heading will help after matched training. GT inputs and zero-jitter sources change the training distribution, and the observation features still contain their original heading information. No retraining, rollout, or SR measurement was performed.

Inference took about 5.4 seconds, excluding preparation. Thirteen new factorial CPU tests plus the previous 44 tests all passed. The existing oracle cell reproduces source tensors, generated actions, and every saved diagnostic metric bit for bit. An independent audit reconstructed GT labels from Zarr and recomputed all **6,400** generated-action samples, checking that only two condition channels changed, source arrays are identical across the condition factor, raw XY norms and other source dimensions are preserved, and invalid-label fallbacks are exact.

```bash
python scripts/evaluate_heading_factorial.py \
  --reference-dir output/mini_heading_prior/angle_sensitivity_20260909 \
  --output-dir output/mini_heading_prior/factorial_20260909

python scripts/summarize_heading_factorial.py \
  --run-dir output/mini_heading_prior/factorial_20260909 --pdf
```

Select an available GPU through `CUDA_VISIBLE_DEVICES` and use a new output directory for another run. Evidence: [report](../output/mini_heading_prior/factorial_20260909/report.md), [four-cell figure](../output/mini_heading_prior/factorial_20260909/factorial_cells.png), [paired contrasts](../output/mini_heading_prior/factorial_20260909/factorial_effects.png), [summary](../output/mini_heading_prior/factorial_20260909/summary.json), and [independent verification](../output/mini_heading_prior/factorial_20260909/verification.json). Code: [evaluator](../scripts/evaluate_heading_factorial.py), [reporter](../scripts/summarize_heading_factorial.py), and [tests](../tests/test_heading_factorial.py).

## Training with predicted heading and zero angular jitter

The factorial result 0.131390 action MSE / 15.60° generated heading error used a **GT-centered source**, predicted heading condition, and zero angular jitter. It is not a measured score for a deployable predicted-heading source. To train the deployable counterpart, use the shared jointly trained head for both condition and source, and explicitly disable angular dither:

```bash
cd /home/haotian/code/fm_dno
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/home/haotian/miniforge3/envs/oat/bin/python scripts/mini_heading_prior.py \
  --modes condition_prior_xy \
  --source-jitter zero \
  --seeds 42 43 \
  --steps 5000 --eval-every 1000 \
  --train-per-task 400 --val-per-task 100 --sample-per-task 20 \
  --output-dir output/mini_heading_prior/predicted_zero_5000_seed42_43
```

This retains the mini protocol: 4,000 fixed training windows, 1,000 validation windows, two separately trained seeds, 5,000 updates per seed, the original train-only Min–Max normalizer, and the shared head trained through condition and auxiliary losses. Only source construction detaches the predicted direction. Future GT directions are supervision targets, not source inputs. This is a mini trainer, not a full-dataset training configuration.

`--source-jitter zero` supplies exact zero angular offsets in training and evaluation and saves `source_jitter` in checkpoint metadata. `load_policy` restores this setting for default inference and secondary sampling. The default `vonmises` mode retains the previous behavior; legacy checkpoint metadata without this field resolves to `vonmises`. Explicit `angular_jitter` tensors still override the policy default for controlled diagnostics. `--source-kappa 0` is uniform angular noise and must not be used to request zero jitter. Gaussian noise and the original source validity gate remain active.

The CLI and a single-update real-data GPU smoke check passed; 69 CPU tests passed, including zero-jitter training/inference parity, checkpoint restoration, and legacy behavior. The full mini command above has now completed; its matched results are recorded below. Historical experiment source snapshots remain unchanged; earlier evaluators that enforce exact source hashes must use their recorded source versions when reproducing those historical runs.

## Completed predicted-heading zero-jitter mini (2026-09-09)

The two zero-jitter models were trained from scratch for the prescribed 5,000 updates per seed, using seeds 42/43 and the exact previous 4,000/1,000 training/validation windows. No early stopping or best-checkpoint selection was used. Training took 314.0 seconds excluding data preparation. The cross-run comparison then evaluated the new models and the previous IID/κ=4 controls on the exact 200 observations and four saved Gaussian arrays, taking 6.6 seconds excluding preparation. All three arms use predicted heading condition; both directional sources use predicted directions, never future labels.

| Trained source | Prefix-8 action MSE | Prefix-8 XYZ MSE | Generated heading MAE | Predictor MAE, full validation |
| --- | ---: | ---: | ---: | ---: |
| IID | 0.140088 | 0.140357 | 51.42° | 34.85° |
| Predicted XY prior, κ=4 | 0.141372 | 0.139278 | 47.54° | 35.15° |
| Predicted XY prior, zero jitter | 0.137806 | 0.131320 | 44.81° | 34.98° |

Generated-action metrics average four draws and two training seeds; predictor MAE uses valid targets in the original full 1,000-window validation set. All errors are scored before averaging. XYZ MSE improves in both seeds versus both controls. Generated direction also improves in both seeds, although the gain over κ=4 is only 0.42° for seed 42.

Overall action gains remain inconsistent across training seeds:

| Seed | IID action MSE | κ=4 action MSE | Zero-jitter action MSE |
| --- | ---: | ---: | ---: |
| 42 | 0.137924 | 0.142140 | 0.143878 |
| 43 | 0.142252 | 0.140604 | 0.131734 |

The new mean action MSE is 1.63% below IID and 2.52% below κ=4, but seed 42 worsens against both controls while seed 43 improves. Against IID, the paired mean effect is −0.002282, with conditional episode-bootstrap 95% interval [−0.006953, +0.002418]. Against κ=4, it is −0.003566, interval [−0.006716, −0.000253]. These intervals average the two fixed seeds and four fixed noise draws before resampling whole episodes within task; even an interval excluding zero does not establish robustness over training seeds. Gripper errors change differently across seeds and offset some of the consistent XYZ improvement.

The original single-noise training endpoint is preserved separately: zero-jitter mean action MSE is 0.140133 versus IID 0.139015 and κ=4 0.150203. It is slightly worse than IID under that noise set, reinforcing that a stable overall advantage over IID has not been established. Predictor error remains about 35°; removing jitter did not show a clear improvement in the heading estimator itself.

**Conclusion: matched zero-jitter training improves XYZ and generated heading in this mini, but does not establish a consistent overall action-quality gain over IID or κ=4 across seeds.** There is no SR measurement. The earlier 0.131390 / 15.60° result remains a separate GT-source intervention on the old κ=4-trained model; it is not the new trained model's result.

The cross-run evaluator verifies identical split metadata, normalization tensors, initialization diagnostics, and all training arguments except output path, selected mode list, and jitter. New code defaults preserve legacy behavior: all four old control checkpoints exactly replay saved source arrays and action metrics; κ=4 generated action tensors additionally replay the saved angle-sensitivity controls. The fixed comparison plan was recorded before training finished. Sixty-nine CPU tests had passed before this run; final saved arrays and checkpoints are independently audited.

Reproduce the comparison after training:

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/home/haotian/miniforge3/envs/oat/bin/python scripts/compare_zero_jitter_training.py \
  --zero-run-dir output/mini_heading_prior/predicted_zero_5000_seed42_43 \
  --output-dir output/mini_heading_prior/zero_jitter_comparison_20260909 --pdf
```

Use a new output directory for another evaluation. Existing results can be rerendered with the same script's `--report-only --output-dir ... --pdf`, without inference. Evidence: [full report](../output/mini_heading_prior/zero_jitter_comparison_20260909/report.md), [comparison figure](../output/mini_heading_prior/zero_jitter_comparison_20260909/comparison.png), [summary](../output/mini_heading_prior/zero_jitter_comparison_20260909/summary.json), [training summary](../output/mini_heading_prior/predicted_zero_5000_seed42_43/summary.json), [comparison plan](../output/mini_heading_prior/predicted_zero_5000_seed42_43/comparison_plan.json), and [independent verification](../output/mini_heading_prior/zero_jitter_comparison_20260909/verification.json). The audit independently recomputed all 4,800 generated-action samples and all ten paired bootstrap intervals. Code: [cross-run evaluator and reporter](../scripts/compare_zero_jitter_training.py).

