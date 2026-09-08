# Task-guided DNO for E/F

These policy wrappers add task-directed initial-noise search to an already trained
E or F checkpoint. They are independent of `OrbitDNO`; the base policy, its visual
encoder, heading predictor, and normalizers remain frozen. No base-policy retraining
is needed. They also work with alternative flow backbones exposing the existing
E/F sampling interface; each initializer is tied to its exact base checkpoint.

## Versions

Each row can be applied to E or F. `reference_use=condition` identifies E;
`reference_use=both` identifies F. A checkpoint from one is never relabeled as the other.

| Version / `--modes` entry | Behavior | Extra training |
| --- | --- | --- |
| `baseline` | One draw from that policy's original source | None |
| `best_of_k` | Independent candidates, choose lowest task cost | None |
| `dno` | Refine candidate noises through the full Euler sampler | None |
| `amortized` | Learned noise initializer, checked against the original candidate | Teacher distillation |
| `amortized_dno` | Learned initialization followed by DNO | Teacher distillation |

The learned versions keep a source candidate as a fallback. `amortized` therefore
uses two decoded candidates and the task scorer, not just a single actor forward.
The current learner distills noise-search results; it does **not** implement online
actor–critic reinforcement learning or train a learned success critic.

For E, the initial source is IID Gaussian. For F, the initial source is canonicalized
using the current frozen heading prediction and a fresh angular dither. Each source
is constructed **once**; DNO then optimizes the resulting noise directly. It does not
re-canonicalize or redraw dither during gradient updates. The heading condition stays
fixed during a solve. Changing source noise is not the same as updating that condition.

The gradient loss is task cost plus `trust_weight * mean(((z-z_base)/sigma)^2)`.
After each update, the noise is projected to an RMS radius
`trust_radius * sigma` around its own initial draw. The best finite task-cost candidate
seen across search iterations is retained, including the original candidates.
This guarantees only a no-worse **proxy score**, not no-worse physical execution.

Each candidate has an independent RNG seed; candidate zero is identical across
baseline, best-of-K, and DNO at the same control cycle. The optimizer preserves the
caller's Torch RNG state. Conditions are encoded once per cycle and reused.

## First task evaluator: privileged pick-and-place geometry

The initial scorer reads the live simulator's BDDL predicates, source-object and
target-site positions, actual gripper contact, and actual OSC controller scaling.
Its five phases are reach, grasp, lift, transport, and release. These phases update
only after real environment execution. During a noise solve, the context is fixed.

The cost predicts end-effector displacement by integrating the **executed prefix**
of unnormalized OSC commands after controller input clipping and scaling. It penalizes
subgoal distance and a phase-dependent gripper command. This approximation does not
model object dynamics, grasp orientation, collision geometry, rolling, or slip.
The actual episode success label always comes from LIBERO after real actions.

Six LIBERO-10 tasks with pure independent On/In placement predicates are supported:

- `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy`
- `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket`
- `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`
- `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket`
- `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate`
- `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate`

Close/Turnon and dependent stacking goals are explicitly rejected. This first evaluator
is an **oracle geometry experiment**, not a complete LIBERO-10 or image-only deployment
method. New object categories and controlled OOD position shifts are not established by
these ordinary-seed evaluations. The goal adapter must be extended and evaluated for
those settings.

## Run a matched E comparison

From the repository root in the `oat` environment, use a completed epoch checkpoint
instead of a `latest.ckpt` that another training process may be overwriting. The
available epoch-200 E checkpoint is:

```text
output/learned_canon/condition/seed42/checkpoints/ep-0200_sr-0.228.ckpt
```

The wrapper does not alter this checkpoint or interact with W&B. Both GPUs were busy
when the versions were implemented; the following CPU/software-rendering command can
run alongside those jobs:

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
python scripts/eval_task_dno_libero.py \
  --config oat/config/task_dno/e_compare.yaml \
  --checkpoint output/learned_canon/condition/seed42/checkpoints/ep-0200_sr-0.228.ckpt \
  --task-name LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket \
  --output-dir output/task_dno/e_ep0200_basket_seed2000 \
  --seed 2000 --episodes 20 --device cpu --cpu-threads 2
```

When system GPU 1 is available, replace the environment prefix with
`CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`
and use `--device cuda:0`.

For F, select `oat/config/task_dno/f_compare.yaml` and an independently trained F
checkpoint. There was no trained F checkpoint available at implementation time;
F source construction is covered by tests, not a claimed closed-loop F result.

For a pipeline-only smoke test, use `oat/config/task_dno/pilot.yaml`. Its 16-step
horizon is intentionally too short to measure task success.

The full comparison presets use 20 episodes per mode and 550 environment steps.
They retain the checkpoint's observation length, 16-action prediction horizon,
8-action execution prefix, and Euler step count. All modes get the same episode
reset seeds. The script reseeds the underlying simulator, observes after objects
settle, and checks a hash of the complete initial simulator state and fixture poses;
a mismatch aborts the comparison.

`best_of_k` defaults to `K * (gradient_steps + 2)` candidates, matching `dno`'s
candidate-weighted Euler **forward** NFE. DNO additionally performs backward passes,
so this is not equal total computation. Both NFE and measured latency are reported.
For `amortized_dno`, matching forward NFE requires
`--best-of-k-candidates K*(gradient_steps+3)` as a numeric value. For example,
K=2 and two gradient steps use best-of-K=8 for DNO, or 10 for amortized DNO.

Every episode is written immediately. The output directory must be new. Artifacts:

- `manifest.json`: checkpoint SHA-256, E/F identity, task, parameters, oracle declaration.
- `episode_<mode>_<id>.json`: success, actual steps, initial-state hash, per-cycle latency and policy statistics.
- `results.json`: per-mode success rate, latency percentiles, forward NFE and backward-step counts.
- `teacher_data.pt`: noise-search examples and real executed-feedback fields, when a teacher mode is enabled.

## Learn an initializer from DNO searches

The comparison presets export DNO teacher data. For each chosen result, the archive
stores the corresponding original candidate noise, optimized noise, cached condition,
nine task features, and episode ID. It also records the **actually executed** action
prefix and length, current/next environment step, chunk return, and episode success.
The unexecuted suffix is not treated as an observed transition. The `accepted` flag
means the teacher improved its own candidate's proxy cost; it does not mean task success.
Best-of-K alone produces no residual-search labels and is not a teacher mode.

The initializer consumes the original random noise as well as context; it does not
average unrelated optimal noises for one observation. It starts as the identity map,
adds a bounded residual, and is trained against the optimized noise. Train/validation
splits are by complete episode, and feature statistics use training episodes only.
The identity model is retained as `best.pt` if all learned updates worsen validation.

After collecting enough completed **successful** teacher episodes, train:

```bash
python scripts/train_noise_initializer.py \
  --data output/task_dno/e_ep0200_basket_seed2000/teacher_data.pt \
  --output-dir output/task_dno/e_ep0200_initializer \
  --successful-only --epochs 50 --device cpu
```

At least two retained episodes are required. A four-example smoke archive can check
the training path but is not sufficient evidence of a useful learned initializer.
Omitting `--successful-only` distills proxy-improving searches even from unsuccessful
episodes; that is useful for plumbing checks, not a substitute for execution validation.

Evaluate the learned versions on **new seeds**, including the base policy again:

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa CUDA_VISIBLE_DEVICES='' \
python scripts/eval_task_dno_libero.py \
  --config oat/config/task_dno/amortized_compare.yaml \
  --checkpoint output/learned_canon/condition/seed42/checkpoints/ep-0200_sr-0.228.ckpt \
  --initializer output/task_dno/e_ep0200_initializer/best.pt \
  --task-name LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket \
  --output-dir output/task_dno/e_ep0200_initializer_eval_seed3000 \
  --seed 3000 --episodes 20 --device cpu
```

Initializer format v2 normalizes only observation features and relative goal positions,
using variance floors and bounded normalized inputs. The source noise and discrete
phase/gripper features retain their original scale. This avoids amplifying unseen
phases or nearly constant features when the teacher archive is small. Format v1
checkpoints are rejected explicitly.

An initializer checkpoint stores its architecture, feature normalizer, E/F source
identity, and exact base-checkpoint SHA-256. Loading with a different base policy fails,
including when it has the same tensor dimensions but different learned weights.

## Completed initial experiment (2026-09-08)

A complete-episode pilot used the epoch-200 E EMA checkpoint above, the
`LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket`
task, seeds 1000–1003, and a 550-step limit. Initial simulator-state hashes matched
across all three modes. These are ordinary randomized resets, not a controlled OOD
pose split. CPU inference used two Torch threads and OSMesa rendering.

| Mode | Successful episodes | Median policy latency / chunk | Forward NFE / chunk | Backward steps / chunk |
| --- | --- | --- | --- | --- |
| E baseline | 2/4 | 63.7 ms | 10 | 0 |
| E + best-of-8 | 4/4 | 105.4 ms | 80 | 0 |
| E + DNO (2 candidates, 2 updates) | 3/4 | 247.7 ms | 80 | 2 |

These are preliminary results from one task and four seeds with privileged geometry.
Candidate selection performed best in this pilot; there is no evidence here that
gradient DNO beats additional independent sampling. Broader independent-seed tests
and F checkpoints are needed before choosing a final method or making generalization
claims. Latency measures the policy call, excluding simulator execution/rendering.

Results and per-episode records:
`output/task_dno/e_ep0200_basket_pilot_seed1000/results.json`.
Teacher data: the same directory's `teacher_data.pt` (169 actual control cycles).
To reproduce the pilot in a fresh directory, use the E comparison command above with
`--seed 1000 --episodes 4 --num-candidates 2 --num-grad-steps 2`.

A v2 initializer was trained on the 100 teacher examples from the three successful
DNO episodes. Episode IDs 0 and 3 were training data (64 examples); episode 2 was
validation data (36 examples). The selected checkpoint was epoch 32, with validation
noise MSE 0.008382 versus 0.009146 for the identity initializer (8.35% lower).
This measures teacher imitation, not task success or generalization. The checkpoint
is `output/task_dno/e_ep0200_initializer_pilot_seed1000/best.pt`.

The actual v2 initializer was loaded and executed with `baseline`, `amortized`, and
`amortized_dno` on fresh seeds 2000/2001. Their initial-state hashes matched and the
learned residuals were nonzero. Results are in
`output/task_dno/e_ep0200_initializer_smoke_seed2000/results.json`. Those episodes
were limited to 16 steps, so they establish compatibility only, not success gains.

The original pilot and smoke artifacts predate the timeout-label audit. They inherit
LIBERO's combined `done` field, so a failed episode ending at the step limit can be
marked `terminated=true, truncated=false`. The recorded success, reward and actual
executed prefixes remain valid; those historical termination flags must be corrected
before using these archives for critic targets. Current evaluations distinguish
unsuccessful time limits from terminal success and preserve the underlying environment
flag separately. The initializer training above does not use termination flags.

## Validation

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
python -m pytest -q tests/test_task_noise_policy.py tests/test_task_objectives.py \
  tests/test_noise_initializer.py tests/test_task_dno_eval.py
```

Tests cover differentiable source optimization, correct F canonical source,
fixed candidate-zero initialization, source randomness drawn only once, frozen
weights, trust-radius bounds, rejection of nonfinite learned actions, real worker-free
rollout bookkeeping, identical reset states, real executed prefixes, checkpoint
identity checks, and episode-level learning splits. Short real-checkpoint rollouts
exercise all five versions with CPU inference and software rendering.
