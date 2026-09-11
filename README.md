# FM Heading

Flow-matching robot policies that predict an XY motion heading from observations
and use it both to condition the velocity model and to shape its initial noise.
The repository supports **observation-only baselines**, **Heading Zero** and
**Heading Gaussian**, each with **Transformer, U-Net, or StarVLA-DiT** backbones.
The baselines use `FlowPolicy` with Gaussian source noise and no heading head,
heading condition or auxiliary heading losses.

LIBERO-10 is the verification benchmark. The heading method uses demonstration
motion labels and observation features, without task-specific geometry or scripted
motion phases. Evaluation uses one policy checkpoint across all tasks.

## Experiment results

Best SR among completed 500-episode LIBERO-10 checkpoint evaluations for each
policy. SR = success rate; DiT = StarVLA-DiT.

| Policy used | SR |
|---|---:|
| [Baseline / Transformer](output/eval_comparison_ep50_ep55_20260910T234741Z/baseline_transformer/summary.json) | 76.8% |
| [Baseline / U-Net](output/eval_comparison_ep50_ep55_20260910T234741Z/baseline_unet/summary.json) | 73.6% |
| [Baseline / DiT](output/eval_comparison_ep50_ep55_20260910T234741Z/baseline_dit/summary.json) | 81.8% |
| [Heading Zero / Transformer](output/eval_comparison_ep50_ep55_20260910T234741Z/headingzero_transformer/summary.json) | 74.2% |
| [Heading Zero / U-Net](output/eval_comparison_ep50_ep55_20260910T234741Z/headingzero_unet/summary.json) | 75.4% |
| [Heading Zero / DiT](output/heading_goal/starvla_dit_150_online_20260910/rollouts/epoch_030/attempt_001/summary.json) | 80.0% |
| [Heading Gaussian / Transformer](output/20260911/001720_train_flowpolicy_headinggaussian_transfomer_libero10_N500/rollouts/epoch_040/attempt_001/summary.json) | 75.0% |
| [Heading Gaussian / U-Net](output/20260911/001648_train_flowpolicy_headinggaussian_unet_libero10_N500/rollouts/epoch_020/attempt_001/summary.json) | 76.8% |
| [Heading Gaussian / DiT](output/20260911/001512_train_flowpolicy_headinggaussian_starvlaDiT_libero10_N500/rollouts/epoch_060/attempt_001/summary.json) | 79.0% |

## Model design

```text
RGB + robot state → shared observation encoder → observation features
                                      └───────→ heading head → direction + validity
                                                                  │
                          ┌───────────────────────────────────────┤
                          ▼                                       ▼
             concatenate with observations               shape Gaussian noise
                          │                                       │
                          └────────→ flow velocity model ←────────┘
                                             │
                                    Euler integration
                                             │
                                      action chunk
```

The heading head predicts a unit XY direction and a motion-validity probability
from two observation frames. Supervision comes from the summed XY translation
commands over the future demonstration chunk; it does not represent robot yaw.
Both predictions are appended to each observation feature token.

| Policy | Noise prior |
|---|---|
| **Heading Zero** | Rotates the sampled raw XY chunk so its summed direction matches the predicted heading. “Zero” means zero angular jitter; the source is still random. |
| **Heading Gaussian** | Constructs a full-rank XY Gaussian with a heading-aligned mean and greater variance along the heading than perpendicular to it. The realized direction remains stochastic. |

Both fall back to ordinary Gaussian noise when predicted validity is low. Heading
is detached for source construction, but the conditioning and auxiliary losses
train the shared encoder and head jointly. Training and inference use the same
predicted-heading source rule.

```text
x1 = normalized demonstration actions
x0 = heading-dependent noise source
t ~ Uniform(0, 1)
xt = (1-t)*x0 + t*x1
loss = MSE(v(xt, t, condition), x1-x0)
       + 0.1 * heading cosine loss + 0.1 * validity BCE
```

Inference integrates the velocity field with ten Euler steps, predicts a 16-action
chunk, and executes eight actions before observing again. The current configs use
a shared ResNet18-based RGB/state encoder; no tokenizer, external heading-reference
checkpoint, DINOv2 encoder, or DNO optimization stage is required.

See [the detailed design](docs/heading_policies.md) for target construction,
normalization, gradient paths, prior equations and implementation inheritance.
See [the backbone guide](docs/flow_backbones.md) for architecture differences.

## Setup

Run commands from the repository root with a Python environment containing this
checkout. GPU training and headless LIBERO rollouts require compatible PyTorch/CUDA
and EGL libraries.

```bash
git clone --recursive https://github.com/seanliu7081/fm_dno.git
cd fm_dno
git submodule update --init --recursive
uv sync
source .venv/bin/activate
export PYTHONPATH="$PWD"
python -c "import torch, oat, libero; print(torch.__version__, torch.cuda.is_available())"
```

For an existing environment, install both packages with
`pip install -e third_party/LIBERO` and `pip install -e .`.
On the current workspace instance the prepared environment is `/venv/oat`.

Before simulator evaluation:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python -c "from libero.libero import get_libero_path; print(get_libero_path('bddl_files')); print(get_libero_path('init_states'))"
```

The printed LIBERO paths must point to installed task assets and saved initial states.

## Dataset

The default task config reads `data/libero/libero10_N500.zarr`. If this dataset is
already available, skip conversion. Otherwise place the ten LIBERO-10 task demo
HDF5 files in `data/libero/hdf5_datasets/`, then run:

```bash
python scripts/convert_libero_dataset.py -n 50
python scripts/compose_libero_multitask_dataset.py -mt libero10
```

Use a directory containing one intended per-task Zarr variant: the composition
script selects the first matching `<task>_N*.zarr` for each task. Fifty demonstrations
per task produce the `libero10_N500.zarr` file expected by the defaults.
`training.num_demo=500` selects that filename; it does not itself subsample episodes.

`HeadingZarrDataset` splits episodes into training and validation sets and fits
action/state normalization and heading-label scale using training episodes only.
RGB normalization uses the fixed pixel range. Inference restores the learned
weights and normalization statistics from the checkpoint.

## Standalone training configurations

Each config contains its full policy and training settings and composes only the
task configuration. The existing `transfomer` filename spelling is preserved.

| Prior | Transformer | U-Net | StarVLA-DiT |
|---|---|---|---|
| Baseline Gaussian | [train_flowpolicy_transformer](oat/config/train_flowpolicy_transformer.yaml) | [train_flowpolicy_unet](oat/config/train_flowpolicy_unet.yaml) | [train_flowpolicy_starvlaDiT](oat/config/train_flowpolicy_starvlaDiT.yaml) |
| Heading Zero | [train_flowpolicy_headingzero_transfomer](oat/config/train_flowpolicy_headingzero_transfomer.yaml) | [train_flowpolicy_headingzero_unet](oat/config/train_flowpolicy_headingzero_unet.yaml) | [train_flowpolicy_headingzero_starvlaDiT](oat/config/train_flowpolicy_headingzero_starvlaDiT.yaml) |
| Heading Gaussian | [train_flowpolicy_headinggaussian_transfomer](oat/config/train_flowpolicy_headinggaussian_transfomer.yaml) | [train_flowpolicy_headinggaussian_unet](oat/config/train_flowpolicy_headinggaussian_unet.yaml) | [train_flowpolicy_headinggaussian_starvlaDiT](oat/config/train_flowpolicy_headinggaussian_starvlaDiT.yaml) |

The baseline configs match the heading-zero training settings and use
`TrainSplitZarrDataset` for the same training-only normalization and fixed data
split, without heading statistics. See [the baseline guide](docs/flow_backbones.md#observation-only-baselines)
for all three commands. For example:

```bash
python scripts/run_workspace.py --config-name=train_flowpolicy_unet
```

For example, train Heading Gaussian with U-Net on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_workspace.py \
  --config-name=train_flowpolicy_headinggaussian_unet
```

To configure a 150-epoch Heading Zero / StarVLA-DiT run with online W&B logging:

```bash
wandb login
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online python scripts/run_workspace.py \
  --config-name=train_flowpolicy_headingzero_starvlaDiT \
  training.num_epochs=150 logging.mode=online
```

This command retains external evaluation (`lazy_eval=true`); it does not schedule
500-episode rollouts during training. Run long experiments through your process
manager or scheduler. Commands shown here do not resume the previously stopped run
unless its existing output directory is explicitly selected.

| Setting | Standalone defaults |
|---|---|
| Seed / batch size | 42 / 64 |
| Configured epochs | 61 |
| Policy / encoder / heading learning rate | 1e-4 each |
| Scheduler | 500-update warmup, then constant |
| Weight decay / gradient clipping | 1e-6 / norm 1.0 |
| EMA | Enabled |
| Observation / prediction / execution horizon | 2 / 16 / 8 |
| Euler steps / noise scale | 10 / 1.0 |
| RGB crop / dropout | 116 × 116 / 0.0 |
| In-training rollouts / W&B | Disabled (`lazy_eval=true`) / offline |

Hydra writes the resolved run configuration, checkpoints and logs below `output/`.
Use the same data split, training budget, prior family and evaluation protocol for
backbone comparisons; architecture parameter counts differ.

## Evaluate one checkpoint over 500 episodes

Choose a completed checkpoint and a new output directory. The following evaluates
50 saved initial states for each of the ten LIBERO-10 tasks on physical GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/eval_heading_policy.py \
  --checkpoint output/YOUR_RUN/checkpoints/latest.ckpt \
  --output output/eval/YOUR_RUN_500 \
  --suite libero_10 --n-per-task 50 --init-start 0 \
  --seed 20260911 --settle-steps 10 --max-episode-steps 550 \
  --workers 4 --device cuda:0 --egl-device 1
```

Replace `YOUR_RUN` with the actual run path. The evaluator copies one immutable
checkpoint, selects its configured EMA/model weights, and uses those weights for
every task. It restores LIBERO's supplied initial states, applies ten settling
actions `[0,0,0,0,0,0,-1]` (gripper open), and fetches a fresh observation afterward.
Environment and policy-noise seeds are deterministically derived from the base seed.

This protocol is not identical to older evaluations using five settling steps,
sequential reset seeds, ordinary random resets, or the legacy evaluator. Match the
saved protocol before comparing success rates. “Official” in the filenames refers
to the supplied LIBERO initial states.

Outputs include `policy.ckpt`, `manifest.json`, per-episode `episodes.jsonl`, and
`summary.json`. The manifest records the protocol and checkpoint hash. Incomplete
evaluations do not report a complete benchmark success rate. Add `--dry-run` to
validate the episode plan and snapshot without starting simulator rollouts.

The evaluation files have distinct roles:

| File | Role |
|---|---|
| [libero_official_eval.py](oat/env_runner/libero_official_eval.py) | Checkpoint loading, saved-state initialization, seeded episode rollouts and aggregation. |
| [libero_official_runner.py](oat/env_runner/libero_official_runner.py) | Training adapter: snapshots the selected policy, synchronously launches checkpoint evaluation, verifies artifacts and returns metrics. |
| [libero_runner.py](oat/env_runner/libero_runner.py) | Original vector-environment runner, inherited by task configs but inactive while `lazy_eval=true`; uses legacy reset behavior. |

For synchronous checkpoint evaluation during training, configure
`LiberoOfficialRunner` with its own constructor fields and set `lazy_eval=false`.
Changing only `lazy_eval` on a standalone config enables the original runner.
`training.rollout_every` controls the interval; scheduling details and the saved
150-epoch experiment are described in [the design guide](docs/heading_policies.md).
Historical results are preserved in [the experiment report](docs/heading_goal_20260910.md).

## Code map and checks

```text
oat/policy/flow_policy.py                  Shared flow/backbone infrastructure
oat/policy/flow_policy_shared_heading.py   Heading head, condition, losses and sampler
oat/policy/flow_policy_heading_zero.py     Exact-heading source variant
oat/policy/flow_policy_heading_gaussian.py  Full-rank Gaussian source variant
oat/model/diffusion/                       Original Transformer and U-Net components
oat/model/flow/                            U-Net adapter and StarVLA-DiT
oat/perception/                            RGB/state observation encoders
oat/dataset/heading_zarr_dataset.py        Training-only statistics and episode splits
oat/workspace/train_policy.py             Training, EMA, checkpoints and rollout scheduling
oat/env_runner/                           Simulator runners and checkpoint evaluation
scripts/                                  Training, conversion, evaluation and reporting
docs/                                     Current design, backbone guide and historical report
```

`flow_policy_shared_heading.py` is required by both heading policies. Its common
implementation does not add a separate supported training variant. Other baseline
code remains in the repository, but the six configs above define the current
heading-policy interface.

CPU checks for the heading methods and backbone integration:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  tests/test_heading_zero_policy.py tests/test_heading_gaussian_policy.py \
  tests/test_shared_heading_policy.py tests/test_heading_zero_full_config.py \
  tests/test_flow_backbones.py tests/test_unet_flow.py tests/test_starvla_dit.py \
  tests/test_libero_official_eval.py tests/test_libero_official_runner.py
```

The training infrastructure builds on OAT/Past2Next, robomimic and LIBERO; the
StarVLA-DiT adapter follows the StarVLA/GR00T-style action-head architecture.
Retired experiment source and documentation were archived under local
`output/heading_goal/cleanup_*` directories, which are not distributed with the
Git repository. Current heading configs do not require those archives.
