# MimicGen six-task training

Both runs use the `fm_dno` Conda environment, a single policy conditioned on scalar
`task_uid`, two 128×128 RGB observations, EEF position/quaternion, gripper position,
and seven-dimensional normalized OSC_POSE **delta** actions. Model settings are
inherited from the existing heading-zero / heading-Gaussian StarVLA DiT configs.

| Task UID | Dataset | Train demos | Held-out rollout starts | Rollout horizon |
|---|---|---:|---:|---:|
| 0 | stack_three_d1 | 100 | 50 | 400 |
| 1 | square_d2 | 100 | 50 | 400 |
| 2 | threading_d1 | 100 | 50 | 400 |
| 3 | hammer_cleanup_d1 | 100 | 50 | 500 |
| 4 | mug_cleanup_d1 | 100 | 50 | 500 |
| 5 | coffee_d2 | 100 | 50 | 400 |

Threading D1 was confirmed during setup. The current official release also
supports Threading D2; the preparation script accepts an explicit `--tasks` list.

## Data and evaluation protocol

The official source is [`amandlek/mimicgen_datasets`](https://huggingface.co/datasets/amandlek/mimicgen_datasets).
Preparation records the resolved source revision and SHA256 digests.
For each task, numerically sorted demonstrations 0–99 train. The next 50 unique
initial states, excluding any training initial-state hashes, define evaluation.
The manifest records exact demo IDs and hashes, and a separate HDF5 stores their
simulator states and scene XML. Held-out action trajectories never enter the training Zarr; a separate Zarr stores them for action MSE.
These are held-out starts from the same task distribution, not new task variants.

`val_ratio=0` keeps all 600 selected demonstrations in training. Normalization and
heading XY statistics use those training frames only. A separate held-out dataset measures action-prediction MSE, and simulator
rollouts measure task success. Held-out data never fits normalization or heading statistics.

The existing LIBERO-style `data/*` and `meta/episode_ends` Zarr layout is retained.
Source actions are copied to float32 without differentiation or absolute-action
conversion. Upright 84×84 source images are resized to 128×128 with OpenCV linear
interpolation. Evaluation uses matching camera processing.

Artifacts, visible through the existing `data -> ../shared_data` link:

- `data/mimicgen/mimicgen6_N100.zarr` — 600 training demonstrations.
- `data/mimicgen/mimicgen6_N100.manifest.json` — task IDs, split and provenance.
- `data/mimicgen/eval_initial_states.hdf5` — 300 held-out starts.
- `data/mimicgen/mimicgen6_action_mse.zarr` and `.manifest.json` — the same 300 held-out demos with ground-truth action chunks and provenance.
- `data/mimicgen/hdf5/core/` — unchanged official source files.

## Training and GPU ownership

| Policy | Config | Physical GPUs |
|---|---|---|
| Heading-zero DiT | `train_mimicgen6_headingzero_dit` | 0, 1 |
| Heading-Gaussian DiT | `train_mimicgen6_headinggaussian_dit` | 2, 3 |

Each run uses two Accelerate/DDP training ranks, batch size 64 per rank (128 global),
600 epochs and `lazy_eval=false`. Both ranks run policy inference on their training
GPU during inline rollouts. Each handles 25 tests per task; simulator workers render
on that rank's physical GPU and close when evaluation ends. Four simulator workers
per rank are created during rollouts only. Rank results aggregate by episode count.
A one-rank launch fails instead of silently leaving the second GPU unused.

Evaluation follows completed epochs 50,100,…,600, with the same held-out starts
and per-episode noise seeds for both policies. Training pauses on both GPUs during
evaluation, then resumes on the same pair. GPU utilization can vary with simulator
CPU work, data loading and synchronization; both GPUs perform training and inference.

Latest checkpoints are saved after every completed epoch, immutable checkpoints accompany each
evaluation, and the top two checkpoints use aggregate task success. Fixed evaluation
starts are reused for checkpoint selection, so these results are validation scores;
an independent final test set would be needed for a selection-unbiased estimate.

## Commands

From this checkout:

```bash
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate fm_dno
./scripts/setup_mimicgen.sh
python scripts/prepare_mimicgen_dataset.py --download
```

The preparation command refuses to overwrite completed artifacts. Use a different
`--output-dir` to create another split. `setup_mimicgen.sh` pins simulator task code
and applies the [official installation guide's](https://mimicgen.github.io/docs/introduction/installation.html)
removal of task-zoo's deprecated `mujoco-py` requirement. It selects robosuite 1.4.1
(required by Coffee D2) and MuJoCo 3.2.6 (verified against the downloaded RGB images and source replays)
inside `fm_dno` only.

Managed launches use:

```bash
./scripts/run_mimicgen_training.sh zero 0,1
./scripts/run_mimicgen_training.sh gaussian 2,3
```

Run those through supervisor for long jobs. Supervisor service names are
`fm-mimicgen-headingzero` and `fm-mimicgen-headinggaussian`. Their output directories
are `output/mimicgen6/headingzero_dit_seed42` and
`output/mimicgen6/headinggaussian_dit_seed42`. Each contains resolved Hydra config,
`logs.json`, checkpoints, and `rollouts/epoch_XXX/rank_Y/attempt_ZZZ/` manifests,
episode results and completion/error summaries. Supervisor logs are under
`/var/log/portal/` with the service name.

## Action MSE and online logging

Both existing dashboards now receive direct online logs. The earlier offline sync
services are stopped and have autostart disabled. Stable run IDs are `1hg9rqh5`
(heading-zero) and `cc5kl9wl` (heading-Gaussian), in
`andyliu7081-northeastern-university/fm_dno_mimicgen`.

`val/action_mse` compares the EMA policy's full 16-step `action_pred` with recorded
seven-channel delta controls, after undoing the policy's action normalization.
Values are in source controller units, not flow-velocity or physical meter units.
The metric includes repeated actions at episode boundaries, matching the existing
LIBERO-style sequence-window convention.

Each task contributes its 50 held-out demos. The 300 demos yield 73,899 windows;
two GPUs process disjoint, unpadded index shards, so every window contributes once.
Both policies use fixed per-window random seeds. Training RNG state is preserved.
The checkpoint's training normalizer is reused, without fitting held-out data.

W&B metrics:

- `val/action_mse`: equal average of the six per-task MSE values.
- `val/<task>/action_mse`: the task's mean squared error across all window elements.
- `val/action_mse_micro`: mean squared error weighted by the number of window elements.
- `test_reconst_mse`: alias of `val/action_mse`.
- Per-task translation, rotation and gripper MSE, plus exact sample counts.

MSE is measured after the first completed epoch of a resumed process and after
completed epochs 50, 100, …, 600. Its W&B x-axis is `action_mse/completed_epochs`.
The first added measurements follow checkpoint resumption; earlier epochs cannot
be backfilled without their saved checkpoints. GPU assignments and the training
split are unchanged. W&B transport steps can differ from actual `global_step`
after resuming synced histories; charts use actual training steps or completed
epochs rather than the transport counter.

To rebuild the held-out MSE artifact from retained official sources:

```bash
conda activate fm_dno
python scripts/prepare_mimicgen_action_mse.py
```
