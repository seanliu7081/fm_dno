# Past2Next

Current heading / flow / DNO research: [长期项目上下文与实验记录](docs/project_context.md).

Two-stage discrete action modelling for robot manipulation, benchmarked on **LIBERO-10**.

1. **Stage 1 — action tokenizer.** An action-only autoencoder with a discrete bottleneck (FSQ)
   compresses a 16-step chunk of continuous 7-DoF actions into 8 ordered tokens and reconstructs
   it. It never sees observations.
2. **Stage 2 — policy.** A causal transformer conditioned on observations (and, for
   **Past2Next**, on past actions and their derivatives) autoregressively predicts the tokens of
   the *next* action chunk, decodes them with the **frozen** Stage-1 tokenizer, and executes them
   receding-horizon.

Continuous baselines that skip Stage 1 entirely (diffusion policy, flow matching, ACT) live in the
same harness, so every method shares the dataset, observation encoder, env runner and eval code.

Two conventions worth internalising up front:

- **Everything is Hydra.** One entry point, `scripts/run_workspace.py`, selects a config in
  `oat/config/`, and that config's `_target_` names the workspace class that runs training.
- **Always run from the repo root.** The entry scripts `os.chdir(ROOT_DIR)`, and all default paths
  (`data/libero/...`, `output/...`) are relative to it.

```
oat/
  common/       replay buffer, checkpointing, hydra resolvers, video/json logging
  config/       all Hydra experiment configs (train_*.yaml) + task/ groups
  dataset/      ZarrDataset, ZarrDatasetWithPastAction, ZarrDatasetWithPrevWindow
  env/          LIBERO env wrapper + hdf5 -> zarr conversion
  env_runner/   parallel sim rollout runner
  model/        act / autoregressive / diffusion backbones, EMA, schedulers
  perception/   observation encoders (robomimic ResNet, state, fused)
  policy/       the 8 policies below
  tokenizer/    the OAT tokenizer family (encoder / decoder / FSQ / SO(3) aug)
  workspace/    Hydra workspaces = training loops (entry points via `_target_`)
scripts/        data conversion, training entry point, sim eval
slurm/          Slurm launcher templates
third_party/    LIBERO (git submodule, branch `oat`)
data/           datasets (gitignored)
output/         Hydra run dirs: checkpoints, wandb, media (gitignored)
```

---

## 1. Environment setup

Requirements: Linux x86_64, Python 3.10, an NVIDIA GPU with a driver supporting CUDA ≥ 12.6 (the
pinned stack is torch 2.10 / cu128–cu129), and ~50 GB free disk for LIBERO data plus runs.

### 1.1 Clone with the LIBERO submodule

LIBERO is vendored as a submodule pinned to the `oat` branch of `github.com/Chaoqi-LIU/LIBERO`;
the code will not import without it.

```bash
git clone --recursive https://github.com/seanliu7081/past2next_clean.git past2next
cd past2next
# if you already cloned without --recursive:
git submodule update --init --recursive
```

### 1.2 Pick an install path

**Option A — `uv` (matches `uv.lock`, what the Slurm templates assume):**

```bash
uv sync                 # creates ./.venv with oat + third_party/LIBERO as workspace members
uv run python -c "import oat, libero"
```

All later commands can then be prefixed with `uv run` instead of activating a venv.

**Option B — `setup_env.sh` (fresh GPU box; installs apt deps too, needs root):**

```bash
chmod +x setup_env.sh
./setup_env.sh --cuda 12.9            # or omit --cuda to auto-detect from nvidia-smi
./setup_env.sh --venv-path /venv/oat  # optional custom location; default ./venv/oat
source ./venv/oat/bin/activate
```

This installs system libraries (`libgl1-mesa-glx`, `libegl1-mesa`, `libglfw3`, `libglew-dev`,
`libosmesa6-dev`, `patchelf`, …), the pinned pip set, then `pip install -e third_party/LIBERO`
and `pip install -e .`, and prints a verification block.

**Option C — conda:**

```bash
conda env create -f environment.yml   # fully pinned, includes the two editable installs
conda activate oat
```

**Option D — an environment you already prepared:**

```bash
pip install -e third_party/LIBERO
pip install -e .
```

Verify any of the above:

```bash
python -c "import torch, oat, libero; print(torch.__version__, torch.cuda.is_available())"
```

### 1.3 LIBERO paths

On first import LIBERO writes `~/.libero/config.yaml` pointing at `assets/`, `bddl_files/`,
`init_files/` and `datasets/` inside the submodule. Override the location with
`LIBERO_CONFIG_PATH`. Confirm it resolves:

```bash
python -c "from libero.libero import get_libero_path; print(get_libero_path('bddl_files'))"
```

If it prints `[Warning]: ... path does not exist`, the submodule was not initialised or the config
points at a stale checkout — delete `~/.libero/config.yaml` and re-import.

### 1.4 Headless rendering

Sim rollouts use MuJoCo offscreen rendering. On headless machines export EGL **before** launching
anything that instantiates an env runner:

```bash
export MUJOCO_GL=egl
```

Tokenizer training does not need it (no env). Policy training only needs it when
`task.policy.lazy_eval=false`.

### 1.5 Weights & Biases

Training logs through `accelerate`'s wandb tracker (project `oat_dev` by default). Either
`wandb login` once, or disable networking per run:

```bash
export WANDB_MODE=offline            # or add the override: logging.mode=offline
```

---

## 2. Data preparation

The pipeline is **download hdf5 → per-task zarr → merged multi-task zarr**.

```bash
# 1. download the raw LIBERO hdf5 demos (libero_100 contains the LIBERO-10 / long suite)
python third_party/LIBERO/benchmark_scripts/download_libero_datasets.py \
    --datasets libero_100 --use-huggingface

# 2. stage the *_demo.hdf5 files where the converter looks for them
mkdir -p data/libero/hdf5_datasets
cp third_party/LIBERO/libero/datasets/libero_10/*.hdf5 data/libero/hdf5_datasets/
#    (symlinks work too, and save the raw dataset size)

# 3. convert every hdf5 into a per-task zarr replay buffer
python scripts/convert_libero_dataset.py            # all demos per task
python scripts/convert_libero_dataset.py -n 50      # or subsample N demos per task

# 4. merge the 10 task zarrs into the multi-task dataset the configs expect
python scripts/compose_libero_multitask_dataset.py -mt libero10
```

Step 3 writes `data/libero/<TASK_NAME>_N<n_demo>.zarr` (it prompts before overwriting an existing
one). Step 4 shells out to `scripts/merge_data.py` and produces `data/libero/libero10_N500.zarr`
(10 tasks × 50 demos). `-mt libero90` is also available.

Each episode stores `action`, `agentview_rgb`, `robot0_eye_in_hand_rgb`, `robot0_joint_pos`,
`robot0_eef_pos`, `robot0_eef_quat` (converted from axis-angle to quaternion),
`robot0_gripper_qpos`, `prompt` and `task_uid`. Expect ~3.4 GB for `libero10_N500.zarr`.

> **`training.num_demo` is a filename token, not a subsampling knob.** Configs resolve
> `zarr_path: data/libero/libero10_N${training.num_demo}.zarr`, so `training.num_demo=500` simply
> means "load `libero10_N500.zarr`". To train on fewer demos, build a smaller zarr (step 3 `-n`)
> and pass the matching `training.num_demo`. Independently, `ZarrDataset` accepts
> `max_train_episodes` to cap training episodes at load time — it is not in the configs, so add it
> with `+task.policy.dataset.max_train_episodes=100`.

---

## 3. Training

### 3.1 Entry point

```bash
python scripts/run_workspace.py --config-name=<config> [hydra overrides...]
```

`<config>` is any file in `oat/config/` without the `.yaml` suffix. Multi-GPU uses `accelerate` —
all workspaces build their own `Accelerator`, so a plain `accelerate launch` in front is the only
change. Every command below shows the 4-GPU form; drop to a single GPU with plain `python`.

Each run creates `output/<YYYYMMDD>/<HHMMSS>_<name>_<task_name>_N<num_demo>/` containing:

```
.hydra/config.yaml             resolved config (eval scripts read this back)
checkpoints/latest.ckpt        rolling checkpoint, every training.checkpoint_every epochs
checkpoints/ep-XXXX_*.ckpt     top-k checkpoints, selected by checkpoint.topk.monitor_key
logs.json                      per-epoch metric log (one JSON object per line)
wandb/, media/                 wandb run dir and rollout videos
```

Checkpoints are self-contained: they embed the full `cfg` plus all state dicts, so
`BasePolicy.from_checkpoint(path)` / `OATTok.from_checkpoint(path)` rebuild the workspace, the
model and the EMA weights without any config on the side.

**Resume** is automatic: `training.resume=true` (default) picks up
`<output_dir>/checkpoints/latest.ckpt` if it exists. Because the output dir is timestamped,
resuming means re-launching with `hydra.run.dir=<the original run dir>` — otherwise you get a
fresh run.

### 3.2 Stage 1 — action tokenizers

Top-k checkpoints are selected on `test_reconst_mse` (lower is better), producing files like
`ep-0900_mse-0.003.ckpt`. Both tokenizers use `horizon=16` and emit **8 tokens per chunk**.

**`train_oattok`** — baseline: RegisterEncoder → FSQ `[8,5,5,5,5]` (5,000 codes) →
SinglePassDecoder.

```bash
HYDRA_FULL_ERROR=1 accelerate launch \
    --num_machines 1 --multi_gpu --num_processes 4 \
    scripts/run_workspace.py \
    --config-name=train_oattok \
    training.num_epochs=5001 \
    training.num_demo=500
```

**`train_oattok_so3aug`** — the Past2Next tokenizer: same backbone, FSQ `[8,5,5,5,5,5]`
(25,000 codes), plus `SO3ActionChunkAug` (rotation-only, `p=0.6`, `max_angle_deg=30`) applied to
the action chunk during training.

```bash
HYDRA_FULL_ERROR=1 accelerate launch \
    --num_machines 1 --multi_gpu --num_processes 4 \
    scripts/run_workspace.py \
    --config-name=train_oattok_so3aug \
    training.num_epochs=5001 \
    training.num_demo=500
```

### 3.3 Stage 2 — token-based policies

These declare `policy.action_tokenizer.checkpoint: ???`, i.e. Hydra *requires* you to supply the
Stage-1 checkpoint path; the run aborts with a missing-mandatory-value error otherwise.

> **The policy's `horizon` must equal the tokenizer's `horizon`.** Both tokenizers and all
> policies here default to 16, so the pairings below work as-is.

**`train_oatpolicy`** — baseline OAT policy: obs-only conditioning (2 frames), AR head over the
8 tokens, executes 8 of 16 steps.

```bash
HYDRA_FULL_ERROR=1 MUJOCO_GL=egl accelerate launch \
    --num_machines 1 --multi_gpu --num_processes 4 \
    scripts/run_workspace.py \
    --config-name=train_oatpolicy \
    training.num_epochs=5001 \
    training.num_demo=500 \
    task.policy.lazy_eval=false \
    policy.action_tokenizer.checkpoint=/abs/path/to/output/<tok-run>/checkpoints/ep-0900_mse-0.003.ckpt
```

**`train_past2next`** — the main method (see §5). Adds 7 raw past actions + explicit acc/jerk to
the conditioning; uses the `libero10_with_past` task group.

```bash
HYDRA_FULL_ERROR=1 MUJOCO_GL=egl accelerate launch \
    --num_machines 1 --multi_gpu --num_processes 4 \
    scripts/run_workspace.py \
    --config-name=train_past2next \
    training.num_epochs=5001 \
    training.num_demo=500 \
    task.policy.lazy_eval=false \
    policy.action_tokenizer.checkpoint=/abs/path/to/output/<tok-run>/checkpoints/ep-0900_mse-0.003.ckpt
```

Pair it with a `train_oattok_so3aug` checkpoint for the full method, or with `train_oattok` to
isolate the enriched-past contribution.

**`train_past2next_self_past`** — same architecture and loss as `train_past2next`; the only change
is *where the 7 past actions come from during training*. Instead of the dataset's ground truth, the
policy re-runs itself on the observation window one execution stride back and keeps the past buffer
that run would have left behind — the exact quantity it conditions on at rollout. This closes the
train/rollout mismatch at depth 1 (the inner call still uses ground-truth past). Requires the
`libero10_with_prev_window` task group, which adds `prev_obs` / `prev_past_action` to each sample.

```bash
HYDRA_FULL_ERROR=1 MUJOCO_GL=egl accelerate launch \
    --num_machines 1 --multi_gpu --num_processes 4 \
    scripts/run_workspace.py \
    --config-name=train_past2next_self_past \
    training.num_epochs=5001 \
    training.num_demo=500 \
    task.policy.lazy_eval=false \
    policy.action_tokenizer.checkpoint=/abs/path/to/output/<tok-run>/checkpoints/ep-0900_mse-0.003.ckpt
```

The extra generation costs ~1.47x per step (batch 64, bf16, one RTX 4090). Knobs:
`policy.self_past_p` (1.0 = always self-generated, lower mixes with ground truth,
scheduled-sampling style), `policy.self_past_warmup_steps` (default 500 — ground-truth past first,
so the policy is not conditioned on a randomly-initialised model's output), and
`policy.self_past_temperature` / `policy.self_past_topk` (`null` reuses the policy's own sampling
settings, which is what rollout does).

### 3.4 Continuous baselines (no tokenizer)

Each is one command; swap `--config-name`. None takes a tokenizer checkpoint.

**`train_diffpolicy`** — DDIM diffusion policy over the 16-step chunk, 10 inference steps.

```bash
HYDRA_FULL_ERROR=1 MUJOCO_GL=egl accelerate launch \
    --num_machines 1 --multi_gpu --num_processes 4 \
    scripts/run_workspace.py --config-name=train_diffpolicy \
    training.num_epochs=5001 training.num_demo=500 task.policy.lazy_eval=false
```

**`train_flowpolicy`** — rectified flow matching, 10 inference steps.

```bash
HYDRA_FULL_ERROR=1 MUJOCO_GL=egl accelerate launch \
    --num_machines 1 --multi_gpu --num_processes 4 \
    scripts/run_workspace.py --config-name=train_flowpolicy \
    training.num_epochs=5001 training.num_demo=500 task.policy.lazy_eval=false
```

The optional **mixed adaLN-Zero** and **StarVLA / GR00T-style** action backbones
each have plain-flow and learned-heading-conditioned configs. See
[flow backbone variants](docs/flow_backbones.md) for the four configs, architecture
differences, reference-checkpoint setup, and training commands.

**`train_flowpolicy_with_enriched_past`** — the same flow policy with Past2Next's enriched-past
conditioning, isolating that idea from the discrete bottleneck. Uses `libero10_with_past`.

```bash
HYDRA_FULL_ERROR=1 MUJOCO_GL=egl accelerate launch \
    --num_machines 1 --multi_gpu --num_processes 4 \
    scripts/run_workspace.py --config-name=train_flowpolicy_with_enriched_past \
    training.num_epochs=5001 training.num_demo=500 task.policy.lazy_eval=false
```

**`train_actpolicy`** — ACT / DETR-VAE, single observation frame (`n_obs_steps=1`), ResNet-18
backbone, `kl_weight=10.0`.

```bash
HYDRA_FULL_ERROR=1 MUJOCO_GL=egl accelerate launch \
    --num_machines 1 --multi_gpu --num_processes 4 \
    scripts/run_workspace.py --config-name=train_actpolicy \
    training.num_epochs=5001 training.num_demo=500 task.policy.lazy_eval=false
```

### 3.5 Config summary

| Config | Class | `horizon` / `n_action_steps` | `n_obs_steps` | Task group | Tokenizer ckpt |
|---|---|---|---|---|---|
| `train_oattok` | `OATTok` | 16 / — | — | `tokenizer/libero/libero10` | — |
| `train_oattok_so3aug` | `OATTokSO3Aug` | 16 / — | — | `tokenizer/libero/libero10` | — |
| `train_oatpolicy` | `OATPolicy` | 16 / 8 | 2 | `policy/libero/libero10` | **yes** |
| `train_past2next` | `Past2NextPolicy` | 16 / 8, `past_n=7` | 2 | `policy/libero/libero10_with_past` | **yes** |
| `train_past2next_self_past` | `Past2NextSelfPastPolicy` | 16 / 8, `past_n=7` | 2 | `policy/libero/libero10_with_prev_window` | **yes** |
| `train_diffpolicy` | `DiffusionTransformerPolicy` | 16 / 8 | 2 | `policy/libero/libero10` | no |
| `train_flowpolicy` | `FlowPolicy` | 16 / 8 | 2 | `policy/libero/libero10` | no |
| `train_flowpolicy_with_enriched_past` | `FlowPolicyWithEnrichedPast` | 16 / 8, `past_n=7` | 2 | `policy/libero/libero10_with_past` | no |
| `train_actpolicy` | `ACTPolicy` | 16 / 8 | 1 | `policy/libero/libero10` | no |

### 3.6 In-training sim evaluation

`task.policy.lazy_eval` controls rollouts during training:

- `lazy_eval=true` (config default) — no env is constructed; training is pure supervised learning.
  Note the consequence: the top-k monitor key is `mean_success_rate`, which then never appears, so
  **only `latest.ckpt` is written**. Evaluate offline instead (§4).
- `lazy_eval=false` — rank 0 rolls out every `training.rollout_every` epochs with `n_test=500`
  episodes spread over the 10 LIBERO-10 tasks, `n_test_vis=20` videos, `n_parallel_envs=20`,
  `max_episode_steps=550`. This populates the `ep-XXXX_sr-0.XXX.ckpt` top-k files.

Useful runner overrides: `task.policy.env_runner.n_test=100`,
`task.policy.env_runner.n_parallel_envs=16`, `task.policy.env_runner.n_test_vis=4`.

---

## 4. Offline evaluation

```bash
MUJOCO_GL=egl python scripts/eval_policy_sim.py \
    -c output/<run>/checkpoints/ep-0800_sr-0.656.ckpt \
    -o output/eval_metrics/<name> \
    -n 3 -d cuda:0
```

`-c` takes a `.ckpt` file or a directory (every `.ckpt` in it except `latest.ckpt`). `-n/--num_exp`
repeats the eval and reports mean/std/stderr. Results go to `eval_log.json` plus rollout videos
in `-o`; the script prompts before overwriting an existing output dir.

The inference knobs `--temperature`, `--topk` and `--use_k_tokens` (decode with only the first *k*
of the ordered tokens — the Matryoshka property) apply to `train_oatpolicy` and `train_past2next`
only; the continuous baselines do not accept them.

---

## 5. Past2Next in brief

`Past2NextPolicy` (`oat/policy/past2next.py`, config `oat/config/train_past2next.yaml`) is an
autoregressive policy over the frozen tokenizer's codebook. Its one idea: **the recent past of the
action stream carries information the observations do not**, so condition on it explicitly.

**Conditioning.** The AR transformer attends to a condition sequence of length
`n_obs_steps + 2 + past_n` = `2 + 2 + 7` = **11**, built by `_build_condition()` as
`[obs, explicit, raw]`:

| Block | Length | Contents |
|---|---|---|
| `obs` | 2 | Fused observation features (2 frames: two RGB views + eef pos/quat, gripper qpos, task uid) |
| `explicit` | 2 | `acc = a₋₁ − a₋₂` and `jerk = a₋₁ − 2a₋₂ + a₋₃`, each through its own `Linear→GELU→Linear` |
| `raw` | 7 | The 7 raw past actions `[a₋₇ … a₋₁]`, through one **shared** `Linear→GELU→Linear` |

Past actions are normalized with the dataset's action normalizer before any differencing, so acc
and jerk are computed in normalized units. The `acc`/`jerk` projections are kept separate from the
`raw` projection because the three live on very different scales.

**Why the explicit derivatives.** Observations give position (`eef_pos`) and, across two frames, a
coarse velocity. They cannot express acceleration or jerk, and the model would otherwise have to
learn to recover them by differencing across condition tokens. Feeding them directly is cheap
(~2 small MLPs) and removes that burden. The raw 7-step history is kept alongside them because it
preserves temporal structure the two derivative scalars discard — command inertia, task phase.

**Prediction.** Given that condition, the model autoregressively emits the 8 tokens of the *next*
chunk (vocab = codebook + 1 `<BOS>`), the frozen tokenizer decodes them into 16 actions, and the
first `n_action_steps=8` are executed. At inference a rolling `_past_buffer` supplies the past
window; `reset()` clears it at episode start, and it is zero-initialised on the first step.

**Relationship to the other configs.** `train_oatpolicy` is the same architecture with the
`explicit` and `raw` blocks removed (condition length 2) — the ablation that isolates the enriched
past. `train_flowpolicy_with_enriched_past` keeps the enriched past but replaces the discrete
bottleneck with flow matching — the ablation that isolates the tokenizer. Pairing
`train_past2next` with a `train_oattok_so3aug` checkpoint gives the full method; pairing it with
`train_oattok` isolates the SO(3) augmentation.

**Exposure bias.** Training reads `past_action` from the dataset, but at rollout the past window is
the policy's own output, so errors there are off-distribution. `train_past2next_self_past`
(`oat/policy/past2next_self_past.py`) is the variant that trains on the self-generated past
instead; see §3.3.

---

## License

See [LICENSE](LICENSE).
