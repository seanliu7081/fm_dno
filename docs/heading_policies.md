# Heading Zero and Heading Gaussian flow policies

This is the current design reference. Both policies jointly predict an XY motion
heading from observations, feed it to the velocity model, and use it to construct
the flow source distribution. Their difference is the source transformation.
Each supports Transformer, U-Net and StarVLA-DiT, giving six standalone configs.
LIBERO-10 is the verification task; the method contains no task-specific geometry,
scripted motion phases, or per-task checkpoint selection.

## Shared structure

```text
Two observation frames: RGB views + robot state
                      │
              Shared observation encoder
                      │ features F: [B, 2, D]
             ┌────────┴───────────┐
             │                    │
     Flatten time + MLP           │
       heading d, validity p     │
             ├───────────────────► concatenate [F_t, d_x, d_y, p]
             │                                │
       detach d and p                         │
             │                                │
Gaussian noise → heading source x0 → velocity backbone v(x,t,condition)
                                              │
                                    Euler integration → actions
```

The implementation hierarchy is:

```text
FlowPolicy → SharedHeadingFlowPolicy → HeadingZeroFlowPolicy → HeadingGaussianFlowPolicy
```

[SharedHeadingFlowPolicy](../oat/policy/flow_policy_shared_heading.py) is required
common code, not a seventh training variant. It owns the shared head, conditioning,
supervision, source interface, flow loss and sampler. [Heading Zero](../oat/policy/flow_policy_heading_zero.py)
fixes condition mode, XY-only source rotation and zero angular jitter, and adds
training-dataset configuration and a separate heading optimizer group.
[Heading Gaussian](../oat/policy/flow_policy_heading_gaussian.py) overrides the
source transformation with a full-rank Gaussian.

## 1. Predicting the heading

The configured encoder fuses ResNet18/SpatialSoftmax RGB features and projected
robot-state features. The same encoder supplies both the heading head and the
velocity model; observations are encoded once per forward call.

For two observation frames, the head flattens `[B, 2, D]` to `[B, 2D]` and applies
`Linear(2D,128) → SiLU → Linear(128,3)`. The first two outputs are normalized to a
unit direction `d=(d_x,d_y)`; a near-zero vector falls back to `(1,0)`. The third
output is a validity logit `l`, with probability `p=sigmoid(l)`.
The predicted angle is `theta=atan2(d_y,d_x)`.

This heading describes the **net XY translation command over the future action
chunk**, not the robot's yaw or end-effector orientation. In LIBERO the action
has seven channels: translation delta XYZ, rotation delta XYZ, and gripper command.
Only translation channels 0 and 1 define heading.

Training labels come from raw, unnormalized demonstration actions:

```text
r = sum(action[k, :2], k=0,...,H_heading-1)
s = sqrt(mean(training_actions[:, :2] ** 2))
c = ||r|| / (s * sqrt(H_heading))
valid = c >= 0.05
d_target = r / ||r||                 # supervised only for valid samples
```

`H_heading=16` in the current configs. A stationary chunk or cancelling motion
can have an invalid net heading. The continuous target score `c` determines a
binary validity label; `p` predicts the probability of that label, not `c` itself.

The direction loss is mean `1-dot(d,d_target)` over valid samples. Validity uses
binary cross-entropy with logits over every sample. Both weights are 0.1.
[HeadingZarrDataset](../oat/dataset/heading_zarr_dataset.py) fits normalization and
`s` using selected training episodes only. The scale is saved in the checkpoint;
inference needs neither demonstration actions nor dataset statistics recomputation.

## 2. Heading as velocity-model conditioning

Every observation token becomes `[F_t, d_x, d_y, p]`, so the feature width grows
from `D` to `D+3`. The same predicted heading is repeated across the two frames.
The backbone consumes these features alongside the noisy action chunk and time.
Transformer and StarVLA use attention conditioning; U-Net uses flattened features
in its conditional residual blocks. See [the backbone guide](flow_backbones.md).

The newly added heading input columns start at zero, allowing the velocity model
to learn their influence. Heading predictions in the condition are **not detached**:
flow-loss gradients can reach the head and shared encoder through this route.
Auxiliary heading/validity losses also train them jointly. Low validity does not
remove the conditioning vector; it gates only the source transformation.
Ground-truth future headings never replace the predicted condition in normal training.

## 3. Heading Zero source

Start with independent Gaussian noise `z = sigma * epsilon`, in normalized action
coordinates, where `epsilon ~ N(0,I)` and `sigma=1` by default. Write action
normalization as `N(a)=S*a+b`, with per-coordinate scale and offset.

1. Convert the sampled chunk to raw coordinates: `u=N^-1(z)`.
2. Compute its XY resultant over `H_heading` steps and angle `theta_z`.
3. Rotate every XY vector in the chunk by `theta-theta_z`.
4. Normalize the rotated XY vectors back to obtain `x0`; retain all other noise channels.

The raw XY resultant now points exactly along the predicted heading. All timesteps
share the same rotation, preserving individual raw XY norms and relative angles.
This is a constrained source distribution, not an ordinary full-rank Gaussian.
Individual steps need not point along the heading: the constraint is on their sum.
“Zero” means **zero angular jitter**, not zero noise or zero robot motion.

If `p<0.5`, the prediction is nonfinite, or the sampled resultant is numerically
undefined, the source stays equal to the original `z`. An explicit jitter argument
exists for diagnostics, but the supported config uses no jitter in training or inference.

## 4. Heading Gaussian source

Gaussian uses the predicted direction to set an XY Gaussian mean and covariance,
without forcing the realized chunk resultant to have an exact angle. Let
`d_perp=(-d_y,d_x)` and `q=sqrt(mean(1/S_xy**2))`. For every timestep independently:

```text
u_xy = q * [(alpha*sigma + beta_parallel*z_x)*d
            + beta_perp*z_y*d_perp]
x0_xy = S_xy*u_xy + b_xy
x0_other = z_other
```

Here `z=sigma*epsilon` as above. Defaults are `alpha=0.5`,
`beta_parallel=1.0`, `beta_perp=0.5`. Thus in raw coordinates:

```text
mean = q * alpha * sigma * d
covariance = (q*sigma)^2 *
             [beta_parallel^2 * d*d^T + beta_perp^2 * d_perp*d_perp^T]
```

Both standard deviations are positive, so the XY source has full support. It
favors forward motion and greater variation along the heading, while still
allowing lateral and backward samples. The raw-coordinate construction matters:
anisotropic action normalization would otherwise distort physical directions.
The scale `q` comes from the action normalizer; it is distinct from the heading
label scale `s`. If `p<0.5` or the prediction is nonfinite, the unchanged Gaussian
`z` is used. Both policies detach heading and validity before source construction.

| Property | Heading Zero | Heading Gaussian |
|---|---|---|
| Head, labels, conditioning, flow loss | Shared | Shared |
| XY source | Rotate realized chunk | Heading-aligned Gaussian |
| Resultant heading | Exact when active | Stochastic, biased toward prediction |
| Full XY support when active | No | Yes |
| Other action channels | Original Gaussian noise | Original Gaussian noise |
| Low-validity fallback | Original Gaussian noise | Original Gaussian noise |

## 5. Flow matching and inference

Both policies train the same straight-path flow-matching objective:

```text
x1 = N(demonstration_action_chunk)
x0 = source_transform(sigma*epsilon, predicted_heading.detach())
t ~ Uniform(0,1)
xt = (1-t)*x0 + t*x1
L = mean((v(xt,t,condition) - (x1-x0))^2)
    + 0.1*L_direction + 0.1*L_validity
```

The source uses the model's prediction during training as well as inference.
Its detached construction prevents the head from reducing flow loss by moving
interpolation endpoints. The conditioning path still learns from flow loss.
Flow MSE magnitudes across different source laws are not directly comparable;
use generated-action metrics and closed-loop success to compare methods.

Inference constructs a source, integrates `dx/dt=v` with ten Euler steps from
`t=0` to `1`, then unnormalizes the output. The policy predicts 16 actions and
executes the first eight before observing again. There is no DNO optimization,
reflow stage, frozen external heading predictor, or task-specific planner in the
current six configs.

## Configurations and defaults

| Prior | Transformer | U-Net | StarVLA-DiT |
|---|---|---|---|
| Exact heading | [headingzero_transfomer](../oat/config/train_flowpolicy_headingzero_transfomer.yaml) | [headingzero_unet](../oat/config/train_flowpolicy_headingzero_unet.yaml) | [headingzero_starvlaDiT](../oat/config/train_flowpolicy_headingzero_starvlaDiT.yaml) |
| Gaussian heading | [headinggaussian_transfomer](../oat/config/train_flowpolicy_headinggaussian_transfomer.yaml) | [headinggaussian_unet](../oat/config/train_flowpolicy_headinggaussian_unet.yaml) | [headinggaussian_starvlaDiT](../oat/config/train_flowpolicy_headinggaussian_starvlaDiT.yaml) |

Each config writes policy and training settings in full, composing only the task
configuration. The existing filename spelling `transfomer` is intentional.

| Setting | Current standalone defaults |
|---|---|
| Seed / batch size | 42 / 64 |
| Configured epochs | 61 |
| Policy / encoder / heading learning rate | 1e-4 each |
| Scheduler | 500-update warmup, then constant |
| Weight decay / gradient clipping | 1e-6 / norm 1.0 |
| EMA | Enabled |
| RGB crop | 116 × 116, fixed evaluation crop |
| Observation / prediction / execution horizon | 2 / 16 / 8 |
| Euler sampling steps / noise scale | 10 / 1.0 |
| Dropout | 0.0 |
| Evaluation / logging | `lazy_eval=true` / offline |

Transformer and StarVLA use width 256, four blocks and four heads; StarVLA adds
32 register tokens. U-Net uses channel widths `[128,256,512]` and time embedding
width 128. These architectures have different parameter counts.

```bash
python scripts/run_workspace.py --config-name=train_flowpolicy_headingzero_starvlaDiT
python scripts/run_workspace.py --config-name=train_flowpolicy_headinggaussian_unet
```

The defaults are not the training history of every checkpoint. The separately
requested 150-epoch StarVLA run has its own saved configuration under
`output/heading_goal/starvla_dit_150_online_20260910/.hydra/config.yaml`, with online
logging and synchronous 500-episode evaluation every 15 completed epochs. That
run is stopped; writing this document does not restart it.

## Evaluation, compatibility and archives

Use one immutable checkpoint for an entire evaluation, report whether EMA is used,
and match initial states, settling actions, seeds, episode count and action horizon
across methods. These six configs do not by themselves select an external evaluation
protocol. The evaluator and its saved settings determine the actual protocol.
See [the historical verified-result report](heading_goal_20260910.md) for recorded
experiment evidence; its retired config names are historical identifiers.

The retained classes and module paths support existing heading checkpoints.
Older canonicalization, orbit, mixed-DiT, reflow and dependent DNO branches require
restoring their source from `output/heading_goal/cleanup_unused_methods_20260910/`.
Retired heading YAML aliases are in `output/heading_goal/cleanup_heading_variants_20260910/`.
Superseded mini-experiment notes, mixed-method project history and the duplicate
heading-zero guide are archived in `output/heading_goal/cleanup_heading_docs_20260910/`.
Training outputs, checkpoints and evaluation results are preserved.
