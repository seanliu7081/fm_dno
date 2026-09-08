# Flow action backbones

Four training configs add two action backbones to the existing plain flow and
learned-heading-conditioned flow policies:

| Policy | Time adaLN-Zero + observation cross-attention | StarVLA / GR00T-style head |
|---|---|---|
| Plain flow | `train_flowpolicy_mixed_dit` | `train_flowpolicy_starvla_dit` |
| Learned heading condition | `train_flowpolicy_heading_condition_mixed_dit` | `train_flowpolicy_heading_condition_starvla_dit` |

All four retain the existing image/state observation encoders, rectified-flow
velocity objective, 16-step action horizon, 8 executed actions, 2 observations,
and 10 Euler inference steps. Their defaults are 4 blocks, width 256, 4 attention
heads, and dropout 0.1. Different block structures have different parameter and
attention counts even at the same depth/width; these are architecture comparisons,
not parameter-matched comparisons. Training startup prints backbone parameter counts.

The heading configs inherit `train_flowpolicy_heading_condition`: a frozen,
pretrained reference supplies direction (2 dimensions) and heading-validity
confidence (1 dimension) to each observation token. They use `reference_use=condition`
and retain an IID Gaussian source. The same reference artifact can be used with
both backbones. See [reference pretraining](learned_canonicalization.md).

## Architectures

### Mixed adaLN-Zero

`oat/model/flow/mixed_dit.py` implements a bidirectional action Transformer.
Each block has self-attention, observation cross-attention, and a 4x-width GELU
feed-forward network. Time is encoded by a sinusoidal embedding and an MLP;
observation tokens are projected and given learned temporal positions.

Time produces separate scale, shift, and residual-gate vectors for each of the
three branches (nine vectors per block). Their final modulation projection is
zero-initialized, so each block initially acts as identity. The final
time-conditioned normalization and velocity projection follow the DiT-style
zero initialization; initial predicted velocity is zero. Output weights learn
first, followed by residual gates and the observation path. This is intentional,
so a first-step zero gradient in an earlier layer is not a frozen-encoder bug.

### StarVLA / GR00T-style action head

`oat/model/flow/starvla_dit.py` adapts StarVLA's GR00T action head into a standalone
PyTorch module. It retains time-conditioned action MLP embeddings, learned action
positions, 32 learned register/future tokens, ordinary time AdaLN before attention,
alternating cross-attention and self-attention blocks, ordinary LayerNorm before
feed-forward layers, and a time-modulated output followed by an action-decoder MLP.
There are no adaLN-Zero residual gates or zero-initialized output projections.

Four blocks mean two cross-attention blocks and two self-attention blocks.
At least two blocks are required so actions and register tokens can communicate.
The register count is configurable through `policy.backbone_kwargs.num_register_tokens`.

The adapter uses the existing fused image/state (and optional heading) features
as cross-attention memory. State is already included in these features, so it
has no separate state-token encoder. Learned observation-frame positions preserve
temporal order that StarVLA normally receives from its VLM. It does not load a
VLM or implement QwenPI's layerwise VLM connections. Dimensions are scaled to the
flow policy's configured width rather than the original 16-block/768-wide preset.
Time remains continuous in the existing 0..1000 embedding range, and training
retains fm_dno's uniform time sampling rather than StarVLA's Beta/bucket schedule.
No dependency on the StarVLA repository, diffusers, or pretrained DiT weights is
required by either new backbone.

## Training

Run from the `fm_dno` repository in its Python environment. For plain flow:

```bash
python scripts/run_workspace.py --config-name=train_flowpolicy_mixed_dit
python scripts/run_workspace.py --config-name=train_flowpolicy_starvla_dit
```

For learned-heading-conditioned flow, supply a pretrained reference artifact:

```bash
python scripts/run_workspace.py --config-name=train_flowpolicy_heading_condition_mixed_dit \
  reference_checkpoint=output/heading_reference/image_state/seed42/checkpoints/best.pt

python scripts/run_workspace.py --config-name=train_flowpolicy_heading_condition_starvla_dit \
  reference_checkpoint=output/heading_reference/image_state/seed42/checkpoints/best.pt
```

Use `task.policy.lazy_eval=false` to enable the existing rollout evaluation path.
The original training, dataset, device, logging and evaluation overrides still
apply. Config names create separate default run directories. Existing reference
artifacts can be reused, but the new action backbones must be trained or restored
from a checkpoint of their own architecture. Once saved in a complete policy
checkpoint, the frozen heading reference no longer needs the external artifact.

The heading variants also support `policy.reference_use=source` or `both`, and
`reference_mode=state` with a matching pretrained state-only artifact.

## Controlled comparisons and compatibility

`FlowPolicy` defaults to `backbone_type=transformer`. Old configs and model state
keys remain compatible. Its subclasses inherit the optional selector, so orbit,
canonical and reflow policies can also select a new backbone. For example, an
IID baseline using the same SO(2) action normalization as the heading variants is:

```bash
python scripts/run_workspace.py --config-name=train_flowpolicy_canon_iid \
  +policy.backbone_type=mixed_dit
```

The plain-flow configs retain their original per-dimension normalization; the
heading configs retain their original SO(2) normalization. Compare backbones
within the same policy/config family, or use the SO(2) IID baseline above when
isolating the effect of heading conditioning. Keep data splits, normalization,
reference artifact, source/coupling, training budget, evaluation starts and sampler
steps fixed. Report closed-loop success and latency alongside parameter counts;
these implementations do not establish a performance improvement.

Both new models keep gradients through actions and continuous time. The existing
heading/orbit differentiable sampler can therefore still backpropagate to its
initial noise for DNO.

## Validation

```bash
python -m pytest -q tests/test_mixed_dit.py tests/test_starvla_dit.py tests/test_flow_backbones.py
```

Tests cover initialization, observation/time dependence, temporal order,
bidirectional action interaction, optimizer updates, frozen heading references,
policy sampling, differentiable noise optimization, checkpoint restoration,
Hydra composition, and exact seeded legacy-backbone initialization compatibility.
These CPU checks require no datasets, simulator rollouts or pretrained reference.
