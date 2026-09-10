# Flow action backbones

The supported backbones are Transformer, U-Net and StarVLA-DiT. Each is available
as an observation-only baseline and for heading zero and heading Gaussian.
See [Heading flow policies](heading_policies.md) for the six heading configs.

## Observation-only baselines

All three baselines use [`FlowPolicy`](../oat/policy/flow_policy.py). The velocity
model receives encoded observations, the noisy action chunk and flow time. There
is no heading head, heading condition, auxiliary heading loss or heading-dependent
source transformation. Training uses straight-path flow matching from independent
Gaussian noise; inference uses ten Euler steps.

| Backbone | Standalone config | `policy.backbone_type` |
|---|---|---|
| Transformer | [train_flowpolicy_transformer](../oat/config/train_flowpolicy_transformer.yaml) | `transformer` |
| U-Net | [train_flowpolicy_unet](../oat/config/train_flowpolicy_unet.yaml) | `unet` |
| StarVLA-DiT | [train_flowpolicy_starvlaDiT](../oat/config/train_flowpolicy_starvlaDiT.yaml) | `starvla_dit` |

These configs use the same standalone format, observation encoder, backbone sizes,
training schedule and checkpoint settings as the corresponding heading-zero
configs. [`TrainSplitZarrDataset`](../oat/dataset/train_split_zarr_dataset.py)
preserves their episode split, training-only action/state normalization and fixed
RGB range without computing heading statistics. The dataset seed stays at 42
when the model seed changes. Each config specifies policy and encoder learning
rates of `1e-4`.

```bash
python scripts/run_workspace.py --config-name=train_flowpolicy_transformer
python scripts/run_workspace.py --config-name=train_flowpolicy_unet
python scripts/run_workspace.py --config-name=train_flowpolicy_starvlaDiT
```

The original `train_flowpolicy.yaml` retains its legacy training settings and
dataset behavior; use the new standalone configs for matched comparisons.

## Transformer

`oat/model/diffusion/transformer_for_diffusion.py` is the original observation-conditioned
Transformer velocity model. Its default constructor and initialization are preserved.

## U-Net

`oat/model/flow/unet.py` adapts the temporal conditional U-Net to flow velocity
prediction. Time and flattened observation/heading features condition its residual
blocks. Padding supports action horizons that do not divide the downsampling factor.

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
required by this backbone.

## Training and comparison

```bash
python scripts/run_workspace.py --config-name=train_flowpolicy_headingzero_starvlaDiT
python scripts/run_workspace.py --config-name=train_flowpolicy_headinggaussian_starvlaDiT
```

The six heading configs share the observation encoder and heading head. Compare
backbones within a heading family while holding data, normalization, training
budget, evaluator and sampling settings fixed. Backbone parameter counts differ.

## Retired methods

An earlier plain StarVLA-DiT config was retired; the standalone baseline above
provides the current matched training recipe. Mixed DiT, canonicalization, orbit,
blockwise coupling, frozen heading references, reflow and
their dependent DNO utilities were removed from the active source tree. Historical
results and checkpoints remain in `output/`. Their source is recoverable from
`output/heading_goal/cleanup_unused_methods_20260910/retired_source.tar.gz`.
Restoring those retired methods is necessary before loading their old checkpoints.

## Validation

```bash
python -m pytest -q tests/test_starvla_dit.py tests/test_unet_flow.py tests/test_flow_backbones.py tests/test_heading_zero_full_config.py
```

CPU tests cover backbone behavior, training gradients, optimizer membership,
checkpoint restoration, the six config compositions and seeded compatibility of
the original Transformer initialization.
