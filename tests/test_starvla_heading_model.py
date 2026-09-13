"""Mathematical and native-backend checks for the StarVLA Gaussian expert."""
import numpy as np
import pytest
import torch

from oat.starvla_heading.prior import ActionStatistics, gaussian_source, heading_prediction, heading_supervision


def statistics():
    return ActionStatistics({
        "action": {"scale": [2., 5., 1., 1., 1., 1., 1.], "offset": [.3, -.4, 0., 0., 0., 0., 0.]},
        "state": {"scale": [1.] * 9, "offset": [0.] * 9}, "heading_xy_rms": .2})


def test_gaussian_mean_covariance_raw_coordinates_and_unchanged_other_channels():
    torch.manual_seed(42)
    stats = statistics()
    noise = torch.randn(80000, 1, 7)
    direction = torch.tensor([[.6, .8]]).expand(len(noise), -1)
    pred = {"direction": direction, "confidence": torch.ones(len(noise))}
    x, active = gaussian_source(noise, pred, stats)
    raw = stats.unnormalize(x)[:, 0, :2]
    q = stats.action_scale[:2].reciprocal().square().mean().sqrt()
    d = direction[0]
    p = torch.tensor([-.8, .6])
    expected_cov = q.square() * (torch.outer(d, d) + .25 * torch.outer(p, p))
    assert torch.allclose(raw.mean(0), q * .5 * d, atol=.004)
    assert torch.allclose(torch.cov(raw.T), expected_cov, atol=.004)
    assert torch.linalg.eigvalsh(torch.cov(raw.T)).min() > 0
    assert active.all()
    assert torch.equal(x[..., 2:], noise[..., 2:])


def test_source_detaches_heading_and_nonfinite_or_low_confidence_falls_back():
    stats = statistics()
    noise = torch.randn(3, 4, 7)
    direction = torch.tensor([[1., 0.], [1., 0.], [float("nan"), 0.]], requires_grad=True)
    pred = {"direction": direction, "confidence": torch.tensor([.8, .1, .9], requires_grad=True)}
    x, active = gaussian_source(noise, pred, stats)
    assert active.tolist() == [True, False, False]
    assert not x.requires_grad
    assert torch.equal(x[1:], noise[1:])
    assert torch.isfinite(x).all()


def test_statistics_keep_original_precision_after_bfloat16_model_conversion():
    stats = statistics()
    before = stats.action_offset.clone()
    stats.bfloat16()
    assert stats.action_offset.dtype == torch.float32
    assert torch.equal(stats.action_offset, before)


def test_heading_labels_exclude_padded_future_actions():
    stats = statistics()
    pred = heading_prediction(torch.tensor([[1., 0., 2.]]))
    actions = torch.tensor([[[.1, 0., 0., 0., 0., 0., 0.], [0., 100., 0., 0., 0., 0., 0.]]])
    mask = torch.tensor([[True, False]])
    direction_loss, validity_loss, angle, valid = heading_supervision(actions, mask, pred, stats)
    assert direction_loss == 0 and angle == 0 and valid == 1
    assert torch.isfinite(validity_loss)
    empty = heading_supervision(actions, torch.zeros_like(mask), pred, stats)
    assert empty[0] == 0 and empty[1] == 0


@pytest.fixture(scope="module")
def native_head():
    torch.set_num_threads(2)
    from oat.starvla_heading.model import HeadingGaussianActionHead
    stats = {"action": {"scale": [1.] * 7, "offset": [0.] * 7},
             "state": {"scale": [1.] * 9, "offset": [0.] * 9}, "heading_xy_rms": .2}
    return HeadingGaussianActionHead({"action_horizon": 4, "num_inference_timesteps": 2,
                                     "diffusion_model_cfg": {"num_layers": 2}}, stats, 16)


def test_native_dit_masks_padding_and_backpropagates_to_vlm(native_head):
    head = native_head.eval()
    tokens = torch.randn(2, 6, 16, requires_grad=True)
    changed = tokens.detach().clone()
    changed[1, :2] = 1000
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]])
    actions, state = torch.randn(2, 4, 7), torch.randn(2, 2, 9)
    noise, t = torch.randn_like(actions), torch.tensor([.2, .8])
    result = head(tokens, actions, state, mask, noise=noise, t=t, return_metrics=True)
    other = head(changed, actions, state, mask, noise=noise, t=t, return_metrics=True)
    torch.testing.assert_close(result["action_loss"], other["action_loss"])
    result["action_loss"].backward()
    assert tokens.grad.abs().sum() > 0
    assert tokens.grad[1, :2].abs().sum() == 0
    assert head.heading_head[-1].weight.grad.abs().sum() > 0


def test_native_sampling_uses_same_source_and_restores_checkpoint(native_head, tmp_path):
    head = native_head.eval()
    tokens, state = torch.randn(1, 5, 16), torch.randn(1, 2, 9)
    mask, noise = torch.ones(1, 5, dtype=torch.bool), torch.randn(1, 4, 7)
    with torch.no_grad():
        expected = head(tokens, state=state, encoder_attention_mask=mask, noise=noise, inference=True)
        direct = head.predict_action(tokens, state, mask, noise=noise)
    torch.testing.assert_close(expected, direct)
    path = tmp_path / "head.pt"
    torch.save(head.state_dict(), path)
    before = head.state_positions.detach().clone()
    with torch.no_grad():
        head.state_positions.add_(4)
    head.load_state_dict(torch.load(path, weights_only=True))
    assert torch.equal(head.state_positions, before)
    with torch.no_grad():
        actual = head(tokens, state=state, encoder_attention_mask=mask, noise=noise, inference=True)
    torch.testing.assert_close(expected, actual)


def test_padded_action_targets_cannot_change_valid_action_loss(native_head):
    head = native_head.eval()
    tokens, state = torch.randn(1, 5, 16), torch.randn(1, 2, 9)
    actions = torch.randn(1, 4, 7)
    changed = actions.clone()
    changed[:, 2:] = 1000
    valid = torch.tensor([[True, True, False, False]])
    noise, t = torch.randn_like(actions), torch.tensor([.6])
    first = head(tokens, actions, state, action_mask=valid, noise=noise, t=t)
    second = head(tokens, changed, state, action_mask=valid, noise=noise, t=t)
    torch.testing.assert_close(first, second)


def test_auxiliary_losses_and_gradients_are_invariant_to_microbatch_partition():
    stats = statistics()
    actions = torch.zeros(5, 4, 7)
    actions[0, :, 0] = .1
    actions[2, :, 1] = .1
    actions[3, :, 0] = -.1
    mask = torch.ones(5, 4, dtype=torch.bool)
    mask[4] = False
    output = torch.tensor([[.4, .8, .5], [.5, .3, -.3], [.9, .2, 1.],
                           [.2, -.7, -.2], [.4, .5, .1]], requires_grad=True)
    full = heading_supervision(actions, mask, heading_prediction(output), stats)
    full_loss = full[0] + full[1]
    full_gradient, = torch.autograd.grad(full_loss, output)
    for widths in ([1, 1, 1, 1, 1], [2, 3], [4, 1]):
        start = 0
        partition_loss = output.sum() * 0
        for width in widths:
            sl = slice(start, start + width)
            losses = heading_supervision(actions[sl], mask[sl], heading_prediction(output[sl]), stats)
            partition_loss = partition_loss + (losses[0] + losses[1]) * width / len(actions)
            start += width
        gradient, = torch.autograd.grad(partition_loss, output)
        torch.testing.assert_close(partition_loss, full_loss)
        torch.testing.assert_close(gradient, full_gradient)
