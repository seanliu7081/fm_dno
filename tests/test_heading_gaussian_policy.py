import contextlib
import io

import pytest
import torch

from oat.policy.flow_policy_heading_gaussian import HeadingGaussianFlowPolicy
from test_shared_heading_policy import SHAPE_META, TinyEncoder, batch, make_source_policy, single_cpu_thread


def make_policy(**overrides):
    settings = dict(shape_meta=SHAPE_META, obs_encoder=TinyEncoder(), horizon=4,
                    n_action_steps=2, n_obs_steps=2, embed_dim=8, n_layers=1,
                    n_heads=2, dropout=0, num_inference_steps=2,
                    heading_hidden_dim=8, heading_xy_rms=.8)
    settings.update(overrides)
    with contextlib.redirect_stdout(io.StringIO()):
        policy = HeadingGaussianFlowPolicy(**settings)
    policy.set_normalizer(make_source_policy(blocks=((0, 1),)).normalizer)
    return policy


def test_prior_raw_mean_follows_heading_with_anisotropic_normalization():
    policy = make_policy()
    angles = torch.tensor([0., .7, -1.6])
    prediction = dict(theta=angles, confidence=torch.ones(3))
    # Antithetic pairs give the exact mean, not a loose Monte Carlo check.
    z = torch.randn(3, 4, 7)
    pos = policy.transform_source(z, prediction)
    neg = policy.transform_source(-z, prediction)
    raw_mean = policy.normalizer["action"].unnormalize((pos + neg) / 2)[..., :2].mean(1)
    direction = raw_mean / raw_mean.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(direction, torch.stack((angles.cos(), angles.sin()), -1), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(pos[..., 2:], z[..., 2:], rtol=0, atol=0)


def test_prior_retains_full_rank_and_dno_noise_gradients_but_detaches_heading():
    policy = make_policy()
    theta = torch.tensor([.6], requires_grad=True)
    z = torch.randn(1, 4, 7, requires_grad=True)
    prediction = dict(theta=theta, confidence=torch.ones(1))
    jacobian = torch.autograd.functional.jacobian(lambda v: policy.transform_source(v, prediction), z).reshape(28, 28)
    assert torch.linalg.matrix_rank(jacobian) == 28
    source = policy.transform_source(z, prediction)
    source.square().sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert theta.grad is None


def test_invalid_heading_falls_back_without_corrupting_values():
    policy = make_policy()
    z = torch.randn(3, 4, 7)
    result, active = policy._transform_source(z, dict(theta=torch.tensor([float('nan'), .2, .3]),
                                                    confidence=torch.tensor([1., .1, 1.])))
    assert active.tolist() == [False, False, True]
    torch.testing.assert_close(result[:2], z[:2], rtol=0, atol=0)
    assert torch.isfinite(result).all()


def test_training_and_inference_use_same_observation_only_prior():
    policy = make_policy().eval()
    data = batch()
    noise = torch.randn_like(data['action'])
    losses = policy.loss_components(data, noise=noise, t=torch.ones(3) * .4)
    sampled = policy.predict_action(data['obs'], noise=noise, return_source=True)
    torch.testing.assert_close(losses['source'], sampled['source'], rtol=0, atol=0)
    changed_actions = dict(data, action=torch.randn_like(data['action']))
    changed = policy.loss_components(changed_actions, noise=noise, t=torch.ones(3) * .4)
    torch.testing.assert_close(changed['source'], losses['source'], rtol=0, atol=0)
    assert torch.isfinite(losses['loss'])
    losses['loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.obs_encoder.head.parameters())


@pytest.mark.parametrize('overrides', [dict(prior_parallel_std=0), dict(prior_perpendicular_std=-1),
                                      dict(prior_heading_mean=float('nan')), dict(prior_noise_scale=0),
                                      dict(prior_noise_scale=-1)])
def test_invalid_distribution_is_rejected(overrides):
    with pytest.raises(ValueError):
        make_policy(**overrides)
