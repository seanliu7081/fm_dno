import copy

import pytest
import torch
from torch import nn

from oat.dno.noise_initializer import NoiseInitializer
from oat.policy.base_policy import BasePolicy
from oat.policy.task_noise_policy import TaskNoisePolicy


class IdentityNormalizer(nn.Module):
    def unnormalize(self, value):
        return value


class TinyFlow(BasePolicy):
    def __init__(self, reference_use='condition'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.normalizer = nn.ModuleDict({'action': IdentityNormalizer()})
        self.reference_use = reference_use
        self.horizon = 4
        self.action_dim = 7
        self.n_obs_steps = 2
        self.n_action_steps = 2
        self.num_inference_steps = 2
        self.prior_noise_scale = 1.0
        self.source_calls = 0

    def _prepare_observation(self, obs):
        cond = obs['state']
        return cond, cond[:, -1, 0], torch.ones(cond.shape[0], dtype=torch.bool)

    def _sample_prior_with_heading(self, theta, valid, batch_size, device, dtype):
        self.source_calls += 1
        z = torch.randn(batch_size, self.horizon, self.action_dim, device=device, dtype=dtype)
        if self.reference_use == 'both':
            # Deliberately anisotropic source, including fresh angular randomness.
            angle = theta + torch.distributions.VonMises(torch.zeros_like(theta), 4.0).sample()
            norm = z[..., :2].norm(dim=-1)
            z[..., 0] = norm * angle.cos()[:, None]
            z[..., 1] = norm * angle.sin()[:, None]
        return z

    def sample_prior(self, *args, **kwargs):
        raise AssertionError('Observation-free source must never be used')

    def sample_chunk(self, cond, z):
        return z * self.weight

    def get_policy_name(self):
        return 'tiny'

    def get_observation_ports(self):
        return ['state']

    def get_observation_modalities(self):
        return ['state']

    def get_observation_encoder(self):
        return nn.Identity()


def obs(batch=2):
    return {'state': torch.zeros(batch, 2, 3)}


def context(batch=2):
    return {'eef_pos': torch.zeros(batch, 3), 'goal_pos': torch.ones(batch, 3) * 0.1,
            'desired_gripper': torch.ones(batch), 'phase': torch.zeros(batch, dtype=torch.long)}


def cost(actions, context):
    return actions.square().mean(dim=(-1, -2))


@pytest.mark.parametrize('source', ['condition', 'both'])
def test_baseline_and_best_k_share_candidate_zero_and_do_not_change_rng(source):
    base = TinyFlow(source)
    baseline = TaskNoisePolicy(base, lambda actions, ctx: actions.new_zeros(actions.shape[0]), mode='baseline')
    best = TaskNoisePolicy(base, lambda actions, ctx: actions.new_zeros(actions.shape[0]),
                           mode='best_of_k', num_candidates=5)
    rng = torch.get_rng_state().clone()
    a = baseline.predict_action(obs())
    b = best.predict_action(obs())
    torch.testing.assert_close(a['z'], b['z'], rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    baseline.reset()
    torch.testing.assert_close(baseline.predict_action(obs())['z'], a['z'], rtol=0, atol=0)


@pytest.mark.parametrize('source', ['condition', 'both'])
def test_dno_improves_proxy_once_source_sampled_and_freezes_weights(source):
    base = TinyFlow(source)
    before = copy.deepcopy(base.state_dict())
    policy = TaskNoisePolicy(base, cost, mode='dno', num_candidates=3, num_grad_steps=4,
                             lr=0.1, trust_weight=0, trust_radius=0.2,
                             record_training_data=True)
    with torch.no_grad():  # A caller's no_grad is fine; inference_mode is not.
        result = policy.predict_action(obs(), context())
    assert base.source_calls == 3
    assert result['info']['final_objective'] < result['info']['initial_objective']
    assert result['teacher_improved'].all()
    assert result['action'].shape == (2, 2, 7)
    assert result['context_features'].shape == (2, 9)
    assert result['info']['nfe'] == 3 * (4 + 2) * base.num_inference_steps
    rms = (result['optimized_noise'] - result['base_noise']).square().mean((-1, -2)).sqrt()
    assert (rms <= 0.200001).all()
    for key, value in base.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(p.grad is None and not p.requires_grad for p in base.parameters())


def test_best_of_k_is_per_example_and_never_worse_than_baseline():
    base = TinyFlow()
    baseline = TaskNoisePolicy(base, cost, mode='baseline').predict_action(obs())
    best = TaskNoisePolicy(base, cost, mode='best_of_k', num_candidates=6).predict_action(obs())
    assert (cost(best['action_pred'], {}) <= cost(baseline['action_pred'], {})).all()


def test_zero_trust_radius_prevents_changes():
    result = TaskNoisePolicy(TinyFlow(), cost, num_candidates=1, num_grad_steps=2,
                              trust_radius=0, record_training_data=True).predict_action(obs(), context())
    torch.testing.assert_close(result['base_noise'], result['optimized_noise'], rtol=0, atol=0)
    assert not result['teacher_improved'].any()


def test_inference_mode_fails_clearly_and_train_keeps_base_in_eval():
    policy = TaskNoisePolicy(TinyFlow(), cost)
    policy.train()
    assert not policy.policy.training
    with torch.inference_mode(), pytest.raises(RuntimeError, match='outside torch.inference_mode'):
        policy.predict_action(obs())


def test_initializer_is_identity_before_training_and_learned_modes_work():
    init = NoiseInitializer(cond_dim=3, horizon=4, action_dim=7, n_obs_steps=2)
    base = TinyFlow()
    baseline = TaskNoisePolicy(base, cost, mode='baseline').predict_action(obs(), context())
    learned = TaskNoisePolicy(base, cost, mode='amortized', initializer=init).predict_action(obs(), context())
    torch.testing.assert_close(learned['z'], baseline['z'], rtol=0, atol=0)
    combined = TaskNoisePolicy(base, cost, mode='amortized_dno', initializer=init,
                              num_grad_steps=2, num_candidates=1).predict_action(obs(), context())
    assert combined['info']['final_objective'] < learned['info']['final_objective']


def test_objective_is_required_and_bad_shapes_do_not_silently_broadcast():
    with pytest.raises(ValueError, match='explicit task objective'):
        TaskNoisePolicy(TinyFlow())
    bad = TaskNoisePolicy(TinyFlow(), lambda a, c: a.square().mean())
    with pytest.raises(ValueError, match='one scalar per candidate'):
        bad.predict_action(obs())


def test_real_f_policy_source_matches_checkpoint_sampler():
    from test_learned_canonical_policy import make_policy, observations
    base = make_policy(reference_use='both', kappa=4.0)
    base.eval()
    observations_dict = observations(2)
    seed = 19
    policy = TaskNoisePolicy(base, cost, mode='baseline', seed=seed)
    actual = policy.predict_action(observations_dict)['z']
    with torch.no_grad():
        cond, theta, valid = base._prepare_observation(observations_dict)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            expected = base._sample_prior_with_heading(theta, valid, batch_size=2,
                                                       device=cond.device, dtype=cond.dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_learned_candidate_with_nonfinite_unscored_action_is_rejected():
    class CorruptRotationFlow(TinyFlow):
        def sample_chunk(self, cond, z):
            actions = z * self.weight
            bad = (z == 0).flatten(1).all(-1)
            actions[..., 3] = torch.where(bad[:, None], float('nan'), actions[..., 3])
            return actions

    class ZeroInitializer(nn.Module):
        def forward(self, cond, base_noise, ctx):
            return torch.zeros_like(base_noise)

    objective = lambda actions, ctx: actions[..., :3].square().mean((-1, -2))
    policy = TaskNoisePolicy(CorruptRotationFlow(), objective, mode='amortized',
                             initializer=ZeroInitializer(), trust_radius=10.0)
    out = policy.predict_action(obs(), context())
    assert torch.isfinite(out['action_pred']).all()
    assert out['info']['final_objective'] == out['info']['initial_objective']
