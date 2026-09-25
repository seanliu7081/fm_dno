"""Causal history and stateless self-past regression tests with a tiny encoder."""

import copy

import pytest
import torch
from torch import nn

from oat.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from oat.model.diffusion.ema_self_past import SelfPastEMAModel
from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.policy.flow_policy_spatial_motion_self_past import SpatialMotionSelfPastFlowPolicy


class TinyObservationEncoder(BaseObservationEncoder):
    rgb_ports = ['image']

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 16)
        self.normalizer = LinearNormalizer()

    def forward(self, obs):
        return self.projection(obs['image'])

    def modalities(self):
        return ['rgb']

    def output_feature_dim(self):
        return 16

    def output_token_count(self):
        return 2

    def set_normalizer(self, normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())


@pytest.fixture(autouse=True)
def deterministic_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(27)
    yield
    torch.set_num_threads(threads)


def make_policy(**kwargs):
    options = dict(
        shape_meta={'obs': {'image': {'shape': [3], 'type': 'rgb'}},
                    'action': {'shape': [7]}},
        obs_encoder=TinyObservationEncoder(), horizon=16, n_action_steps=8,
        n_obs_steps=2, embed_dim=32, n_layers=2, n_heads=4, dropout=0.0,
        num_inference_steps=2, motion_decoder_layers=1, motion_decoder_heads=2,
        motion_decoder_ff_dim=32, inference_bf16=False,
        backbone_kwargs={'num_register_tokens': 1}, min_source_confidence=0.0,
        self_past_warmup_steps=2, self_past_ramp_steps=2,
    )
    options.update(kwargs)
    policy = SpatialMotionSelfPastFlowPolicy(**options)
    normalizer = LinearNormalizer()
    normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
        torch.stack((-torch.ones(7), torch.ones(7))), mode='limits')
    policy.set_normalizer(normalizer)
    return policy


def make_batch(batch_size=3):
    context = {
        'past_action': torch.randn(batch_size, 7, 7),
        'past_mask': torch.ones(batch_size, 7, dtype=torch.bool),
        'prev_obs': {'image': torch.randn(batch_size, 2, 3)},
        'prev_past_action': torch.randn(batch_size, 7, 7),
        'prev_past_mask': torch.ones(batch_size, 7, dtype=torch.bool),
        'prev_window_valid': torch.ones(batch_size, dtype=torch.bool),
    }
    context['past_mask'][0, :3] = False
    obs = {'image': torch.randn(batch_size, 2, 3), '_p2n_context': context}
    return {'obs': obs, 'action': torch.randn(batch_size, 16, 7),
            'action_mask': torch.ones(batch_size, 16, dtype=torch.bool), **context}


def make_optimizer(policy):
    return policy.get_optimizer(policy_lr=1e-3, obs_enc_lr=1e-3,
                                weight_decay=1e-4, betas=(0.9, 0.999))


def test_history_order_and_invalid_values_cannot_leak_into_condition():
    policy = make_policy().eval()
    batch = make_batch()
    past, mask = batch['past_action'], batch['past_mask']
    condition, prediction = policy.encode_condition(batch['obs'], past, mask)
    assert condition.shape == (3, 2 + 7 + 2 + 4, 16)
    assert policy.model.cond_len == condition.shape[1]
    invalid_values = past.clone()
    invalid_values[~mask] = float('nan')
    sanitized, sanitized_prediction = policy.encode_condition(batch['obs'], invalid_values, mask)
    assert torch.isfinite(sanitized).all()
    torch.testing.assert_close(sanitized, condition, rtol=0, atol=0)
    torch.testing.assert_close(sanitized_prediction['direction'], prediction['direction'], rtol=0, atol=0)

    # Swap only older valid actions: the last-three difference tokens stay the
    # same, so the motion decoder must use the raw tokens' lag embeddings.
    reordered = past.clone()
    reordered[1, :2] = reordered[1, :2].flip(0)
    changed, changed_prediction = policy.encode_condition(batch['obs'], reordered, mask)
    assert not torch.allclose(changed_prediction['direction'][1], prediction['direction'][1], atol=1e-7)
    assert not torch.equal(changed[1, 2:4], condition[1, 2:4])
    torch.testing.assert_close(changed[1, 9:11], condition[1, 9:11])


def test_self_past_selects_executed_slice_and_skips_early_windows(monkeypatch):
    policy = make_policy()
    batch = make_batch()
    batch['prev_window_valid'] = torch.tensor([True, False, True])
    calls = []

    def fake_sample(obs, history, mask):
        calls.append((obs, history, mask))
        return torch.arange(16.).view(1, 16, 1).expand(len(history), 16, 7).requires_grad_()

    monkeypatch.setattr(policy, 'sample_actions', fake_sample)
    selected, mask, used = policy._select_history(batch, probability=1.0)
    assert used.tolist() == [True, False, True]
    expected = torch.arange(1., 8.).view(7, 1).expand(7, 7)
    torch.testing.assert_close(selected[0], expected)
    torch.testing.assert_close(selected[2], expected)
    torch.testing.assert_close(selected[1], batch['past_action'][1])
    assert mask[[0, 2]].all()
    assert not selected.requires_grad
    assert len(calls) == 1 and len(calls[0][1]) == 2
    torch.testing.assert_close(calls[0][0]['image'], batch['prev_obs']['image'][[0, 2]])

    # Dataset marks t < execution stride as invalid; no generation is allowed.
    batch['prev_window_valid'].zero_()
    selected, _, used = policy._select_history(batch, probability=1.0)
    assert not used.any() and len(calls) == 1
    torch.testing.assert_close(selected, batch['past_action'])


def test_sampler_restores_heterogeneous_modes_and_has_no_inner_graph():
    policy = make_policy().train()
    policy.history_encoder.delta2_proj.eval()
    before = [module.training for module in policy.modules()]
    observed = []

    def inspect(module, args, output):
        observed.append((torch.is_grad_enabled(), module.training, output.requires_grad))

    handles = [module.register_forward_hook(inspect) for module in
               (policy.obs_encoder, policy.history_encoder, policy.model)]
    batch = make_batch()
    try:
        output = policy.sample_actions(batch['obs'], batch['past_action'], batch['past_mask'])
    finally:
        for handle in handles:
            handle.remove()
    assert observed and all(item == (False, False, False) for item in observed)
    assert not output.requires_grad and output.grad_fn is None
    assert [module.training for module in policy.modules()] == before
    assert policy.self_past_optimizer_step.item() == 0


def test_future_mask_changes_labels_without_changing_condition_source_or_flow():
    policy = make_policy().train()
    batch = make_batch()
    noise = torch.randn(3, 16, 7)
    time = torch.tensor([0.1, 0.5, 0.9])
    normal = policy.loss_components(batch, noise=noise, t=time)
    masked = policy.loss_components({**batch, 'action_mask': torch.zeros_like(batch['action_mask'])},
                                    noise=noise, t=time)
    for key in normal['prediction']:
        torch.testing.assert_close(masked['prediction'][key], normal['prediction'][key], rtol=0, atol=0)
    torch.testing.assert_close(masked['source'], normal['source'], rtol=0, atol=0)
    torch.testing.assert_close(masked['flow_loss'], normal['flow_loss'], rtol=0, atol=0)
    assert normal['target']['has_data'].all() and not masked['target']['has_data'].any()
    assert not normal['source'].requires_grad
    assert normal['prediction']['motion_tokens'].requires_grad


def test_context_evaluation_is_stateless_and_independent_of_rollout_history(monkeypatch):
    policy = make_policy().eval()
    batch = make_batch()
    generated = torch.arange(16.).view(1, 16, 1).expand(3, 16, 7)
    calls = []

    def fake_sample(obs, history, mask, noise=None):
        calls.append((history.detach().clone(), mask.clone()))
        # Make output history-dependent to detect accidental online-buffer use.
        return generated[:len(history)] + history.mean(dim=(1, 2))[:, None, None]

    monkeypatch.setattr(policy, 'sample_actions', fake_sample)
    policy.record_executed_actions(torch.full((3, 5, 7), 123.))
    saved_buffer, saved_mask = policy._past_buffer.clone(), policy._past_mask.clone()
    result = policy.predict_action(batch['obs'])
    assert len(calls) == 2  # previous window, then current window
    torch.testing.assert_close(calls[0][0], batch['prev_past_action'])
    expected_generated = generated + batch['prev_past_action'].mean(dim=(1, 2))[:, None, None]
    torch.testing.assert_close(calls[1][0], expected_generated[:, 1:8])
    torch.testing.assert_close(policy._past_buffer, saved_buffer)
    torch.testing.assert_close(policy._past_mask, saved_mask)
    policy.record_executed_actions(torch.full((3, 7, 7), -456.))
    repeat = policy.predict_action(batch['obs'])
    torch.testing.assert_close(repeat['action_pred'], result['action_pred'])
    assert policy.self_past_optimizer_step.item() == 0


def test_explicit_history_and_actual_execution_update_have_separate_state():
    policy = make_policy().eval()
    batch = make_batch()
    obs = {'image': batch['obs']['image']}
    noise = torch.zeros(3, 16, 7)
    policy.predict_action(obs, past_action=batch['past_action'], past_mask=batch['past_mask'], noise=noise)
    assert policy._past_buffer is None
    policy.predict_action(obs, noise=noise, update_history=False)
    assert not policy._past_mask.any()
    actual = torch.randn(3, 3, 7)
    policy.record_executed_actions(actual)
    torch.testing.assert_close(policy._past_buffer[:, -3:], actual)
    assert policy._past_mask[:, -3:].all() and not policy._past_mask[:, :-3].any()
    policy.reset()
    assert policy._past_buffer is None and policy._past_mask is None


def test_training_self_past_preserves_outer_gradients_and_optimizer_coverage():
    policy = make_policy(self_past_warmup_steps=0, self_past_ramp_steps=0).train()
    optimizer = make_optimizer(policy)
    batch = make_batch()
    batch['prev_obs']['image'].requires_grad_()
    batch['prev_past_action'].requires_grad_()
    expected = {id(parameter) for parameter in policy.parameters() if parameter.requires_grad}
    covered = [id(parameter) for group in optimizer.param_groups for parameter in group['params']]
    assert len(covered) == len(set(covered)) and set(covered) == expected
    loss = policy(batch)
    loss.backward()
    assert batch['prev_obs']['image'].grad is None
    assert batch['prev_past_action'].grad is None
    for name, parameter in policy.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    for module in (policy.history_encoder.raw_proj, policy.history_encoder.delta_proj,
                   policy.history_encoder.delta2_proj, policy.obs_encoder):
        assert sum(parameter.grad.abs().sum().item() for parameter in module.parameters()
                   if parameter.requires_grad) > 0
    assert policy.self_past_optimizer_step.item() == 0
    optimizer.step()
    assert policy.self_past_optimizer_step.item() == 1


def test_actual_policy_schedule_counter_ema_and_resume_follow_optimizer_updates():
    policy = make_policy().train()
    optimizer = make_optimizer(policy)
    ema = SelfPastEMAModel(copy.deepcopy(policy))
    batch = make_batch()
    for step, expected_probability in enumerate((0., 0., 0., .5, 1.)):
        assert policy.self_past_optimizer_step.item() == step
        assert policy.training_self_past_probability() == expected_probability
        # Accumulation does not advance the schedule until the optimizer runs.
        for _ in range(2):
            (policy(batch) / 2).backward()
        assert policy.self_past_optimizer_step.item() == step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        ema.step(policy)
        assert ema.averaged_model.self_past_optimizer_step.item() == step + 1
    policy.eval()
    with torch.inference_mode():
        result = policy.loss_components(batch)
    assert result['self_past_probability'] == 1 and result['self_past_used'].all()
    assert policy.self_past_optimizer_step.item() == 5

    restored = make_policy()
    restored_optimizer = make_optimizer(restored)
    restored.load_state_dict(copy.deepcopy(policy.state_dict()))
    restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    assert restored.training_self_past_probability() == 1
    restored(batch).backward()
    restored_optimizer.step()
    assert restored.self_past_optimizer_step.item() == 6
