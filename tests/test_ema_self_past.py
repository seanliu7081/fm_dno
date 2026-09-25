"""The self-past clock counts real optimizer updates and survives EMA/resume."""

import copy

import pytest
import torch
from torch import nn

from oat.model.diffusion.ema_self_past import SelfPastEMAModel


class TinySelfPastPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("self_past_optimizer_step", torch.tensor(0, dtype=torch.long))

    def forward(self, value):
        return self.weight * value

    def get_optimizer(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.01)

        @torch.no_grad()
        def increment_step(optimizer, args, kwargs):
            self.self_past_optimizer_step.add_(1)

        optimizer.register_step_post_hook(increment_step)
        return optimizer


def test_counter_tracks_updates_without_averaging_or_forward_increments():
    policy = TinySelfPastPolicy()
    optimizer = policy.get_optimizer()
    averaged_policy = copy.deepcopy(policy)
    ema = SelfPastEMAModel(averaged_policy)

    for update in range(1, 8):
        for _ in range(3):
            policy(torch.tensor(2.0)).square().backward()
        assert policy.self_past_optimizer_step.item() == update - 1
        old_ema_weight = averaged_policy.weight.detach().clone()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        ema.step(policy)
        assert policy.self_past_optimizer_step.item() == update
        assert averaged_policy.self_past_optimizer_step.item() == update
        expected_weight = old_ema_weight * ema.decay + policy.weight * (1 - ema.decay)
        torch.testing.assert_close(averaged_policy.weight, expected_weight)

    assert ema.decay > 0
    with torch.inference_mode():
        policy.eval()(torch.tensor(2.0))
        averaged_policy.eval()(torch.tensor(2.0))
    assert policy.self_past_optimizer_step.item() == 7
    assert averaged_policy.self_past_optimizer_step.item() == 7


def test_checkpoint_state_and_optimizer_hook_resume_the_same_clock():
    policy = TinySelfPastPolicy()
    optimizer = policy.get_optimizer()
    ema = SelfPastEMAModel(copy.deepcopy(policy))
    for _ in range(5):
        policy(torch.tensor(2.0)).square().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        ema.step(policy)

    restored = TinySelfPastPolicy()
    restored_optimizer = restored.get_optimizer()
    restored_ema = SelfPastEMAModel(copy.deepcopy(restored))
    restored.load_state_dict(copy.deepcopy(policy.state_dict()))
    restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    restored_ema.averaged_model.load_state_dict(copy.deepcopy(ema.averaged_model.state_dict()))
    restored_ema.load_state_dict(ema.state_dict())

    for model, opt, controller in ((policy, optimizer, ema),
                                   (restored, restored_optimizer, restored_ema)):
        model(torch.tensor(2.0)).square().backward()
        opt.step()
        controller.step(model)

    assert restored.self_past_optimizer_step.item() == 6
    assert restored_ema.averaged_model.self_past_optimizer_step.item() == 6
    assert restored_ema.state_dict() == ema.state_dict()
    for key, value in ema.averaged_model.state_dict().items():
        torch.testing.assert_close(restored_ema.averaged_model.state_dict()[key], value,
                                   rtol=0, atol=0)


@pytest.mark.parametrize("counter", [None, torch.zeros(1, dtype=torch.long),
                                    torch.tensor(0.0), torch.tensor(False)])
def test_rejects_missing_or_non_scalar_integer_counter(counter):
    policy = nn.Linear(1, 1)
    if counter is not None:
        policy.register_buffer("self_past_optimizer_step", counter)
    with pytest.raises(ValueError, match="self_past_optimizer_step"):
        SelfPastEMAModel(policy)


def test_rejects_invalid_online_counter_before_mutating_ema():
    ema = SelfPastEMAModel(TinySelfPastPolicy())
    invalid = TinySelfPastPolicy()
    invalid.self_past_optimizer_step = torch.tensor(1.5)
    old_weight = ema.averaged_model.weight.detach().clone()
    with pytest.raises(ValueError, match="scalar integer"):
        ema.step(invalid)
    assert ema.optimization_step == 0
    torch.testing.assert_close(ema.averaged_model.weight, old_weight, rtol=0, atol=0)


def test_rejects_nonpersistent_counter():
    policy = nn.Linear(1, 1)
    policy.register_buffer("self_past_optimizer_step", torch.tensor(0), persistent=False)
    with pytest.raises(ValueError, match="persistent"):
        SelfPastEMAModel(policy)
