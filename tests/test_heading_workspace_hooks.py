"""Optional dataset initialization and independent clipping in the standard workspace."""

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from oat.workspace.train_policy import _clip_grad_norm, _configure_models_from_dataset


class DatasetConfiguredModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("heading_xy_rms", torch.tensor(1.))
        self.configure_calls = []
        self.normalizer_is_set = False

    def configure_from_dataset(self, dataset):
        assert self.normalizer_is_set
        self.configure_calls.append(dataset)
        self.heading_xy_rms.fill_(dataset.get_heading_xy_rms())


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Parameter(torch.zeros(2))
        self.heading = nn.Parameter(torch.zeros(1))
        self.frozen = nn.Parameter(torch.zeros(1), requires_grad=False)


class GroupedModel(TinyModel):
    def get_gradient_clip_groups(self):
        return OrderedDict(core=[self.core], heading=[self.heading])


class MockAccelerator:
    def __init__(self, policy, scale=1.):
        self.policy = policy
        self.scale = scale
        self.unscale_calls = []
        self.legacy_calls = []

    def unwrap_model(self, model):
        return self.policy

    def unscale_gradients(self, optimizer):
        self.unscale_calls.append(optimizer)
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.div_(self.scale)

    def clip_grad_norm_(self, parameters, max_norm):
        parameters = list(parameters)
        self.legacy_calls.append((parameters, max_norm))
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)


def gradients(model, core=(3., 4.), heading=12.):
    model.core.grad = torch.tensor(core)
    model.heading.grad = torch.tensor([heading])
    return torch.optim.SGD([model.core, model.heading], lr=.1)


def test_dataset_hook_is_optional_for_legacy_models():
    legacy = nn.Linear(2, 1)
    before = {key: value.clone() for key, value in legacy.state_dict().items()}
    _configure_models_from_dataset(object(), legacy, None, SimpleNamespace(configure_from_dataset=None))
    for key, value in legacy.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_dataset_hook_configures_model_and_ema_once_after_normalizers():
    dataset = SimpleNamespace(get_heading_xy_rms=lambda: .37)
    model, ema = DatasetConfiguredModel(), DatasetConfiguredModel()
    model.normalizer_is_set = ema.normalizer_is_set = True
    _configure_models_from_dataset(dataset, model, ema)
    for policy in (model, ema):
        assert policy.configure_calls == [dataset]
        torch.testing.assert_close(policy.heading_xy_rms, torch.tensor(.37))


def test_resumed_buffer_overrides_dataset_initialization():
    model = DatasetConfiguredModel()
    model.normalizer_is_set = True
    _configure_models_from_dataset(SimpleNamespace(get_heading_xy_rms=lambda: .37), model)
    model.load_state_dict({"heading_xy_rms": torch.tensor(.52)})
    torch.testing.assert_close(model.heading_xy_rms, torch.tensor(.52))
    assert len(model.configure_calls) == 1


def test_legacy_clipping_still_uses_accelerator_global_norm():
    model = TinyModel()
    optimizer = gradients(model)
    accelerator = MockAccelerator(model)
    norm = _clip_grad_norm(accelerator, model, optimizer, 1.)
    torch.testing.assert_close(norm, torch.tensor(13.))
    torch.testing.assert_close(model.core.grad.norm(), torch.tensor(5. / 13.))
    torch.testing.assert_close(model.heading.grad.norm(), torch.tensor(12. / 13.))
    assert len(accelerator.legacy_calls) == 1
    parameters, threshold = accelerator.legacy_calls[0]
    assert [id(p) for p in parameters] == [id(p) for p in model.parameters()]
    assert threshold == 1.
    assert not accelerator.unscale_calls


def test_groups_clip_independently_after_exactly_one_unscale():
    model = GroupedModel()
    optimizer = gradients(model, core=(6., 8.), heading=24.)
    accelerator = MockAccelerator(model, scale=2.)
    # Discovery uses the unwrapped policy even if the caller holds a wrapper.
    wrapper = SimpleNamespace(parameters=model.parameters)
    norms = _clip_grad_norm(accelerator, wrapper, optimizer, 1.)
    assert list(norms) == ["core", "heading"]
    torch.testing.assert_close(norms["core"], torch.tensor(5.))
    torch.testing.assert_close(norms["heading"], torch.tensor(12.))
    torch.testing.assert_close(model.core.grad.norm(), torch.tensor(1.))
    torch.testing.assert_close(model.heading.grad.norm(), torch.tensor(1.))
    assert accelerator.unscale_calls == [optimizer]
    assert not accelerator.legacy_calls


@pytest.mark.parametrize("problem", ["duplicate_across", "duplicate_within", "missing", "foreign", "frozen", "not_mapping"])
def test_invalid_groups_fail_before_mutating_gradients(problem):
    model = GroupedModel()
    optimizer = gradients(model)
    accelerator = MockAccelerator(model, scale=2.)
    cases = {
        "duplicate_across": {"core": [model.core], "heading": [model.heading, model.core]},
        "duplicate_within": {"core": [model.core, model.core], "heading": [model.heading]},
        "missing": {"core": [model.core]},
        "foreign": {"core": [model.core], "heading": [nn.Parameter(torch.zeros(1))]},
        "frozen": {"core": [model.core], "heading": [model.heading, model.frozen]},
        "not_mapping": [model.core, model.heading],
    }
    model.get_gradient_clip_groups = lambda: cases[problem]
    before = [model.core.grad.clone(), model.heading.grad.clone()]
    with pytest.raises(ValueError, match="Gradient clip"):
        _clip_grad_norm(accelerator, model, optimizer, 1.)
    assert not accelerator.unscale_calls and not accelerator.legacy_calls
    torch.testing.assert_close(model.core.grad, before[0], rtol=0, atol=0)
    torch.testing.assert_close(model.heading.grad, before[1], rtol=0, atol=0)


def test_group_clipping_rejects_nonfinite_gradients():
    model = GroupedModel()
    optimizer = gradients(model, core=(float("inf"), 1.))
    accelerator = MockAccelerator(model)
    with pytest.raises(RuntimeError, match="non-finite"):
        _clip_grad_norm(accelerator, model, optimizer, 1.)
    assert accelerator.unscale_calls == [optimizer]
    assert not accelerator.legacy_calls


def test_disabled_clipping_does_not_unscale_or_mutate_gradients():
    model = GroupedModel()
    optimizer = gradients(model)
    accelerator = MockAccelerator(model, scale=2.)
    assert _clip_grad_norm(accelerator, model, optimizer, None) is None
    assert not accelerator.unscale_calls and not accelerator.legacy_calls
    torch.testing.assert_close(model.core.grad, torch.tensor([3., 4.]), rtol=0, atol=0)
    torch.testing.assert_close(model.heading.grad, torch.tensor([12.]), rtol=0, atol=0)
