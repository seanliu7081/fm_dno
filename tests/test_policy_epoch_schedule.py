"""Check actual evaluation milestones, including the final completed epoch."""
import pytest
from omegaconf import OmegaConf

from oat.workspace.train_policy import _epoch_events


def config(offset=0):
    return OmegaConf.create(dict(num_epochs=150, rollout_every=15,
                                 checkpoint_every=5, rollout_epoch_offset=offset))


def test_completed_epoch_schedule_and_permanent_checkpoint_coverage():
    events = [_epoch_events(i, config(1), lazy_eval=False,
                           save_rollout_checkpoint=True, save_final_checkpoint=True)
              for i in range(150)]
    assert [i + 1 for i, (rollout, _) in enumerate(events) if rollout] == list(range(15, 151, 15))
    assert all(saved for rollout, saved in events if rollout)
    assert events[-1] == (True, True)
    assert not events[0][0]


def test_legacy_default_cadence_is_preserved():
    cfg = config()
    del cfg.rollout_epoch_offset
    events = [_epoch_events(i, cfg, lazy_eval=False) for i in range(150)]
    assert [i for i, (rollout, _) in enumerate(events) if rollout] == list(range(0, 150, 15))
    assert [i for i, (_, saved) in enumerate(events) if saved] == list(range(0, 150, 5))


def test_resume_at_completed_milestone_does_not_repeat_it():
    # Epoch14 completed; restored workspace starts at zero-based epoch15.
    assert [i + 1 for i in range(15, 150)
            if _epoch_events(i, config(1), lazy_eval=False)[0]] == list(range(30, 151, 15))


def test_lazy_eval_suppresses_rollouts_but_final_checkpoint_is_retained():
    events = [_epoch_events(i, config(1), lazy_eval=True,
                           save_rollout_checkpoint=True, save_final_checkpoint=True)
              for i in range(150)]
    assert not any(rollout for rollout, _ in events)
    assert events[-1] == (False, True)


@pytest.mark.parametrize('key,value', [('rollout_epoch_offset', 2),
                                     ('rollout_every', 0), ('checkpoint_every', 0)])
def test_invalid_schedule_is_rejected(key, value):
    cfg = config()
    cfg[key] = value
    with pytest.raises(ValueError):
        _epoch_events(0, cfg, lazy_eval=False)
