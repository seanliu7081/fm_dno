"""EMA weights with exact synchronization of the self-past training clock."""

import torch

from oat.model.diffusion.ema_model import EMAModel


class SelfPastEMAModel(EMAModel):
    """Keep an optimizer-update counter as state, never as an averaged weight.

    The policy advances ``self_past_optimizer_step`` in its optimizer's
    post-step hook. The workspace updates EMA after that optimizer step, so
    both policies use the same self-past schedule and retain it on resume.
    """

    _COUNTER_NAME = "self_past_optimizer_step"
    _INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)

    def __init__(self, model, *args, **kwargs):
        self._counter(model)
        super().__init__(model, *args, **kwargs)

    @classmethod
    def _counter(cls, model):
        counter = dict(model.named_buffers(recurse=False)).get(cls._COUNTER_NAME)
        if counter is None:
            raise ValueError(f"Model must register the {cls._COUNTER_NAME!r} buffer")
        if counter.ndim != 0 or counter.dtype not in cls._INTEGER_DTYPES:
            raise ValueError(f"{cls._COUNTER_NAME} must be a scalar integer buffer")
        if cls._COUNTER_NAME in model._non_persistent_buffers_set:
            raise ValueError(f"{cls._COUNTER_NAME} must be persistent for checkpoint resume")
        return counter

    @torch.no_grad()
    def step(self, new_model):
        counter = self._counter(new_model)
        ema_counter = self._counter(self.averaged_model)
        if counter.dtype != ema_counter.dtype:
            raise ValueError("Online and EMA self-past counter dtypes must match")
        super().step(new_model)
        ema_counter.copy_(counter)
