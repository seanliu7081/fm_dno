"""
A ``BasePolicy``-shaped wrapper that runs orbit DNO inside ``predict_action``.

``LiberoRunner`` only ever touches a policy through ``device``, ``dtype``,
``get_policy_name()``, ``get_observation_ports()``, ``reset()`` and
``predict_action(obs_dict)['action']``.  Implementing exactly that interface means the
stock runner and the stock ``eval_policy_sim.py`` can drive a DNO-steered policy with no
change to any existing file.

ONE HARD CONSTRAINT
-------------------
``LiberoRunner.run`` is decorated with ``@torch.inference_mode()`` and wraps the policy call
in another ``inference_mode`` block.  Stage 1 of orbit DNO (the angular grid search) is pure
forward evaluation and runs fine there.  Stage 2 (gradient descent through the sampler)
cannot: tensors created under inference mode are not usable in an autograd graph.

So:

    n_grad_steps == 0  ->  works with the stock LiberoRunner
    n_grad_steps  > 0  ->  needs oat.env_runner.libero_dno_runner.OrbitDnoLiberoRunner

``predict_action`` raises a clear error rather than failing deep inside autograd if that is
violated.  Run stage-1-only first anyway: it is the configuration that isolates the claim.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch

from oat.policy.base_policy import BasePolicy


def libero_ctx_fn(obs_dict: Dict[str, torch.Tensor]) -> Dict:
    """Context for the task losses, read out of the raw (un-normalized) observation.

    ``robot0_eef_pos`` is a ``state`` port of shape [3], so ``[:, -1]`` is the current
    end-effector position in metres.  Observations reach ``predict_action`` un-normalized
    (the encoder normalizes internally), which is what the geometric losses want.
    """
    ctx: Dict = {}
    if "robot0_eef_pos" in obs_dict:
        ctx["eef_pos"] = obs_dict["robot0_eef_pos"][:, -1].float()
    return ctx


class DNOPolicy(BasePolicy):
    """Wrap a frozen ``OrbitFlowPolicy`` so every control cycle runs orbit DNO.

    Args:
        policy: the frozen policy.  Put it in ``eval()`` and freeze it before wrapping;
            DNO optimizes the noise, never the weights.
        dno: an ``OrbitDNO`` bound to ``policy``.
        ctx_fn: builds the task-loss context from the observation dict.
        enabled: set False to fall through to the underlying ``predict_action``.  This is
            how the {DNO on, off} half of the 2x2 is run against one checkpoint.
    """

    def __init__(
        self,
        policy,
        dno,
        ctx_fn: Optional[Callable[[Dict], Dict]] = None,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.dno = dno
        self.ctx_fn = ctx_fn or libero_ctx_fn
        self.enabled = bool(enabled)
        self.info_log: List[Dict] = []
        for p in self.policy.parameters():
            p.requires_grad_(False)

    # -- BasePolicy plumbing -----------------------------------------------------------

    def get_policy_name(self) -> str:
        tag = "dno" if self.enabled else "nodno"
        return f"{tag}_{self.policy.get_policy_name()}"

    def get_observation_ports(self) -> List[str]:
        return self.policy.get_observation_ports()

    def get_observation_modalities(self) -> List[str]:
        return self.policy.get_observation_modalities()

    def get_observation_encoder(self):
        return self.policy.get_observation_encoder()

    def set_normalizer(self, normalizer):
        return self.policy.set_normalizer(normalizer)

    @property
    def n_obs_steps(self):
        return self.policy.n_obs_steps

    @property
    def n_action_steps(self):
        return self.policy.n_action_steps

    def reset(self):
        """Called by the runner at every episode boundary -- clears the DNO warm start."""
        self.dno.reset()
        self.policy.reset()

    # -- the actual work ---------------------------------------------------------------

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        if not self.enabled:
            return self.policy.predict_action(obs_dict)

        if self.dno.n_grad_steps > 0 and torch.is_inference_mode_enabled():
            raise RuntimeError(
                "OrbitDNO stage 2 needs gradients, but the rollout is running under "
                "torch.inference_mode(). Either set n_grad_steps=0 (stage-1 orbit search "
                "only, which needs no gradients), or drive the rollout with "
                "oat.env_runner.libero_dno_runner.OrbitDnoLiberoRunner."
            )

        out = self.dno(obs_dict, self.ctx_fn(obs_dict))
        info = dict(out["info"])
        info.pop("grad_loss_curve", None)
        self.info_log.append(info)
        return {"action": out["action"], "action_pred": out["action_pred"]}

    # -- reporting ---------------------------------------------------------------------

    def summarize(self) -> Dict[str, float]:
        """Mean of every scalar in the per-cycle info log.  Log this next to success rate.

        ``wall_time`` is the number that decides whether this is a control method or an
        offline one: at 20 Hz with n_action_steps=8 the budget is 0.4 s per call.
        """
        if not self.info_log:
            return {}
        keys = [k for k, v in self.info_log[0].items() if isinstance(v, (int, float))]
        n = len(self.info_log)
        out = {f"dno/{k}": sum(d.get(k, 0.0) for d in self.info_log) / n for k in keys}
        out["dno/n_calls"] = float(n)
        times = [d["wall_time"] for d in self.info_log if "wall_time" in d]
        if times:
            times = sorted(times)
            out["dno/wall_time_p50"] = times[len(times) // 2]
            out["dno/wall_time_p95"] = times[min(len(times) - 1, int(0.95 * len(times)))]
            out["dno/wall_time_max"] = times[-1]
        return out

    def clear_log(self) -> None:
        self.info_log = []
