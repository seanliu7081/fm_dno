"""
Orbit-structured Diffusion Noise Optimization for a frozen flow policy.

THE IDEA
--------
Plain DNO treats the source noise as an unstructured vector in R^{H x A} and searches it
with hundreds of gradient steps through the full sampler.  That is why the literature has
DNO doing offline motion editing and environment generation, and almost never closed-loop
robot control: at LIBERO's replanning rate you get a few tens of milliseconds, not
minutes.

The orbit-aligned coupling changes what the source space *is*.  Because training paired
each source with a target on the same orbit, the source's global phase becomes the
generated chunk's global heading: rotating ``z`` by ``phi`` rotates the produced action
chunk by roughly ``phi`` (this is what ``metrics.orbit_steering_gain`` measures, and it
is near 0 for an iid-coupled policy).  That hands DNO a one-dimensional, compact,
periodic coordinate that captures the single most important thing a manipulation policy
gets wrong -- which way to go -- and it is a coordinate you can search *exhaustively*
rather than descend.

So the optimization splits:

    Stage 1  ORBIT SEARCH.  Evaluate K rotations of z on a grid over [0, 2pi).  One
             batched forward pass, no gradients, no local minima.  K = 32 costs the same
             as one forward with batch 32.
    Stage 2  RESIDUAL DESCENT.  A handful of IrrepAdam steps on the full z, started from
             the best orbit point, with a per-irrep typical-set regulariser and a trust
             region around the initial noise.

THE CONTRACT (measured, not assumed)
------------------------------------
A steering gain near 1 means the policy has stopped inferring the chunk heading from the
observation and reads it off the noise phase instead.  Run open-loop from random noise,
that policy is *worse*: in the synthetic study conditional heading error rises from 0.68
to 1.16 rad while the marginal stays plausible.  That is not a defect to regularise away,
it is the deal.  A policy that delegates the heading to the noise must be *given* the
heading, and stage 1 is how you give it -- after which the coupled policy is more
conditionally accurate than the iid one (0.436 vs 0.476 rad) and lands at -0.99 against an
objective whose optimum is -1.0.

So the coupling and this module are one mechanism, not two that compose.  Never report an
orbit-coupled policy's open-loop numbers as the method; that configuration is the ablation.

The cleanest experiment in the project is the 2x2: {iid, orbit-coupled} x {stage 1 on,
off}.  Be precise about the prediction: with steering gain 0, rotating an iid policy's
source yields K *unrelated* samples, so stage 1 degenerates into best-of-K resampling,
which still helps a noisy objective.  The claim is the sharper one -- at identical compute
the coupling converts a blind best-of-K draw into a structured 1-D search.  Measured: 3.8x
more task-loss reduction per forward pass.

BUDGET
------
At LIBERO's 20 Hz with ``n_action_steps=8`` a new chunk is due every 0.4 s.  For the
repo's 4-layer / 256-dim transformer with ``num_inference_steps=10``, a batch-1 forward
is on the order of 5-10 ms on a modern GPU, and a backward through the 10-step chain
roughly 3x that.  So a budget of one stage-1 grid (~1 forward-equivalent at K=32) plus
6-10 gradient steps lands inside the control period.  Measure it on your hardware before
believing it; ``last_info['wall_time']`` is reported for that purpose.

HONEST LIMITS
-------------
* A task loss is a proxy.  Satisfying it is not success, and a strong weight on a crude
  proxy is reward hacking with extra steps.  The typicality regulariser and the trust
  region bound how far the search can drag the chunk off the demonstration manifold; they
  do not make the proxy correct.
* The kinematic rollout inside the geometric losses ignores contact and controller lag.
* Stage 1's benefit is *contingent on* the steering gain being high.  Report the gain
  alongside every DNO result; if it is near zero, stage 1 is a lottery and should be
  reported as one.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, Optional

import torch

from oat.dno.irrep_adam import IrrepAdam, ScalarSGD
from oat.symmetry.so2_chunk import SO2ChunkSpec, rotate_chunk


class OrbitDNO:
    """Two-stage test-time noise optimization for an ``OrbitFlowPolicy``.

    Args:
        policy: a frozen policy exposing ``encode_obs``, ``sample_chunk``,
            ``sample_prior``, ``normalizer``, ``horizon``, ``action_dim``,
            ``n_action_steps``.
        task_loss: callable ``(actions_unnormalized (B,H,A), ctx) -> (B,)``.
        n_orbit: size of the stage-1 angular grid.  0 disables stage 1.
        n_grad_steps: stage-2 iterations.  0 disables stage 2.
        lr: stage-2 step size in noise space.
        typicality_weight: weight of the per-irrep Gaussian typical-set penalty.
        trust_weight: weight of ``||z - z_init||^2 / D``.
        trust_radius: optional hard projection radius (in units of sqrt(D)); ``None``
            leaves only the soft penalty.
        n_steps: sampler steps during optimization.  Defaults to the policy's.  Using a
            smaller value here than at execution time is a legitimate speedup and a
            legitimate source of a mismatch -- report which you did.
        optimizer: ``irrep_adam`` (rotation-compatible) or ``sgd`` or ``adam``.
        warm_start: carry the previous cycle's solution forward, shifted along the
            horizon, MPC style.
        max_batch: chunk size for the stage-1 grid evaluation.
    """

    def __init__(
        self,
        policy,
        task_loss: Callable[[torch.Tensor, Optional[Dict]], torch.Tensor],
        spec: Optional[SO2ChunkSpec] = None,
        n_orbit: int = 32,
        n_grad_steps: int = 8,
        lr: float = 0.05,
        typicality_weight: float = 0.02,
        trust_weight: float = 0.01,
        trust_radius: Optional[float] = None,
        n_steps: Optional[int] = None,
        optimizer: str = "irrep_adam",
        warm_start: bool = True,
        warm_start_noise: float = 0.3,
        max_batch: int = 256,
    ) -> None:
        self.policy = policy
        self.task_loss = task_loss
        self.spec = spec or getattr(policy, "action_spec", None) or SO2ChunkSpec.libero_osc_pose()
        self.n_orbit = int(n_orbit)
        self.n_grad_steps = int(n_grad_steps)
        self.lr = float(lr)
        self.typicality_weight = float(typicality_weight)
        self.trust_weight = float(trust_weight)
        self.trust_radius = trust_radius
        self.n_steps = n_steps
        self.optimizer_name = optimizer
        self.warm_start = bool(warm_start)
        self.warm_start_noise = float(warm_start_noise)
        self.max_batch = int(max_batch)
        self.reset()

    def reset(self) -> None:
        self._prev_z: Optional[torch.Tensor] = None
        self._prev_action: Optional[torch.Tensor] = None
        self.last_info: Dict = {}

    # -- main entry point ---------------------------------------------------------------

    def __call__(self, obs_dict: Dict[str, torch.Tensor], ctx: Optional[Dict] = None) -> Dict:
        t_start = time.perf_counter()
        ctx = dict(ctx or {})
        policy = self.policy

        with torch.no_grad():
            cond = policy.encode_obs(obs_dict)
        B = cond.shape[0]
        device, dtype = cond.device, cond.dtype

        z_init = self._initial_noise(B, device, dtype)
        if ctx.get("prev_action") is None and self._prev_action is not None:
            ctx["prev_action"] = self._prev_action

        info: Dict = {}
        z = z_init

        if self.n_orbit > 0:
            z, orbit_info = self._orbit_search(cond, z, ctx)
            info.update(orbit_info)

        if self.n_grad_steps > 0:
            z, grad_info = self._residual_descent(cond, z, z_init, ctx)
            info.update(grad_info)

        with torch.no_grad():
            x = policy.sample_chunk(cond, z, n_steps=self.n_steps)
            action_pred = policy.normalizer["action"].unnormalize(x)
            info["final_loss"] = float(self.task_loss(action_pred, ctx).mean())
            info["z_norm_ratio"] = float(
                z.reshape(B, -1).norm(dim=-1).mean()
                / z_init.reshape(B, -1).norm(dim=-1).mean().clamp_min(1e-8)
            )
            info["z_shift"] = float((z - z_init).reshape(B, -1).norm(dim=-1).mean())

        self._prev_z = z.detach()
        self._prev_action = action_pred[:, policy.n_action_steps - 1].detach()
        info["wall_time"] = time.perf_counter() - t_start
        self.last_info = info

        return {
            "action": action_pred[:, : policy.n_action_steps],
            "action_pred": action_pred,
            "z": z.detach(),
            "info": info,
        }

    # -- stages -------------------------------------------------------------------------

    @torch.no_grad()
    def _orbit_search(self, cond, z, ctx):
        """Stage 1: exhaustive search over the group orbit of ``z``."""
        policy, spec = self.policy, self.spec
        B, K = z.shape[0], self.n_orbit
        angles = torch.arange(K, device=z.device, dtype=z.dtype) * (2 * math.pi / K)

        z_rep = z.repeat_interleave(K, dim=0)                       # (B*K, H, A)
        phi = angles.repeat(B)                                      # (B*K,)
        z_grid = rotate_chunk(z_rep, phi, spec)
        cond_rep = cond.repeat_interleave(K, dim=0)

        losses = []
        for s in range(0, z_grid.shape[0], self.max_batch):
            e = min(s + self.max_batch, z_grid.shape[0])
            x = policy.sample_chunk(cond_rep[s:e], z_grid[s:e], n_steps=self.n_steps)
            a = policy.normalizer["action"].unnormalize(x)
            losses.append(self.task_loss(a, self._slice_ctx(ctx, s, e, K)))
        loss = torch.cat(losses).reshape(B, K)

        best = loss.argmin(dim=-1)                                  # (B,)
        z_best = rotate_chunk(z, angles[best], spec)
        return z_best, {
            "orbit_best_angle": float(angles[best].mean()),
            "orbit_loss_best": float(loss.min(dim=-1).values.mean()),
            "orbit_loss_mean": float(loss.mean()),
            "orbit_loss_spread": float((loss.max(-1).values - loss.min(-1).values).mean()),
            "orbit_n_eval": B * K,
        }

    def _residual_descent(self, cond, z0, z_init, ctx):
        """Stage 2: a few preconditioned gradient steps through the whole sampler."""
        policy = self.policy
        z = z0.detach().clone()
        D = z[0].numel()
        opt = self._make_optimizer()

        history = []
        for _ in range(self.n_grad_steps):
            z = z.detach().requires_grad_(True)
            x = policy.sample_chunk(cond, z, n_steps=self.n_steps)
            a = policy.normalizer["action"].unnormalize(x)
            loss = self.task_loss(a, ctx)
            reg = (
                self.typicality_weight * typicality_penalty(z, self.spec)
                + self.trust_weight * ((z - z_init) ** 2).sum(dim=(-1, -2)) / D
            )
            total = (loss + reg).sum()
            (grad,) = torch.autograd.grad(total, z)
            history.append(float(loss.mean()))
            z = opt.step(z.detach(), grad)
            if self.trust_radius is not None:
                z = project_to_ball(z, z_init, self.trust_radius * math.sqrt(D))

        return z.detach(), {
            "grad_loss_start": history[0] if history else float("nan"),
            "grad_loss_end": history[-1] if history else float("nan"),
            "grad_loss_curve": history,
        }

    # -- helpers ------------------------------------------------------------------------

    def _make_optimizer(self):
        if self.optimizer_name == "irrep_adam":
            return IrrepAdam(self.spec, lr=self.lr, equivariant=True)
        if self.optimizer_name == "adam":
            return IrrepAdam(self.spec, lr=self.lr, equivariant=False)
        if self.optimizer_name == "sgd":
            return ScalarSGD(lr=self.lr)
        raise ValueError(f"unknown optimizer {self.optimizer_name}")

    def _initial_noise(self, B, device, dtype):
        if self.warm_start and self._prev_z is not None and self._prev_z.shape[0] == B:
            z = shift_noise(self._prev_z, self.policy.n_action_steps)
            if self.warm_start_noise > 0:
                z = math.sqrt(1 - self.warm_start_noise ** 2) * z + self.warm_start_noise * torch.randn_like(z)
            return z.to(device=device, dtype=dtype)
        return self.policy.sample_prior(B, device=device, dtype=dtype)

    @staticmethod
    def _slice_ctx(ctx, s, e, K):
        """Repeat per-sample context entries to line up with the K-fold orbit grid."""
        out = {}
        for k, v in ctx.items():
            if torch.is_tensor(v) and v.ndim >= 1:
                out[k] = v.repeat_interleave(K, dim=0)[s:e]
            else:
                out[k] = v
        return out


# --------------------------------------------------------------------------------------
# Regularisers
# --------------------------------------------------------------------------------------


def typicality_penalty(z: torch.Tensor, spec: SO2ChunkSpec) -> torch.Tensor:
    """Per-irrep-group Gaussian typical-set penalty.  (B, H, A) -> (B,).

    The DNO review's section 8.1 is exactly right that ``R(z) = ||z||^2`` is the wrong
    regulariser: it drags the noise toward the origin, while a d-dimensional standard
    Gaussian keeps essentially all of its mass in a thin shell at ``||z|| ~ sqrt(d)``.
    The right penalty is the negative log density of the norm,

        -log p_chi_d(r) = -(d - 1) log r + r^2 / 2 + const,

    shifted so its minimum is 0 and divided by ``d`` so that groups of different sizes
    contribute comparably.

    And it is applied **per irrep group** -- once over all frequency-1 coordinates, once
    over all frequency-0 coordinates -- rather than as one global norm, because those two
    groups have different roles and one global constraint lets the optimizer trade the
    scale of one against the other.
    """
    groups = []
    vec_idx = list(spec.vector_index)
    if vec_idx:
        groups.append(z[..., vec_idx].reshape(z.shape[0], -1))
    if spec.scalar_dims:
        groups.append(z[..., list(spec.scalar_dims)].reshape(z.shape[0], -1))

    total = z.new_zeros(z.shape[0])
    for g in groups:
        d = g.shape[-1]
        if d < 2:
            continue
        r = g.norm(dim=-1).clamp_min(1e-6)
        f = -(d - 1) * torch.log(r) + 0.5 * r ** 2
        r_star = math.sqrt(d - 1)
        f_min = -(d - 1) * math.log(r_star) + 0.5 * r_star ** 2
        total = total + (f - f_min) / d
    return total


def project_to_ball(z: torch.Tensor, center: torch.Tensor, radius: float) -> torch.Tensor:
    """Hard trust region: project ``z`` into the ball of given radius around ``center``."""
    delta = (z - center).reshape(z.shape[0], -1)
    n = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = torch.clamp(radius / n, max=1.0)
    return center + (delta * scale).reshape_as(z)


def shift_noise(z: torch.Tensor, n_executed: int) -> torch.Tensor:
    """MPC-style warm start: slide the solved noise along the horizon, refill the tail.

    After executing ``n_executed`` of the chunk's ``H`` steps, the next chunk's first
    ``H - n_executed`` steps cover the same wall-clock interval as the old chunk's tail,
    so reusing that part of the solution is the noise-space analogue of an MPC warm start
    (the ``Shift(z*) + eps`` of the DNO review, section 7.3).
    """
    H = z.shape[1]
    n = min(int(n_executed), H)
    out = torch.randn_like(z)
    if n < H:
        out[:, : H - n] = z[:, n:]
    return out
