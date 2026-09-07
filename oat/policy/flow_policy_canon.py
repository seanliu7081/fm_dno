"""
Construction A: the orbit coupling made correct -- observation-canonicalized source phase.

THE ADMISSIBILITY CRITERION, WHICH IS THE WHOLE POINT
-----------------------------------------------------
Flow matching transports the *train-time conditional source law* ``q_o(z) = p_train(z | o)``
onto ``p(x1 | o)``.  At inference we integrate from whatever we sample, ``s_o(z)``.  The
endpoint is right **iff ``s_o = q_o``**.  So a coupling is admissible iff the transformation
it applies to the source is a function of ``(o, fresh independent noise)`` only -- something
the deployed policy can re-run verbatim at test time.

P4 set ``theta(z) := theta(x1)`` -- a statistic of *the answer*.  ``q_o`` therefore depended
on the target, inference could not reproduce it, and the arm scored 0.008 on all ten tasks.
P5 tried to repair it by matching the source's heading *marginal*, but ``q_o`` is a
*conditional*; no fixed marginal can equal it, and P5 scored 0.000.  Both are derivable from
the criterion rather than surprises.

This class sets ``theta(z) := theta_ref(o)`` instead -- a quantity the policy computes at test
time from the observation it already receives -- and applies the identical transformation on
both paths.  ``q_o = s_o`` **exactly**, by construction.  What P4 bought with leakage, this
buys legitimately::

    z ~ N(0, I)
    z <- rho( theta_ref(o) - theta(z) ) . z          # both in forward() and predict_action()

WHY IT SHOULD DO ANYTHING (AND THE HONEST COUNTER)
--------------------------------------------------
Measured on LIBERO-10 (Gate M1, ``scripts/m1_residual_frame_audit.py``): the chunk heading
has absolute concentration ``R1 = 0.123``, but its residual against the recent end-effector
motion direction concentrates at ``R1 = 0.593`` -- and consistently, 0.54-0.69 on every one
of the ten tasks.  So the source can be handed to the network already pointing approximately
where the chunk goes, and its phase task shrinks from "regress the heading through the vision
stack" to "correct a small residual".

The objection with teeth: ``theta_ref(o)`` is computable from ``o``, so the network could
learn it anyway; canonicalization only changes *how hard that function is to represent*.
That is an optimization/representation bet, not a theorem -- the same class of bet as
warm-start and residual parameterizations.  It must be sold as such.  The counterweight is
that P4 demonstrated this network *prefers* to read phase off the source when offered the
channel (steering gain 1.04, learned eagerly); A offers the same easy channel with content
that is still valid at test time.

THE REFERENCE FRAME
-------------------
``eef_motion`` (default): ``theta_ref(o) = atan2(d_eef_y, d_eef_x)`` over the ``n_obs_steps``
the policy receives -- no learning, nothing to ship, nothing to keep in sync.  Below
``min_ref_speed`` of planar motion the direction is meaningless, so canonicalization is
skipped for that sample; the rule reads only ``o``, so skipping is admissible too.

M1 also measured a state-only MLP frame at ``R1 = 0.821`` on held-out episodes, so a learned
``theta_ref`` head has substantial headroom over the cheap frame.  It stays out of this class
on purpose: it would have to be fitted offline, frozen, and shipped inside the checkpoint,
and any drift between the training-time and inference-time head silently violates the
criterion this whole construction exists to satisfy.  Do the cheap frame first; if it moves
SR, the head is the obvious follow-up.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from oat.policy.flow_policy_orbit import OrbitFlowPolicy
from oat.symmetry.so2_chunk import chunk_heading, rotate_chunk, wrap_angle

_VALID_FRAMES = ("eef_motion",)


class CanonicalPhaseFlowPolicy(OrbitFlowPolicy):
    """``OrbitFlowPolicy`` whose source phase is canonicalized to an observation-only frame.

    Args:
        ref_frame: which observation-only heading to canonicalize onto.
        ref_key: observation key holding the end-effector position, ``(B, To, 3)``.
        min_ref_speed: planar motion (metres) over the observation window below which
            ``theta_ref`` carries no direction and the sample is left alone.  Applied
            identically on both paths.
        kappa: von Mises concentration of a dither around ``theta_ref``.  ``inf`` is exact
            canonicalization; finite values interpolate back toward an isotropic source and
            are the knob for trading the representation shortcut against source diversity.
            The dither is *fresh independent noise*, so it does not break admissibility.
        canon_prob: fraction of samples that receive the canonicalization; the rest stay
            isotropic.  Also admissible (independent noise), also applied on both paths.
    """

    def __init__(
        self,
        *args,
        ref_frame: str = "eef_motion",
        ref_key: str = "robot0_eef_pos",
        min_ref_speed: float = 1e-4,
        kappa: float = float("inf"),
        canon_prob: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if ref_frame not in _VALID_FRAMES:
            raise ValueError(f"ref_frame must be one of {_VALID_FRAMES}, got {ref_frame}")
        if self.coupling.mode != "iid":
            raise ValueError(
                f"CanonicalPhaseFlowPolicy requires coupling.mode='iid'. The canonicalization "
                f"IS the coupling here, and it is admissible precisely because it never looks "
                f"at x1; stacking a target-side coupling on top would destroy that. Got "
                f"'{self.coupling.mode}'."
            )
        if self.normalizer_mode != "so2_block":
            raise ValueError(
                "CanonicalPhaseFlowPolicy requires normalizer_mode='so2_block'. theta_ref is "
                "a WORLD-frame direction and the source phase lives in normalized space; only "
                "the SO(2) repair (one shared scale per vector block, zero offset) makes those "
                "the same angle. Under per-dimension min-max the frame is sheared by ~0.24 "
                "relative error and the canonicalization points somewhere else entirely."
            )
        if isinstance(kappa, str):
            kappa = float(kappa)
        self.ref_frame = ref_frame
        self.ref_key = str(ref_key)
        self.min_ref_speed = float(min_ref_speed)
        self.kappa = float(kappa if kappa is not None else float("inf"))
        self.canon_prob = float(canon_prob)
        if math.isnan(self.kappa) or self.kappa < 0:
            raise ValueError("kappa must be non-negative (or inf for exact alignment)")
        if not 0.0 <= self.canon_prob <= 1.0:
            raise ValueError("canon_prob must be in [0, 1]")
        self.last_canon_diagnostics: Dict[str, float] = {}
        print(f"  canon     : frame={self._reference_description()} "
              f"kappa={self.kappa} canon_prob={self.canon_prob}\n")

    def _reference_description(self):
        return f"{self.ref_frame}({self.ref_key}) min_speed={self.min_ref_speed:g}"

    def get_policy_name(self) -> str:
        return "canon_" + super().get_policy_name()

    # -- the frame ---------------------------------------------------------------------

    def reference_heading(self, obs_dict: Dict[str, torch.Tensor]):
        """``(theta_ref, valid)`` from the observation alone.  Identical on both paths.

        ``theta_ref`` is the direction of planar end-effector travel across the observation
        window -- the last obs step minus the first.  ``valid`` gates out samples whose
        planar motion is below ``min_ref_speed``, where the direction is numerical noise.
        """
        if self.ref_key not in obs_dict:
            raise KeyError(
                f"CanonicalPhaseFlowPolicy needs obs['{self.ref_key}'] to build its reference "
                f"frame; the batch has {sorted(obs_dict)}."
            )
        eef = obs_dict[self.ref_key]
        if eef.ndim != 3 or eef.shape[-1] < 2:
            raise ValueError(f"expected obs['{self.ref_key}'] as (B, To, >=2), "
                             f"got {tuple(eef.shape)}")
        d = eef[:, -1, :2].float() - eef[:, 0, :2].float()          # (B, 2)
        speed = d.norm(dim=-1)
        theta = torch.atan2(d[:, 1], d[:, 0])
        return theta, speed > self.min_ref_speed

    def _prepare_observation(self, obs_dict):
        """Encode observations and obtain the frame once per policy call."""
        cond = self.encode_obs(obs_dict)
        theta, valid = self.reference_heading(obs_dict)
        return cond, theta, valid

    # -- the canonicalization ------------------------------------------------------------

    def canonicalize_source(
        self,
        z: torch.Tensor,
        obs_dict: Dict[str, torch.Tensor],
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Rotate each source so its chunk heading equals ``theta_ref(o)``.

        This and both policy paths use ``_canonicalize_with_heading``. The policy
        paths can reuse a precomputed reference without evaluating a visual encoder
        twice. Nothing in the source transformation reads ``x1``.
        """
        theta_ref, valid = self.reference_heading(obs_dict)
        return self._canonicalize_with_heading(z, theta_ref, valid, generator=generator)

    def _canonicalize_with_heading(self, z, theta_ref, valid, generator=None):
        """Shared source map, also usable with a precomputed learned reference."""
        if theta_ref.shape != (z.shape[0],) or valid.shape != (z.shape[0],):
            raise ValueError("reference heading and validity must match the source batch")
        theta_ref = theta_ref.to(device=z.device)
        valid = valid.to(device=z.device)

        theta_z, _ = chunk_heading(z.float(), self.action_spec)
        phi = wrap_angle(theta_ref - theta_z.to(theta_ref.dtype))

        if self.kappa == 0:
            # VonMises requires positive concentration; zero is the uniform limit.
            phi = phi + 2 * math.pi * torch.rand_like(phi) - math.pi
        elif math.isfinite(self.kappa):
            dither = torch.distributions.VonMises(
                loc=torch.zeros_like(phi),
                concentration=torch.full_like(phi, self.kappa),
            ).sample()
            phi = phi + dither
        if self.canon_prob < 1.0:
            # global RNG on purpose: `generator` may live on another device, and this draw is
            # fresh independent noise either way, so it costs nothing in admissibility
            valid = valid & (torch.rand(phi.shape[0], device=z.device) < self.canon_prob)

        phi = torch.where(valid, phi, torch.zeros_like(phi))
        out = rotate_chunk(z, phi.to(z.dtype), self.action_spec)

        with torch.no_grad():
            self.last_canon_diagnostics = {
                "canon/valid_frac": float(valid.float().mean()),
                "canon/abs_phi_mean": float(phi.abs().mean()),
                # a permutation-free rotation cannot change any norm; assert it cheaply
                "canon/norm_drift": float(
                    (out.reshape(out.shape[0], -1).norm(dim=-1)
                     - z.reshape(z.shape[0], -1).norm(dim=-1)).abs().max()
                ),
            }
        return out

    def sample_prior_from_obs(
        self,
        obs_dict: Dict[str, torch.Tensor],
        batch_size: Optional[int] = None,
        device=None,
        dtype=None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """The inference source law ``s_o``.  Diagnostics should drive the field from this.

        ``sample_prior`` (obs-free, inherited) returns the *un*-canonicalized isotropic draw,
        which for this arm is NOT the law the field was trained on.  Anything that integrates
        from it is measuring the field off its own training distribution -- correct for an
        off-manifold probe, wrong for a like-for-like comparison against another arm.
        """
        theta, valid = self.reference_heading(obs_dict)
        return self._sample_prior_with_heading(
            theta, valid, batch_size, device, dtype, generator)

    def _sample_prior_with_heading(
        self, theta, valid, batch_size=None, device=None, dtype=None, generator=None,
    ):
        B = theta.shape[0] if batch_size is None else batch_size
        if B != theta.shape[0]:
            raise ValueError("batch_size must match the observation batch")
        z = self.prior_noise_scale * torch.randn(
            B, self.horizon, self.action_dim,
            device=device or theta.device, dtype=dtype or torch.float32, generator=generator,
        )
        return self._canonicalize_with_heading(z, theta, valid, generator=generator)

    # -- training ------------------------------------------------------------------------

    def forward(self, batch) -> torch.Tensor:
        """The parent's loss with the source canonicalized instead of coupled to the target."""
        x1 = self.normalizer["action"].normalize(batch["action"])        # (B, H, A)
        B = x1.shape[0]
        device = x1.device

        cond, theta, valid = self._prepare_observation(batch["obs"])

        z0 = self.prior_noise_scale * torch.randn_like(x1)
        z0 = self._canonicalize_with_heading(z0, theta, valid).to(x1.dtype)

        t = torch.rand(B, device=device, dtype=x1.dtype)
        t_b = t[:, None, None]
        xt = (1.0 - t_b) * z0 + t_b * x1
        v_target = x1 - z0

        v_pred = self.model(xt, self._scale_t(t), cond)
        return F.mse_loss(v_pred, v_target)

    # -- inference -----------------------------------------------------------------------

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Same source law as training -- that identity is the whole construction."""
        cond, theta, valid = self._prepare_observation(obs_dict)
        z = self._sample_prior_with_heading(
            theta, valid, batch_size=cond.shape[0], device=cond.device, dtype=cond.dtype)
        x = self.sample_chunk(cond, z.to(cond.dtype))
        action_pred = self.normalizer["action"].unnormalize(x)
        return {"action": action_pred[:, : self.n_action_steps], "action_pred": action_pred}

    def predict_action_from_noise(
        self,
        obs_dict: Dict[str, torch.Tensor],
        z: torch.Tensor,
        n_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """``z`` is canonicalized before integration, so DNO searches the trained source law."""
        cond, theta, valid = self._prepare_observation(obs_dict)
        z = self._canonicalize_with_heading(
            z.to(device=cond.device, dtype=cond.dtype), theta, valid)
        x = self.sample_chunk(cond, z, n_steps=n_steps)
        action_pred = self.normalizer["action"].unnormalize(x)
        return {"action": action_pred[:, : self.n_action_steps], "action_pred": action_pred}
