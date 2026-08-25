"""
Flow-matching policy with an SO(2) orbit-aligned source-target coupling.

``OrbitFlowPolicy`` is a drop-in subclass of ``oat.policy.flow_policy.FlowPolicy``.  In the
default configuration exactly one thing changes: the source ``z0`` is coupled to the target
chunk before the rectified-flow interpolation.  Architecture, loss, observation encoder,
optimizer, and -- critically -- ``predict_action`` behave exactly as the parent's, so
inference still draws ``z0 ~ N(0, I)`` and the comparison against the baseline is controlled
the same way the toy study's A-vs-C comparison was.

Two opt-in departures, both off by default and both gated by an explicit flag:
``normalizer_mode='so2_block'`` repairs the action normalizer inside ``set_normalizer`` (on
by default, because without it nothing geometric downstream is meaningful -- see that
method), and ``inference_prior='matched'`` redraws the source from the coupling's own
trained heading law (off by default; it is what the P5 arm turns on).

    training   z0 ~ N(0, I);  z0 <- Pi(z0, x1)          <-- the only change
               t ~ U(0,1);  xt = (1-t) z0 + t x1;  u = x1 - z0
               loss = MSE(v_theta(xt, t, cond), u)

    inference  z ~ N(0, I);  Euler(v_theta, N steps)     <-- identical to the baseline

The extra methods (``encode_obs``, ``sample_chunk``, ``velocity``) exist so the DNO
module and the equivariance metrics can drive the sampler with a supplied noise tensor
and keep gradients.  They are additive; nothing in the training path depends on them.

A note on minibatch couplings under conditioning: the assignment is computed within a
batch whose elements have *different* observations, so a source may be handed to a target
from another condition.  Averaged over batches this is still a valid coupling with the
right marginals (Pooladian et al., "Multisample Flow Matching", ICML 2023,
arXiv:2304.14772); what it introduces is the usual minibatch-OT bias, which shrinks with
batch size while the coupling itself strengthens.  ``coupling.coupling_prob`` and
``kappa`` are the knobs for trading those off; treat batch size as a coupling
hyper-parameter, not a free training detail.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from oat.policy.flow_policy import FlowPolicy
from oat.symmetry.coupling import SO2OrbitCoupling, build_coupling
from oat.symmetry.so2_chunk import SO2ChunkSpec, chunk_heading, rotate_chunk, wrap_angle


class OrbitFlowPolicy(FlowPolicy):
    """FlowPolicy + symmetry-aware coupling.

    Args:
        coupling: dict forwarded to ``SO2OrbitCoupling`` (``mode``, ``cost``, ``kappa``,
            ``scalar_weight``, ``coupling_prob``, ``min_heading_confidence``,
            ``assignment``).  ``None`` or ``mode='iid'`` reproduces the baseline exactly.
        action_spec: dict forwarded to ``SO2ChunkSpec``.  Defaults to the LIBERO /
            robosuite ``OSC_POSE`` layout.
    """

    def __init__(
        self,
        *args,
        coupling: Optional[Dict] = None,
        action_spec: Optional[Dict] = None,
        normalizer_mode: str = "so2_block",
        normalizer_vector_mode: str = "rms",
        inference_prior: str = "standard",
        heading_hist_bins: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        spec = SO2ChunkSpec(**action_spec) if action_spec else SO2ChunkSpec.libero_osc_pose()
        if spec.action_dim != self.action_dim:
            raise ValueError(
                f"action_spec.action_dim={spec.action_dim} does not match the task's "
                f"action_dim={self.action_dim}"
            )
        if normalizer_mode not in ("so2_block", "inherit"):
            raise ValueError("normalizer_mode must be 'so2_block' or 'inherit'")
        if inference_prior not in ("standard", "matched"):
            raise ValueError("inference_prior must be 'standard' or 'matched'")
        self.action_spec = spec
        self.normalizer_mode = normalizer_mode
        self.normalizer_vector_mode = normalizer_vector_mode
        self.inference_prior = inference_prior
        self.coupling: SO2OrbitCoupling = build_coupling(coupling, spec)
        self.last_coupling_diagnostics: Dict[str, float] = {}

        # Running circular histogram of the source headings the model is ACTUALLY trained
        # on.  Registered as a buffer so it rides along in the checkpoint, which turns
        # P5 into "the same P4 checkpoint evaluated with inference_prior='matched'"
        # rather than a separate training run -- a perfectly controlled comparison.
        # Measured on the actual coupled sources, so it is exact for every coupling
        # variant, not only align='heading' with kappa=inf.
        self.register_buffer(
            "source_heading_hist", torch.zeros(int(heading_hist_bins)), persistent=True
        )
        print(f"  coupling  : {self.coupling.extra_repr()}")
        print(f"  normalizer: {normalizer_mode} ({normalizer_vector_mode})")
        print(f"  inf prior : {inference_prior}\n")

    def get_policy_name(self) -> str:
        return "orbit_" + super().get_policy_name()

    # -- normalizer ------------------------------------------------------------------

    def set_normalizer(self, normalizer) -> None:
        """Load the dataset normalizer, then make the *action* map commute with rho(phi).

        Doing the repair here rather than in the workspace is deliberate: it keeps
        ``TrainPolicyWorkspace`` untouched, so ``run_workspace.py``, ``eval_policy_sim.py``
        and ``BasePolicy.from_checkpoint`` all keep working with no edits anywhere.

        Per-dimension min-max gives ``scale[dx] != scale[dy]`` and a nonzero offset, so a
        rotation in normalized space is not a rotation in raw space (measured relative error
        0.24).  Rebuilding the action entry with one shared scale per frequency-1 block and a
        zero offset there brings it to ~1e-7.  Observation normalizers are left alone --
        with fixed RGB cameras the observation carries no group action to preserve.

        Idempotent: the fit is recomputed from ``input_stats``, which is never modified.
        Checkpoints store the repaired ``scale`` / ``offset`` directly, so loading a
        checkpoint reproduces them without calling this again.
        """
        super().set_normalizer(normalizer)
        if self.normalizer_mode != "so2_block":
            return

        from oat.symmetry.normalizer import so2_scale_offset_from_stats

        params = self.normalizer.params_dict["action"]
        stats = {k: v.detach().cpu() for k, v in params["input_stats"].items()}
        scale, offset = so2_scale_offset_from_stats(
            stats, self.action_spec, vector_mode=self.normalizer_vector_mode
        )
        ref = params["scale"]
        with torch.no_grad():
            params["scale"].data = scale.to(device=ref.device, dtype=ref.dtype)
            params["offset"].data = offset.to(device=ref.device, dtype=ref.dtype)
            params["scale"].requires_grad_(False)
            params["offset"].requires_grad_(False)
        blocks = ", ".join(
            f"({i},{j})->{float(scale[i]):.4f}" for i, j in self.action_spec.vector_blocks
        )
        print(f"  [SO2] action normalizer repaired: {blocks}, vector offsets 0")

    # -- training --------------------------------------------------------------------

    def forward(self, batch) -> torch.Tensor:
        x1 = self.normalizer["action"].normalize(batch["action"])       # (B, H, A)
        B = x1.shape[0]
        device = x1.device

        cond = self.obs_encoder(batch["obs"])                            # (B, To, d)

        z0 = self.prior_noise_scale * torch.randn_like(x1)
        z0, diag = self.coupling(z0, x1)
        self.last_coupling_diagnostics = diag
        self._accumulate_source_headings(z0)

        t = torch.rand(B, device=device, dtype=x1.dtype)
        t_b = t[:, None, None]
        xt = (1.0 - t_b) * z0 + t_b * x1
        v_target = x1 - z0

        v_pred = self.model(xt, self._scale_t(t), cond)
        return F.mse_loss(v_pred, v_target)

    # -- additive inference hooks (used by DNO and by the metrics) --------------------

    def encode_obs(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """(B, To, d) conditioning tokens.  Encode once, reuse across DNO iterations."""
        return self.obs_encoder(obs_dict)

    def velocity(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Raw velocity field, ``t`` in [0, 1].  Keeps gradients."""
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        return self.model(x, self._scale_t(t), cond)

    def sample_chunk(
        self,
        cond: torch.Tensor,
        z: torch.Tensor,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Euler integration from a *supplied* source, differentiable w.r.t. ``z``.

        Numerically identical to the inherited ``predict_action`` sampler; the only
        difference is that the noise comes in as an argument and no ``no_grad`` is
        applied, so DNO can backpropagate through the whole chain.
        """
        n_steps = n_steps or self.num_inference_steps
        dt = 1.0 / n_steps
        x = z
        for i in range(n_steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device, dtype=x.dtype)
            x = x + dt * self.model(x, self._scale_t(t), cond)
        return x

    def predict_action_from_noise(
        self,
        obs_dict: Dict[str, torch.Tensor],
        z: torch.Tensor,
        n_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """``predict_action`` with the source noise supplied instead of sampled."""
        cond = self.obs_encoder(obs_dict)
        x = self.sample_chunk(cond, z.to(dtype=cond.dtype), n_steps=n_steps)
        action_pred = self.normalizer["action"].unnormalize(x)
        return {"action": action_pred[:, : self.n_action_steps], "action_pred": action_pred}

    def sample_prior(
        self, batch_size: int, device=None, dtype=None, generator=None
    ) -> torch.Tensor:
        """The inference prior.

        ``inference_prior='standard'`` is plain N(0, I) -- identical to the baseline, which
        is what makes the A-vs-C style comparison controlled.  ``'matched'`` re-imposes the
        heading law the coupling actually produced during training, removing the prior shift
        of design-doc S5 at the cost of a non-isotropic (but correct) source.
        """
        z = self.prior_noise_scale * torch.randn(
            batch_size,
            self.horizon,
            self.action_dim,
            device=device or self.device,
            dtype=dtype or torch.float32,
            generator=generator,
        )
        return self.apply_inference_prior(z)

    # -- matched inference prior --------------------------------------------------------

    @torch.no_grad()
    def _accumulate_source_headings(self, z0: torch.Tensor) -> None:
        # Only a rotation-APPLYING coupling induces a source heading law worth matching.
        # Accumulating for every mode made an iid checkpoint carry a populated histogram,
        # which defeated both the EXPERIMENT_PLAN S6 assertion and the --inference-prior
        # guard in eval_orbit_dno_libero.py: neither could tell P5 was being run on an arm
        # that never deformed its prior. For iid/perm the induced law IS the isotropic one,
        # so an empty histogram is the honest record.
        if self.coupling.mode not in ("rot", "perm_rot"):
            return
        theta, _ = chunk_heading(z0.detach().float(), self.action_spec)
        theta = torch.remainder(theta, 2 * math.pi)
        n_bins = self.source_heading_hist.numel()
        h = torch.histc(theta, bins=n_bins, min=0.0, max=2 * math.pi)
        self.source_heading_hist += h.to(self.source_heading_hist.device)

    @torch.no_grad()
    def apply_inference_prior(self, z: torch.Tensor) -> torch.Tensor:
        """Re-impose the trained source heading law on an isotropic sample (no-op if standard)."""
        if self.inference_prior != "matched":
            return z
        counts = self.source_heading_hist
        if float(counts.sum()) <= 0:
            raise RuntimeError(
                "inference_prior='matched' but source_heading_hist is empty. The histogram "
                "is filled during training; a checkpoint trained before this field existed, "
                "or with mode='iid', has nothing to match."
            )
        n_bins = counts.numel()
        counts = counts + 1e-3 * counts.mean()
        cdf = torch.cumsum(counts, 0)
        cdf = (cdf / cdf[-1]).to(z.device)
        edges = torch.linspace(0, 2 * math.pi, n_bins + 1, device=z.device, dtype=z.dtype)
        u = torch.rand(z.shape[0], device=z.device, dtype=cdf.dtype)
        idx = torch.searchsorted(cdf, u.clamp(0, 1 - 1e-6)).clamp(0, n_bins - 1)
        width = edges[1] - edges[0]
        theta_star = edges[idx] + width * torch.rand(z.shape[0], device=z.device, dtype=z.dtype)
        theta_z, _ = chunk_heading(z, self.action_spec)
        return rotate_chunk(z, wrap_angle(theta_star - theta_z), self.action_spec)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inherited verbatim when ``inference_prior='standard'``; otherwise redraws the source.

        The branch matters for the experimental logic: with the default the sampler is
        byte-for-byte the parent's, so P1-P4 and P6 differ from the baseline only in training.
        Only P5 takes the other path, and it uses the same weights as P4.
        """
        if self.inference_prior == "standard":
            return super().predict_action(obs_dict)
        cond = self.obs_encoder(obs_dict)
        z = self.sample_prior(cond.shape[0], device=cond.device, dtype=cond.dtype)
        x = self.sample_chunk(cond, z)
        action_pred = self.normalizer["action"].unnormalize(x)
        return {"action": action_pred[:, : self.n_action_steps], "action_pred": action_pred}


# --------------------------------------------------------------------------------------
# Diffusion variant -- read the warning
# --------------------------------------------------------------------------------------


class OrbitDiffusionTransformerPolicy:
    """Guidance, not a class you should instantiate blindly.  See the design doc, S8.

    Flow matching admits *any* coupling with correct marginals: the FM objective is
    defined for an arbitrary pi(z0, x1) and the induced marginal path still connects
    p_0 to p_1.  That licence is what makes ``rot`` legitimate for ``OrbitFlowPolicy``.

    DDPM does **not** inherit that licence.  Its forward process requires the noise to be
    conditionally standard Gaussian *given the data*:

        q(x_t | x_0) = N(sqrt(abar_t) x_0, (1 - abar_t) I),   eps ~ N(0, I) _|_ x_0.

    Rotating ``eps`` so that its phase matches ``x_0`` makes ``p(eps | x_0)`` non-Gaussian,
    which changes the forward process the score is being fitted to, while sampling still
    starts from an isotropic N(0, I).  So ``mode='rot'`` and ``mode='perm_rot'`` are
    **not** valid for the repo's ``DiffusionTransformerPolicy`` / ``DiffusionUnetPolicy``.

    What is defensible:

      * ``mode='perm'`` -- a within-minibatch permutation.  The batch's noise marginal is
        exactly preserved; only the per-sample conditional changes.  This is precisely
        the construction and the (acknowledged, empirical rather than proved) argument of
        Immiscible Diffusion (Li et al., NeurIPS 2024, arXiv:2406.12303).  Inherit their
        caveat; do not re-derive it as if it were a theorem.
      * Reformulate as a bridge -- DDBM (Zhou et al., ICLR 2024, arXiv:2309.16948) or
        I2SB -- which is the rigorous way to give a diffusion model a nontrivial coupling.
      * Or simply use the flow policy, whose probability-flow ODE is the same object with
        none of this trouble.  That is why this project targets ``FlowPolicy`` first.

    To use the permitted variant, subclass ``DiffusionTransformerPolicy`` and replace the
    two lines of ``forward``::

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        noise, diag = self.coupling(noise, trajectory)     # mode must be 'perm'
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

    with ``SO2OrbitCoupling(spec, mode='perm', cost='group_ot')``, and assert the mode at
    construction time.
    """

    def __init__(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError(
            "Read this class's docstring: sample-modifying couplings are unsound for "
            "DDPM. Subclass DiffusionTransformerPolicy with mode='perm' instead."
        )
