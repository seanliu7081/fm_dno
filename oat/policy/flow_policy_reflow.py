"""
Rectified-flow *reflow* policy: same field, retrained on its own transport plan.

    baseline training   z ~ N(0,I) fresh every step;  x1 = a_demo
    reflow  training    z = the stored source that PRODUCED this target;  x1 = F(z | o)

``F`` is the frozen baseline checkpoint and ``F(z | o) = Euler_{N_gen}(F; z, o)``.  The pairs
are generated once by ``scripts/generate_reflow_pairs.py`` and served by
``oat.dataset.reflow_pairs.ReflowPairDataset``.

WHY THIS IS THE ARM WITH THE STRONGEST PRIOR
--------------------------------------------
Few-step Euler is bad exactly when the learned velocity field's trajectories are curved.
The data-side couplings in ``oat/symmetry/coupling.py`` straighten the *ideal* field for a
particular chosen coupling: ``angle`` buys one global phase per chunk (2.4% path-length
reduction on this dataset), ``euclidean`` minibatch OT buys a batch-approximate assignment
(8.1% at B=32, and minibatch OT weakens in the chunk's 112 dimensions).  Reflow instead
re-couples against **the map the network actually learned**, so it attacks whatever
curvature is there -- multimodality-induced and approximation-induced alike.  It is Liu et
al.'s rectified flow (arXiv:2209.03003), not this project's invention, and the write-up
must say so.  What it *is* here is the strongest member of the coupling family under test.

WHAT THIS CLASS ADDS, AND WHAT IT DELIBERATELY DOES NOT
-------------------------------------------------------
It subclasses ``OrbitFlowPolicy`` with ``coupling.mode='iid'`` asserted, so the SO(2)
normalizer repair, the checkpoint round-trip, ``sample_chunk`` / ``encode_obs`` and every
diagnostic hook come along unchanged and the arm stays comparable to the rest of the table.
Exactly two things are new:

  1. ``init_ckpt`` -- warm-start from the donor's weights.  Standard for reflow, and what
     makes a 300-epoch cap plausible instead of a from-scratch re-run.
  2. ``forward`` -- one line different from the parent: ``x0`` comes out of the batch
     instead of ``torch.randn_like``.

``predict_action`` is untouched, so inference still draws ``z ~ N(0, I)``: the source
marginal is standard Gaussian **by construction** here (``z`` is drawn iid at generation
time and never modified), which is why this arm cannot leak label information the way the
rotation-applying couplings did.

GATES (see PLAN_fewstep_coupling.md S3.2)
-----------------------------------------
R1  the workspace-rebuilt action normalizer must equal the one the pairs were generated
    under, and ``normalize(action_env)`` must reproduce the stored ``x1_norm``.  The pairs
    live in the generation-time normalized space; a drifted normalizer silently corrupts
    every one of them.  Deterministic by construction -- gated anyway.
R2  index alignment; enforced in ``ReflowPairDataset`` (fingerprint + length).
R3  ``init_ckpt`` really loaded; asserted here on the state-dict key sets, and visible in
    the loss curve (a from-scratch-looking curve means it did not).
"""

from __future__ import annotations

import pathlib
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

from oat.policy.flow_policy_orbit import OrbitFlowPolicy

# Keys a donor checkpoint is allowed to be missing / to carry extra without it meaning the
# warm start failed.  Empty today: ReflowFlowPolicy adds no parameters or buffers of its
# own, so an exact match is the expectation and anything else is a real mismatch.
_ALLOWED_STATE_DICT_EXTRAS: tuple = ()

_R1_SCALE_TOL = 1e-6
_R1_ROUNDTRIP_TOL = 1e-5
_R1_ROUNDTRIP_ROWS = 4096


class ReflowFlowPolicy(OrbitFlowPolicy):
    """``OrbitFlowPolicy`` trained on stored ``(z, F(z|o))`` pairs.

    Args:
        init_ckpt: baseline checkpoint whose (EMA) weights this run starts from.  ``None``
            trains from scratch -- supported, but it is not the configuration the plan
            budgets for.
        pairs_path: the pair directory (or list of them) the dataset is reading, used only
            to run Gate R1 at ``set_normalizer`` time.  Point it at the same value as
            ``task.policy.dataset.pairs_path``; ``None`` skips the gate and says so loudly.
        reflow_key: batch key holding the stored source.
        strict_init: if True (default), a missing/unexpected key outside the allowlist
            raises instead of warning.
    """

    def __init__(
        self,
        *args,
        init_ckpt: Optional[str] = None,
        pairs_path: Optional[Union[str, Sequence[str]]] = None,
        reflow_key: str = "reflow_z",
        strict_init: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.coupling.mode != "iid":
            raise ValueError(
                f"ReflowFlowPolicy requires coupling.mode='iid' (the source is the stored "
                f"z, so a data-side coupling has nothing to act on); got "
                f"'{self.coupling.mode}'. Set policy.coupling.mode=iid."
            )
        self.reflow_key = str(reflow_key)
        self.init_ckpt = init_ckpt
        self.strict_init = bool(strict_init)
        self.pairs_path: List[str] = (
            [] if pairs_path is None
            else [str(pairs_path)] if isinstance(pairs_path, (str, pathlib.Path))
            else [str(p) for p in pairs_path]
        )
        self._r1_checked = False
        self._warm_start_ok = init_ckpt is None

        if init_ckpt is None:
            print("  reflow    : init_ckpt=None -- training the reflow field FROM SCRATCH.\n"
                  "              That is not the configuration S3.3 budgets 300 epochs for.")
        elif not pathlib.Path(str(init_ckpt)).is_file():
            # Reached on two very different paths, so it cannot simply raise here:
            #   * training with a bad path  -> must be fatal; caught in set_normalizer,
            #     which only the training workspace calls, before the first gradient step;
            #   * BasePolicy.from_checkpoint on a finished reflow run -> the payload
            #     overwrites these weights a moment later, so the donor is irrelevant and
            #     a hard failure would make every reflow checkpoint unloadable once the
            #     donor moved.
            print(f"  !! reflow : init_ckpt not found: {init_ckpt}\n"
                  f"     Skipping the warm start. Harmless when loading a finished "
                  f"checkpoint; FATAL for training, and set_normalizer will say so.")
        else:
            self._warm_start(str(init_ckpt))

    def get_policy_name(self) -> str:
        # OrbitFlowPolicy prefixes 'orbit_'; keep the arm identifiable in wandb / logs.
        return "reflow_" + super().get_policy_name()

    # -- warm start (Gate R3) ------------------------------------------------------------

    def _warm_start(self, init_ckpt: str) -> None:
        """Load the donor's weights in place.

        ``BasePolicy.from_checkpoint`` returns the **EMA** policy when the donor was trained
        with EMA -- the same weights that produced the reported success rate, and the same
        weights ``generate_reflow_pairs.py`` integrated to make the targets.  Anything else
        would distil one model and initialise from another.

        The workspace deep-copies the model into the EMA track *after* instantiation, so
        the EMA starts from these weights too.
        """
        from oat.policy.base_policy import BasePolicy  # local: avoids an import cycle

        path = pathlib.Path(init_ckpt)
        if not path.is_file():
            raise FileNotFoundError(f"policy.init_ckpt does not exist: {path}")
        print(f"  reflow    : warm-starting from {path}")
        donor = BasePolicy.from_checkpoint(str(path))
        donor_sd = {k: v.detach().clone() for k, v in donor.state_dict().items()}
        del donor

        before = self._param_signature()
        missing, unexpected = self.load_state_dict(donor_sd, strict=False)
        after = self._param_signature()

        missing = [k for k in missing if k not in _ALLOWED_STATE_DICT_EXTRAS]
        unexpected = [k for k in unexpected if k not in _ALLOWED_STATE_DICT_EXTRAS]
        msg = (f"  reflow    : loaded {len(donor_sd)} donor tensors; "
               f"{len(missing)} missing, {len(unexpected)} unexpected")
        if missing or unexpected:
            detail = (f"{msg}\n    missing   : {missing[:12]}\n    unexpected: {unexpected[:12]}")
            if self.strict_init:
                raise RuntimeError(
                    detail + "\n  Gate R3: the warm start did not fully take. The donor and "
                    "this policy must have the same architecture -- check embed_dim / "
                    "n_layers / n_heads / shape_meta against the donor's config."
                )
            print("  !! " + detail)
        else:
            print(msg + "  (exact match)")

        if before == after:
            raise RuntimeError(
                "Gate R3: load_state_dict left every parameter bit-identical, which means "
                "the donor weights were not applied. Refusing to train a run that would "
                "silently be 'from scratch'."
            )
        self._donor_signature = after
        self._warm_start_ok = True

    def _param_signature(self) -> tuple:
        """Cheap fingerprint of the trainable weights; used only to prove the load landed."""
        with torch.no_grad():
            return tuple(
                round(float(p.detach().float().sum()), 6)
                for p in list(self.model.parameters())[:8]
            )

    # -- normalizer (Gate R1) ------------------------------------------------------------

    def set_normalizer(self, normalizer) -> None:
        """Parent's SO(2) repair, then check the pairs still mean what they meant.

        Called once by ``TrainPolicyWorkspace.run`` before the first optimizer step, so
        this is the "at training start" the plan asks for.
        """
        super().set_normalizer(normalizer)
        # Only the training workspace calls this, so it is the right place to turn a
        # skipped warm start into a hard failure without breaking checkpoint loading.
        if not self._warm_start_ok:
            raise RuntimeError(
                f"Gate R3: policy.init_ckpt={self.init_ckpt!r} could not be loaded, so this "
                f"run would train from scratch under a config that says otherwise. Fix the "
                f"path, or set policy.init_ckpt=null if from-scratch is really intended."
            )
        if self._r1_checked:
            return
        if not self.pairs_path:
            print("  !! Gate R1 SKIPPED: policy.pairs_path is unset, so the pairs' "
                  "generation-time normalizer cannot be compared against this one. Set it "
                  "to ${task.policy.dataset.pairs_path}.")
            return
        self.verify_pairs_normalizer(self.pairs_path)
        self._r1_checked = True

    @torch.no_grad()
    def verify_pairs_normalizer(
        self,
        pairs_path: Union[str, Sequence[str]],
        splits: Iterable[str] = ("train", "val"),
    ) -> Dict[str, float]:
        """Gate R1.  Raises on any drift; returns the measured residuals.

        Two independent checks, because they fail differently:

          * the stored ``scale`` / ``offset`` / ``input_stats`` against the ones this policy
            just built -- catches a changed dataset or a changed normalizer_mode;
          * ``normalize(action_env) == x1_norm`` on real rows -- catches everything else,
            including a repair that agrees on paper but not in fp32.
        """
        from oat.dataset.reflow_pairs import ReflowPairSet  # local: keeps the import cheap

        paths = ([pairs_path] if isinstance(pairs_path, (str, pathlib.Path))
                 else list(pairs_path))
        params = self.normalizer.params_dict["action"]
        mine = {
            "scale": params["scale"].detach().float().cpu().numpy(),
            "offset": params["offset"].detach().float().cpu().numpy(),
            **{f"input_stats.{k}": v.detach().float().cpu().numpy()
               for k, v in params["input_stats"].items()},
        }

        worst = {"param_absmax": 0.0, "roundtrip_absmax": 0.0}
        for p in paths:
            for split in splits:
                ps = ReflowPairSet(p, split)
                snap = ps.normalizer_snapshot()
                for k, v in mine.items():
                    key = k.replace("input_stats.", "stats_")
                    if key not in snap:
                        raise RuntimeError(
                            f"Gate R1: pair file {ps.path} has no stored normalizer entry "
                            f"'{key}'. It predates this format -- regenerate the pairs."
                        )
                    d = float(np.abs(np.asarray(snap[key]) - v).max())
                    worst["param_absmax"] = max(worst["param_absmax"], d)
                    if d > _R1_SCALE_TOL:
                        raise RuntimeError(
                            f"Gate R1 FAILED: action normalizer '{k}' differs from the one "
                            f"the pairs were generated under by {d:.3e} (> {_R1_SCALE_TOL:g}).\n"
                            f"  pairs  : {ps.path}\n"
                            f"  stored : {np.asarray(snap[key])}\n"
                            f"  now    : {v}\n"
                            f"  The pairs live in the generation-time normalized space; "
                            f"training against a drifted normalizer corrupts every pair."
                        )

                n = min(_R1_ROUNDTRIP_ROWS, len(ps))
                a = torch.from_numpy(ps.action_env[:n]).to(params["scale"].device)
                got = self.normalizer["action"].normalize(a).float().cpu().numpy()
                d = float(np.abs(got - ps.x1_norm[:n]).max())
                worst["roundtrip_absmax"] = max(worst["roundtrip_absmax"], d)
                if d > _R1_ROUNDTRIP_TOL:
                    raise RuntimeError(
                        f"Gate R1 FAILED: normalize(action_env) departs from the stored "
                        f"x1_norm by {d:.3e} (> {_R1_ROUNDTRIP_TOL:g}) on {n} rows of "
                        f"{ps.path}."
                    )
                del ps

        print(f"  [R1] pairs/normalizer agree: params <= {worst['param_absmax']:.2e}, "
              f"normalize(action_env) vs x1_norm <= {worst['roundtrip_absmax']:.2e}")
        return worst

    # -- training ------------------------------------------------------------------------

    def forward(self, batch) -> torch.Tensor:
        """The parent's rectified-flow loss with the source read from the batch.

        Every other line is ``OrbitFlowPolicy.forward`` verbatim, so a difference between
        this arm and the baseline is the coupling and nothing else.
        """
        if self.reflow_key not in batch:
            raise KeyError(
                f"batch has no '{self.reflow_key}'. ReflowFlowPolicy must be trained on "
                f"oat.dataset.reflow_pairs.ReflowPairDataset -- a plain ZarrDataset would "
                f"silently train the baseline objective on distilled targets."
            )
        x1 = self.normalizer["action"].normalize(batch["action"])        # (B, H, A)
        B = x1.shape[0]
        device = x1.device

        cond = self.obs_encoder(batch["obs"])                             # (B, To, d)

        x0 = batch[self.reflow_key].to(device=device, dtype=x1.dtype)     # the STORED source
        if x0.shape != x1.shape:
            raise ValueError(
                f"reflow source shape {tuple(x0.shape)} != target shape {tuple(x1.shape)}"
            )

        t = torch.rand(B, device=device, dtype=x1.dtype)
        t_b = t[:, None, None]
        xt = (1.0 - t_b) * x0 + t_b * x1
        v_target = x1 - x0

        v_pred = self.model(xt, self._scale_t(t), cond)
        return F.mse_loss(v_pred, v_target)
