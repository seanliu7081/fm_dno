"""
Dataset wrapper that swaps the demonstration target for a reflow (rectified-flow) pair.

WHAT THIS IS FOR
----------------
``PLAN_fewstep_coupling.md`` Arm R.  A trained flow policy ``F`` defines a transport plan
of its own: ``z -> Euler_N(F; z, o)``.  Retraining the same architecture on *that* plan --
``(z, x_hat_1 | o)`` instead of ``(z_fresh, a_demo | o)`` -- is the rectified-flow "reflow"
step of Liu et al. (arXiv:2209.03003).  It leaves the endpoints where the baseline's
N-step sampler put them while (in the ideal limit) straightening the paths, which is the
one property a 1- or 2-step Euler sampler actually needs.  This class is the data half of
that: same observations, same window indexing, different ``action`` and one extra key.

    base ZarrDataset item        {'obs': ..., 'action': a_demo}
    ReflowPairDataset item       {'obs': ..., 'action': x_hat_1_env, 'reflow_z': z}

``scripts/generate_reflow_pairs.py`` produces the pair files; it is the only thing that
should ever write them.

THE FAILURE MODE THIS CLASS EXISTS TO PREVENT
---------------------------------------------
The pairs are stored as one row per dataset index, so index ``i`` must mean *the same
window* at generation time and at training time.  Nothing in the file format enforces
that -- a different ``num_demo``, ``val_ratio``, ``seed``, ``horizon`` or even a rewritten
zarr silently renumbers every window, and the result is a training set of observations
paired with other windows' endpoints.  That does not crash and it does not look wrong in
the loss curve; it just quietly destroys the experiment.

So the pair file carries a **fingerprint** of the exact ``SequenceSampler`` index table it
was generated against (a SHA1 of the ``(buffer_start, buffer_end, sample_start,
sample_end)`` rows plus the dataset's construction parameters), and this class hard-fails
on any mismatch.  Gate R2 in the plan is this check; it is not optional and there is no
flag to turn it off.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Dict, List, Sequence, Union

import numpy as np
import torch

from oat.dataset.base_dataset import BaseDataset
from oat.dataset.zarr_dataset import ZarrDataset
from oat.model.common.normalizer import LinearNormalizer

_SPLITS = ("train", "val")
_ARRAY_KEYS = ("z", "x1_norm", "action_env")


# --------------------------------------------------------------------------------------
# Fingerprint -- shared with the generator so the two cannot drift apart
# --------------------------------------------------------------------------------------


def dataset_fingerprint(dataset: ZarrDataset, zarr_path: str) -> Dict[str, object]:
    """Everything that has to agree for index ``i`` to mean the same window.

    The SHA1 over the sampler's index table is the load-bearing part: it changes if the
    episode split changes, if the horizon changes, if the zarr contents change length, or
    if the padding rules change.  The scalar fields are there so that a mismatch report
    says *which* thing moved rather than just "hashes differ".
    """
    idx = np.ascontiguousarray(dataset.seq_sampler.indices.astype(np.int64))
    return {
        "zarr_path": str(zarr_path),
        "action_key": str(dataset.action_key),
        "n_obs_steps": int(dataset.n_obs_steps),
        "n_action_steps": int(dataset.n_action_steps),
        "seq_len": int(dataset.seq_len),
        "pad_before": int(dataset.pad_before),
        "pad_after": int(dataset.pad_after),
        "n_episodes": int(dataset.replay_buffer.n_episodes),
        "n_steps": int(dataset.replay_buffer[dataset.action_key].shape[0]),
        "n_episodes_in_split": int(np.sum(dataset.train_mask)),
        "n_windows": int(len(dataset.seq_sampler)),
        "obs_keys": [str(k) for k in dataset.obs_keys],
        "index_sha1": hashlib.sha1(idx.tobytes()).hexdigest(),
    }


def describe_mismatch(want: Dict[str, object], got: Dict[str, object]) -> str:
    keys = sorted(set(want) | set(got))
    lines = []
    for k in keys:
        a, b = want.get(k, "<absent>"), got.get(k, "<absent>")
        if a != b:
            lines.append(f"    {k}:  pairs={a!r}   dataset={b!r}")
    return "\n".join(lines) if lines else "    (no scalar field differs -- compare index_sha1)"


# --------------------------------------------------------------------------------------
# Pair file I/O
# --------------------------------------------------------------------------------------


def resolve_pair_file(path: Union[str, pathlib.Path], split: str) -> pathlib.Path:
    """Accept either a directory holding ``{train,val}.npz`` or one of those files."""
    if split not in _SPLITS:
        raise ValueError(f"split must be one of {_SPLITS}, got {split!r}")
    p = pathlib.Path(path)
    if p.is_dir():
        return p / f"{split}.npz"
    if p.suffix == ".npz":
        # a sibling file of the same pair set -- honour the requested split, not the
        # split whose filename happened to be typed into the config
        return p.parent / f"{split}.npz"
    return pathlib.Path(str(p) + f"/{split}.npz")


class ReflowPairSet:
    """One split of one generated pair file, kept memory-resident (~170 MB for LIBERO-10)."""

    def __init__(self, path: Union[str, pathlib.Path], split: str) -> None:
        self.path = resolve_pair_file(path, split)
        self.split = split
        if not self.path.is_file():
            raise FileNotFoundError(
                f"reflow pair file {self.path} not found. Generate it with\n"
                f"  python scripts/generate_reflow_pairs.py -c <checkpoint> "
                f"-o {self.path.parent}"
            )
        with np.load(self.path, allow_pickle=False) as f:
            self.meta: Dict = json.loads(str(f["meta_json"]))
            self.z = np.ascontiguousarray(f["z"])
            self.x1_norm = np.ascontiguousarray(f["x1_norm"])
            self.action_env = np.ascontiguousarray(f["action_env"])
        for k in _ARRAY_KEYS:
            arr = getattr(self, k)
            if arr.dtype != np.float32:
                raise ValueError(f"{self.path}: '{k}' must be float32, got {arr.dtype}")
        if not (self.z.shape == self.x1_norm.shape == self.action_env.shape):
            raise ValueError(
                f"{self.path}: array shapes disagree: z={self.z.shape} "
                f"x1_norm={self.x1_norm.shape} action_env={self.action_env.shape}"
            )
        if self.meta.get("split") != split:
            raise ValueError(
                f"{self.path}: file says split={self.meta.get('split')!r}, asked for {split!r}"
            )
        if self.meta.get("truncated"):
            raise RuntimeError(
                f"{self.path} was written with --limit and covers only "
                f"{self.meta.get('n_windows')} of the split's windows. It exists for "
                f"debugging the generator, not for training. Regenerate without --limit."
            )

    def __len__(self) -> int:
        return int(self.z.shape[0])

    @property
    def fingerprint(self) -> Dict[str, object]:
        return dict(self.meta["fingerprint"])

    def normalizer_snapshot(self) -> Dict[str, np.ndarray]:
        """The generating policy's action normalizer, as plain arrays (Gate R1)."""
        with np.load(self.path, allow_pickle=False) as f:
            return {k[5:]: np.asarray(f[k]) for k in f.files if k.startswith("norm_")}


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------


class ReflowPairDataset(BaseDataset):
    """``ZarrDataset`` with the demo action replaced by a stored reflow endpoint.

    Args:
        pairs_path: directory written by ``scripts/generate_reflow_pairs.py`` (it holds
            ``train.npz`` / ``val.npz``), or a list of such directories.  With a list the
            dataset length is the sum of the pair sets and the base index is ``i % len(base)``:
            that is how the plan's R4 retry adds a second pair-per-window set (a fresh
            ``z_seed``) without regenerating the first.
        **zarr_kwargs: forwarded verbatim to ``ZarrDataset``.  The task config already
            supplies these, so the only change a config needs is ``_target_`` + ``pairs_path``.
    """

    def __init__(
        self,
        pairs_path: Union[str, Sequence[str]],
        zarr_path: str,
        **zarr_kwargs,
    ) -> None:
        super().__init__()
        base = ZarrDataset(zarr_path=zarr_path, **zarr_kwargs)
        paths = [pairs_path] if isinstance(pairs_path, (str, pathlib.Path)) else list(pairs_path)
        if not paths:
            raise ValueError("pairs_path is empty")
        self._init(base=base, paths=[str(p) for p in paths], split="train",
                   zarr_path=str(zarr_path))

    # -- construction ------------------------------------------------------------------

    def _init(self, base: ZarrDataset, paths: List[str], split: str, zarr_path: str) -> None:
        self.base = base
        self.pairs_path = paths
        self.split = split
        self.zarr_path = zarr_path
        self.pair_sets = [ReflowPairSet(p, split) for p in paths]
        self._verify()
        self.base_len = len(self.base)
        # exposed so the workspace / policy can gate on them without re-reading the file
        self.reflow_meta = [ps.meta for ps in self.pair_sets]

    def _verify(self) -> None:
        """Gate R2.  A silent misalignment here poisons every pair, so it is a hard fail."""
        want = dataset_fingerprint(self.base, self.zarr_path)
        for ps in self.pair_sets:
            got = ps.fingerprint
            if got != want:
                raise RuntimeError(
                    f"reflow pair/dataset MISALIGNMENT for split={self.split!r}\n"
                    f"  pairs : {ps.path}\n"
                    f"  These were generated against a different dataset, so index i does "
                    f"not mean the same window.\n"
                    f"  differing fields:\n{describe_mismatch(got, want)}\n"
                    f"  Regenerate the pairs against this dataset, or point the config at "
                    f"the pair set that matches."
                )
            if len(ps) != len(self.base):
                raise RuntimeError(
                    f"reflow pair length {len(ps)} != dataset length {len(self.base)} "
                    f"for split={self.split!r} ({ps.path})"
                )
        horizon = self.base.n_action_steps
        for ps in self.pair_sets:
            if ps.z.shape[1] != horizon:
                raise RuntimeError(
                    f"{ps.path}: pair horizon {ps.z.shape[1]} != dataset horizon {horizon}"
                )

    @classmethod
    def _wrap(cls, base: ZarrDataset, paths: List[str], split: str,
              zarr_path: str) -> "ReflowPairDataset":
        obj = cls.__new__(cls)
        BaseDataset.__init__(obj)
        obj._init(base=base, paths=paths, split=split, zarr_path=zarr_path)
        return obj

    # -- BaseDataset interface ---------------------------------------------------------

    def get_validation_dataset(self) -> "ReflowPairDataset":
        # ``ZarrDataset.get_validation_dataset`` shallow-copies, so the 3.4 GB replay
        # buffer is shared rather than reloaded.
        return ReflowPairDataset._wrap(
            base=self.base.get_validation_dataset(),
            paths=self.pairs_path,
            split="val",
            zarr_path=self.zarr_path,
        )

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        """Forwarded to the base dataset.

        The base fits on the **full** replay buffer (train and val episodes alike), so this
        is identical to the baseline's normalizer by construction.  Gate R1 in
        ``ReflowFlowPolicy.set_normalizer`` checks it against the snapshot stored in the
        pair file anyway -- deterministic by construction is not the same as verified.
        """
        return self.base.get_normalizer(mode=mode, **kwargs)

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(np.concatenate([ps.action_env for ps in self.pair_sets], axis=0))

    def __len__(self) -> int:
        return self.base_len * len(self.pair_sets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        set_idx, base_idx = divmod(int(idx), self.base_len)
        ps = self.pair_sets[set_idx]
        item = self.base[base_idx]
        item["action"] = torch.from_numpy(ps.action_env[base_idx].copy())
        item["reflow_z"] = torch.from_numpy(ps.z[base_idx].copy())
        return item

    # -- diagnostics -------------------------------------------------------------------

    def summary(self) -> str:
        m = self.pair_sets[0].meta
        return (
            f"ReflowPairDataset(split={self.split}, windows={self.base_len}, "
            f"pair_sets={len(self.pair_sets)}, len={len(self)})\n"
            f"  donor      : {m.get('checkpoint')} (epoch {m.get('checkpoint_epoch')})\n"
            f"  N_gen      : {m.get('n_gen')}   z_seed: "
            f"{[ps.meta.get('z_seed') for ps in self.pair_sets]}\n"
            f"  index_sha1 : {m['fingerprint']['index_sha1'][:12]}"
        )
