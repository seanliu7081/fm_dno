#!/usr/bin/env python3
"""Export the checkpoint's selected policy into one self-contained inference file.

The original training checkpoint remains necessary for optimizer continuation.
Run final rollouts on this exported file so its SHA256 is the reported artifact.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import dill
import torch
from oat.common.hydra_util import register_new_resolvers
from oat.env_runner.libero_official_eval import file_sha256, load_policy


def export_checkpoint(source, destination):
    source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    register_new_resolvers()
    source_hash = file_sha256(source)
    payload = torch.load(source, map_location="cpu", pickle_module=dill)
    cfg = copy.deepcopy(payload["cfg"])
    selected = "ema_model" if cfg.training.get("use_ema", False) else "model"
    if selected not in payload["state_dicts"]:
        raise ValueError(f"Checkpoint does not contain its configured {selected}")
    # Avoid accidentally treating this inference artifact as a resumable run.
    cfg.training.resume = False
    provenance = {
        "format": "oat_inference_policy_v1", "inference_only": True,
        "source_checkpoint": str(source), "source_checkpoint_sha256": source_hash,
        "weights": selected,
    }
    exported = {
        "cfg": cfg,
        "state_dicts": {selected: payload["state_dicts"][selected]},
        "pickles": {key: value for key, value in payload.get("pickles", {}).items()
                    if key in ("epoch", "global_step")},
        "export_metadata": provenance,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    try:
        torch.save(exported, temporary, pickle_module=dill)
        restored, _, state_name = load_policy(temporary, "cpu")
        if state_name != selected:
            raise RuntimeError("Export selected different policy weights")
        original = payload["state_dicts"][selected]
        loaded = restored.state_dict()
        if original.keys() != loaded.keys() or any(
            not torch.equal(original[key].cpu(), loaded[key].cpu()) for key in original
        ):
            raise RuntimeError("Exported policy differs from selected checkpoint weights")
        if file_sha256(source) != source_hash:
            raise RuntimeError("Source checkpoint changed during export")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    provenance.update(checkpoint=str(destination), checkpoint_sha256=file_sha256(destination),
                      bytes=destination.stat().st_size, exact_state_dict_verified=True)
    destination.with_suffix(destination.suffix + ".json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "-c", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
