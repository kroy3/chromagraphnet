"""
Inference utilities for ChromaGraphNet.

This module wraps the model with:
    * A `from_pretrained()` classmethod that loads either ChromaFold-only
      weights into the backbone or full ChromaGraphNet weights.
    * A `predict_window()` helper that takes raw modality tensors and
      returns the multi-task output dict.
    * A `predict_genome_wide()` helper that tiles 4 Mb windows across a
      chromosome and stitches the per-window predictions back into a
      full contact matrix.

For the v1 release we ship random-initialized weights only; users can
provide their own weights via `--checkpoint` or train from scratch
using the (forthcoming) training pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..models.chromagraphnet import ChromaGraphNet, ChromaGraphNetConfig


def load_model(
    checkpoint_path: str | Path | None = None,
    config: ChromaGraphNetConfig | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = False,
) -> ChromaGraphNet:
    """Construct a ChromaGraphNet and optionally load weights from disk."""
    model = ChromaGraphNet(config)
    if checkpoint_path is None:
        return model

    sd = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model_state" in sd:
        sd = sd["model_state"]

    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing:
        print(f"[load_model] {len(missing)} missing keys (newly initialized).")
    if unexpected:
        print(f"[load_model] {len(unexpected)} unexpected keys (ignored).")
    return model


@torch.no_grad()
def predict_window(
    model: ChromaGraphNet,
    acc_ctcf: torch.Tensor,
    coacc: torch.Tensor,
    rna: torch.Tensor | None = None,
    chip: torch.Tensor | None = None,
    motif: torch.Tensor | None = None,
    edge_index: torch.Tensor | None = None,
    edge_attr: torch.Tensor | None = None,
    return_uncertainty: bool = False,
) -> dict[str, torch.Tensor]:
    """Single-window forward pass in eval mode. Inputs follow the same
    shape conventions as ChromaGraphNet.forward."""
    model.eval()
    return model.predict(
        acc_ctcf=acc_ctcf,
        coacc=coacc,
        rna=rna,
        chip=chip,
        motif=motif,
        edge_index=edge_index,
        edge_attr=edge_attr,
        return_uncertainty=return_uncertainty,
    )
