"""
Command-line interface.

Entry points (registered in pyproject.toml):
    chromagraphnet-predict   single-window or multi-window inference
    chromagraphnet-info      print model size, config, and modality status
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .inference.predict import load_model
from .data.graph_builder import build_graph_for_window
from .models.chromagraphnet import ChromaGraphNetConfig


def _load_npz(path: str) -> dict:
    """Load an NPZ file and convert arrays to torch tensors."""
    data = np.load(path, allow_pickle=False)
    out: dict[str, torch.Tensor] = {}
    for k in data.files:
        out[k] = torch.from_numpy(data[k]).float()
    return out


def predict_main(argv: list[str] | None = None) -> int:
    """Entry point for `chromagraphnet-predict`."""
    parser = argparse.ArgumentParser(
        prog="chromagraphnet-predict",
        description=(
            "Run ChromaGraphNet inference on a single window of preprocessed "
            "inputs. Inputs are read from an NPZ file with keys 'acc_ctcf' "
            "and 'coacc' (required) and optionally 'rna', 'chip', 'motif', "
            "'coacc_matrix', 'hic_prior', and 'ctcf_orientation'."
        ),
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a .pt checkpoint. If omitted, the model "
                             "is initialized with random weights.")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to an NPZ file with preprocessed inputs.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to write predictions as NPZ.")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--uncertainty", action="store_true",
                        help="Return MC-dropout means and standard deviations.")
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Number of MC dropout samples (default: 20).")
    parser.add_argument("--no-graph", action="store_true",
                        help="Disable the GAT module (lightweight inference).")
    args = parser.parse_args(argv)

    cfg = ChromaGraphNetConfig()
    cfg.use_graph = not args.no_graph

    device = torch.device(args.device)
    model = load_model(args.checkpoint, config=cfg, map_location=device)
    model.to(device).eval()

    inputs = _load_npz(args.input)
    if "acc_ctcf" not in inputs or "coacc" not in inputs:
        print("ERROR: input NPZ must contain 'acc_ctcf' and 'coacc'.",
              file=sys.stderr)
        return 1

    # Add a batch dim if absent.
    def _ensure_batched(t: torch.Tensor, expected_ndim: int) -> torch.Tensor:
        if t.ndim == expected_ndim - 1:
            return t.unsqueeze(0)
        return t

    acc_ctcf = _ensure_batched(inputs["acc_ctcf"], 3).to(device)
    coacc = _ensure_batched(inputs["coacc"], 3).to(device)
    rna = inputs.get("rna")
    chip = inputs.get("chip")
    motif = inputs.get("motif")
    if rna is not None:
        rna = _ensure_batched(rna, 3).to(device)
    if chip is not None:
        chip = _ensure_batched(chip, 3).to(device)
    if motif is not None:
        motif = _ensure_batched(motif, 3).to(device)

    edge_index = edge_attr = None
    if cfg.use_graph:
        coacc_mat = inputs.get("coacc_matrix")
        hic_prior = inputs.get("hic_prior")
        ctcf_or = inputs.get("ctcf_orientation")
        if ctcf_or is not None:
            ctcf_or = ctcf_or.long()
        edge_index, edge_attr = build_graph_for_window(
            n_anchor_bins=cfg.backbone.n_anchor_bins,
            coaccessibility=coacc_mat,
            hic_prior=hic_prior,
            ctcf_orientation=ctcf_or,
            device=device,
        )

    with torch.no_grad():
        out = model.predict(
            acc_ctcf=acc_ctcf, coacc=coacc, rna=rna, chip=chip, motif=motif,
            edge_index=edge_index, edge_attr=edge_attr,
            return_uncertainty=args.uncertainty,
            n_uncertainty_samples=args.n_samples,
        )

    np_out = {k: v.cpu().numpy() for k, v in out.items()}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **np_out)
    print(f"Wrote predictions to {args.output}")
    print(f"  Keys: {list(np_out.keys())}")
    return 0


def info_main(argv: list[str] | None = None) -> int:
    """Entry point for `chromagraphnet-info`."""
    parser = argparse.ArgumentParser(
        prog="chromagraphnet-info",
        description="Print model size and configuration summary.",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args(argv)

    from .models.chromagraphnet import ChromaGraphNet

    cfg = ChromaGraphNetConfig()
    model = ChromaGraphNet(cfg) if args.checkpoint is None else load_model(
        args.checkpoint, config=cfg
    )

    print(f"ChromaGraphNet version: {__version__}")
    print(f"Total parameters:       {model.num_parameters(False):,}")
    print(f"Trainable parameters:   {model.num_parameters(True):,}")
    print()
    print(f"Receptive field:        {cfg.backbone.receptive_field_bp:,} bp")
    print(f"Anchor resolution:      {cfg.backbone.anchor_resolution_bp:,} bp")
    print(f"Anchor bins per window: {cfg.backbone.n_anchor_bins}")
    print(f"Embedding dim:          {cfg.backbone.fused_dim}")
    print()
    print("Modules:")
    print(f"  Backbone:   ChromaFold (2-branch CNN, fine-tunable)")
    print(f"  Modalities: RNA={cfg.modalities.rna_use}, "
          f"ChIP={cfg.modalities.chip_use}, "
          f"Motif={cfg.modalities.motif_use}")
    print(f"  Fusion:     FiLM + Bottleneck Transformer "
          f"({cfg.fusion.n_layers} layers, {cfg.fusion.n_latents} latents)")
    print(f"  Graph:      GATv2 ({cfg.graph.n_layers} layers, "
          f"{cfg.graph.n_heads} heads) [enabled={cfg.use_graph}]")
    print(f"  Heads:      contact, vstripe, compartment(5), loop, insulation")
    return 0


if __name__ == "__main__":
    sys.exit(predict_main())
