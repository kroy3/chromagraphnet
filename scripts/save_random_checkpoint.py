"""
Save a random-initialized ChromaGraphNet checkpoint as the v0.1 release
artifact. This lets users test the loading pipeline immediately, before
trained weights are available.

The checkpoint is saved with the standard PyTorch `state_dict` wrapped
in a dict alongside the config used to build the model, so users can
reconstruct the exact architecture without guessing.

Run:
    python scripts/save_random_checkpoint.py \
        --output checkpoints/chromagraphnet-base-v0.1.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from chromagraphnet import ChromaGraphNet, ChromaGraphNetConfig
from chromagraphnet import __version__


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    cfg = ChromaGraphNetConfig()
    model = ChromaGraphNet(cfg)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": __version__,
        "state_dict": model.state_dict(),
        "config": {
            "backbone": cfg.backbone.__dict__,
            "modalities": cfg.modalities.__dict__,
            "fusion": cfg.fusion.__dict__,
            "graph": cfg.graph.__dict__,
            "heads": cfg.heads.__dict__,
            "use_graph": cfg.use_graph,
        },
        "training": "random_init",
        "n_parameters": model.num_parameters(False),
    }
    torch.save(payload, out_path)
    print(f"Wrote random-init checkpoint to {out_path}")
    print(f"  version: {__version__}")
    print(f"  parameters: {model.num_parameters(False):,}")


if __name__ == "__main__":
    main()
