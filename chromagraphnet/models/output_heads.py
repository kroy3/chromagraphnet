"""
Multi-task output heads.

ChromaGraphNet predicts four interrelated genomic structural features
from the same fused per-bin embedding:

    1. Contact map: pairwise (anchor, flank) contact strength.
       We predict at the same 10 kb anchor resolution as ChromaFold,
       and additionally provide a 5 kb upsampling head.

    2. Compartments: the 5-state Rao et al. (2014) classification
       (A1, A2, B1, B2, B3) per anchor bin.

    3. Loop anchors: a per-bin probability of being a CTCF/cohesin
       loop anchor.

    4. Insulation score: a continuous TAD boundary score.

The contact and compartment heads share an MC-Dropout layer that
provides epistemic uncertainty estimates at inference time.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HeadsConfig:
    embed_dim: int = 128
    n_anchor_bins: int = 401
    vstripe_length: int = 201
    n_compartment_subtypes: int = 5     # A1, A2, B1, B2, B3
    upsample_factor: int = 2            # 10 kb -> 5 kb
    mc_dropout: float = 0.1


class ContactMapHead(nn.Module):
    """
    Predicts the pairwise contact map by combining anchor and flank
    embeddings via a bilinear interaction, then applying a small MLP.

    Output shape: (B, n_anchor_bins, n_anchor_bins) at 10 kb resolution.
    For sparse training we additionally support the V-stripe parameterization
    used by ChromaFold (per-anchor-bin contact vector).
    """

    def __init__(self, cfg: HeadsConfig):
        super().__init__()
        self.cfg = cfg
        # Learnable bilinear map for pairwise interaction.
        self.bilinear = nn.Bilinear(cfg.embed_dim, cfg.embed_dim,
                                    cfg.embed_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
            nn.GELU(),
            nn.Dropout(cfg.mc_dropout),
            nn.Linear(cfg.embed_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        B, L, D = x.shape
        # Expand into pairwise (B, L, L, D) tensors.
        a = x.unsqueeze(2).expand(B, L, L, D).contiguous()
        b = x.unsqueeze(1).expand(B, L, L, D).contiguous()
        pair = self.bilinear(a.view(-1, D), b.view(-1, D))
        contact = self.mlp(pair).view(B, L, L)
        # Symmetrize.
        contact = 0.5 * (contact + contact.transpose(1, 2))
        return contact


class VStripeHead(nn.Module):
    """ChromaFold-style per-anchor-bin V-stripe head."""

    def __init__(self, cfg: HeadsConfig):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, cfg.vstripe_length),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)            # (B, L, vstripe_length)


class CompartmentHead(nn.Module):
    """5-way classification head for A1/A2/B1/B2/B3 compartments."""

    def __init__(self, cfg: HeadsConfig):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
            nn.GELU(),
            nn.Dropout(cfg.mc_dropout),
            nn.Linear(cfg.embed_dim, cfg.n_compartment_subtypes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)            # (B, L, 5) logits


class LoopAnchorHead(nn.Module):
    """Per-bin probability of being a CTCF/cohesin loop anchor."""

    def __init__(self, cfg: HeadsConfig):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, cfg.embed_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.embed_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)  # (B, L)


class InsulationHead(nn.Module):
    """Continuous insulation/boundary score per bin."""

    def __init__(self, cfg: HeadsConfig):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, cfg.embed_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.embed_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)  # (B, L)


class MultiTaskHeads(nn.Module):
    """
    Bundle the four heads into a single module so the main model can call
    them with one forward pass and a clean output dict.
    """

    def __init__(self, cfg: HeadsConfig | None = None):
        super().__init__()
        self.cfg = cfg or HeadsConfig()
        self.contact = ContactMapHead(self.cfg)
        self.vstripe = VStripeHead(self.cfg)
        self.compartment = CompartmentHead(self.cfg)
        self.loop = LoopAnchorHead(self.cfg)
        self.insulation = InsulationHead(self.cfg)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "contact_map": self.contact(x),
            "vstripe": self.vstripe(x),
            "compartment_logits": self.compartment(x),
            "loop_anchor_logits": self.loop(x),
            "insulation": self.insulation(x),
        }

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 20,
    ) -> dict[str, torch.Tensor]:
        """
        Monte-Carlo dropout uncertainty estimation.

        Performs `n_samples` stochastic forward passes through the heads
        with dropout enabled, returning the per-output mean and standard
        deviation. Useful for flagging low-confidence predictions in
        clinical / liquid-biopsy applications.
        """
        # Re-enable dropout layers even though we are in eval mode.
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        samples = [self.forward(x) for _ in range(n_samples)]
        # Restore eval mode.
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.eval()

        out: dict[str, torch.Tensor] = {}
        for k in samples[0]:
            stacked = torch.stack([s[k] for s in samples], dim=0)
            out[f"{k}_mean"] = stacked.mean(dim=0)
            out[f"{k}_std"] = stacked.std(dim=0)
        return out
