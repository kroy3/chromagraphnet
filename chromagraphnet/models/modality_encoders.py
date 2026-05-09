"""
Modality-specific encoders.

Each encoder takes a per-anchor-bin signal (or a higher-resolution signal
that is downsampled internally) and returns a (B, n_anchor_bins, embed_dim)
tensor that lives in the same embedding space as the ChromaFold backbone.

All encoders share the same output interface so the downstream FiLM /
bottleneck transformer fusion module can treat them interchangeably.

Supported modalities (v1):
    1. scRNA-seq:        per-bin pseudobulk expression (log1p, length-norm).
    2. Histone ChIP-seq: 1-D CNN over multiple histone marks at 50 bp.
    3. TF motif grammar: per-bin counts of motif hits across a TF panel.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModalityEncoderConfig:
    n_anchor_bins: int = 401
    embed_dim: int = 128

    # scRNA-seq
    rna_use: bool = True
    rna_in_dim: int = 1                # one channel: log1p pseudobulk expression
    rna_hidden_dim: int = 64

    # Histone ChIP / CUT&RUN
    chip_use: bool = True
    chip_n_marks: int = 5              # H3K4me1/3, H3K27ac, H3K27me3, H3K36me3
    chip_hidden_channels: int = 32
    chip_kernel_size: int = 7
    chip_n_layers: int = 4
    chip_input_resolution_bp: int = 50    # native ChIP coverage resolution
    chip_anchor_resolution_bp: int = 10_000

    # TF motif grammar
    motif_use: bool = True
    motif_n_factors: int = 200            # JASPAR core vertebrates panel
    motif_hidden_dim: int = 64


class _PerBinMLP(nn.Module):
    """Tiny MLP applied per anchor bin (channel-mixing only)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class scRNAEncoder(nn.Module):
    """
    Encodes per-bin pseudobulk gene expression.

    Expected input: (B, n_anchor_bins, rna_in_dim) — typically a single
    channel of log1p-normalized expression, summed/averaged over genes
    overlapping each 10 kb anchor bin. Multi-channel input is supported
    (e.g., separate channels for nascent vs. mature transcripts).
    """

    def __init__(self, cfg: ModalityEncoderConfig):
        super().__init__()
        self.mlp = _PerBinMLP(cfg.rna_in_dim, cfg.rna_hidden_dim, cfg.embed_dim)

        # Small 1-D conv to capture neighbourhood context (e.g., gene bodies
        # spanning multiple bins).
        self.context = nn.Conv1d(cfg.embed_dim, cfg.embed_dim,
                                 kernel_size=5, padding=2)
        self.norm = nn.LayerNorm(cfg.embed_dim)

    def forward(self, rna: torch.Tensor) -> torch.Tensor:
        # rna: (B, L, in_dim)
        h = self.mlp(rna)                                  # (B, L, D)
        h = self.context(h.transpose(1, 2)).transpose(1, 2)
        return self.norm(h)


class HistoneChIPEncoder(nn.Module):
    """
    1-D CNN over multiple histone-mark coverage tracks.

    Expected input: (B, n_marks, L_fine) where L_fine is the number of
    50 bp bins covering the 4.01 Mb context. The encoder applies a
    short Conv1D stack and downsamples to the 10 kb anchor grid.
    """

    def __init__(self, cfg: ModalityEncoderConfig):
        super().__init__()
        self.cfg = cfg

        layers = []
        in_ch = cfg.chip_n_marks
        for _ in range(cfg.chip_n_layers):
            layers += [
                nn.Conv1d(in_ch, cfg.chip_hidden_channels,
                          cfg.chip_kernel_size,
                          padding=cfg.chip_kernel_size // 2),
                nn.BatchNorm1d(cfg.chip_hidden_channels),
                nn.GELU(),
            ]
            in_ch = cfg.chip_hidden_channels
        self.conv_stack = nn.Sequential(*layers)

        downsample_factor = (cfg.chip_anchor_resolution_bp
                             // cfg.chip_input_resolution_bp)   # 200
        self.downsample = nn.AvgPool1d(downsample_factor, downsample_factor)
        self.proj = nn.Linear(cfg.chip_hidden_channels, cfg.embed_dim)
        self.norm = nn.LayerNorm(cfg.embed_dim)

    def forward(self, chip: torch.Tensor) -> torch.Tensor:
        # chip: (B, n_marks, L_fine)
        h = self.conv_stack(chip)            # (B, C, L_fine)
        h = self.downsample(h)               # (B, C, L_anchor)
        h = h.transpose(1, 2)                # (B, L_anchor, C)
        return self.norm(self.proj(h))


class MotifEncoder(nn.Module):
    """
    Encodes a per-bin TF motif-count matrix (e.g., max FIMO score per
    factor in each 10 kb anchor bin).

    Expected input: (B, n_anchor_bins, n_factors). For sparse panels the
    user can pass log1p(count) or normalized motif scores.
    """

    def __init__(self, cfg: ModalityEncoderConfig):
        super().__init__()
        self.proj_in = nn.Linear(cfg.motif_n_factors, cfg.motif_hidden_dim)
        self.gelu = nn.GELU()
        # Learnable factor-embedding: lets the model express that two
        # related TFs (e.g., NPAS4 and FOS) should drive similar contact
        # changes by getting similar effective embeddings post-projection.
        self.factor_embed = nn.Parameter(
            torch.randn(cfg.motif_n_factors, cfg.motif_hidden_dim) * 0.02
        )
        self.proj_out = nn.Linear(cfg.motif_hidden_dim, cfg.embed_dim)
        self.norm = nn.LayerNorm(cfg.embed_dim)

    def forward(self, motif: torch.Tensor) -> torch.Tensor:
        # motif: (B, L, n_factors); softly attend over the factor panel
        # using a learnable per-factor embedding.
        weighted = motif @ self.factor_embed                # (B, L, hidden)
        h = self.gelu(self.proj_in(motif) + weighted)       # (B, L, hidden)
        return self.norm(self.proj_out(h))                  # (B, L, embed)


class ModalityEncoderBank(nn.Module):
    """
    Convenience wrapper that constructs whichever encoders are enabled
    in the config and returns a dict {modality_name: tensor}.
    """

    def __init__(self, cfg: ModalityEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.rna = scRNAEncoder(cfg) if cfg.rna_use else None
        self.chip = HistoneChIPEncoder(cfg) if cfg.chip_use else None
        self.motif = MotifEncoder(cfg) if cfg.motif_use else None

    def forward(
        self,
        rna: torch.Tensor | None = None,
        chip: torch.Tensor | None = None,
        motif: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        if self.rna is not None and rna is not None:
            out["rna"] = self.rna(rna)
        if self.chip is not None and chip is not None:
            out["chip"] = self.chip(chip)
        if self.motif is not None and motif is not None:
            out["motif"] = self.motif(motif)
        return out
