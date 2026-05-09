"""
ChromaFold backbone module.

A faithful PyTorch reimplementation of the two-branch CNN architecture
described in:
    Gao et al., "ChromaFold predicts the 3D contact map from single-cell
    chromatin accessibility", Nature Communications 15:9432 (2024).
    https://github.com/viannegao/ChromaFold

The backbone has two branches:

    Branch 1 (accessibility + CTCF):
        Input:  (B, 2, L_50bp)  where L_50bp = 80,200 (4.01 Mb @ 50 bp bins)
                channels = [pseudobulk scATAC, CTCF motif/ChIP track]
        Pipeline: 15 stacked Conv1D-BN-ReLU -> outer concat (L,L,2C) ->
                  3 Conv2D layers -> linear consolidator
        Output: pairwise (anchor x flank) feature map.

    Branch 2 (co-accessibility):
        Input:  (B, 40, 8020)  Jaccard co-accessibility V-stripe slice
                (40 anchor bins x 8020 flanks, 500 bp bins).
        Pipeline: 3 Conv1D -> 2 residual blocks -> 3 Conv1D -> linear.

ChromaFold's authors have not published per-layer kernel sizes/channel
counts, so the exact values below are reasonable choices that match the
public input/output shapes; users wanting bit-for-bit reproduction can
override them via the constructor.

Crucially, this implementation exposes the *fused latent immediately
before the final V-stripe predictor* via `forward_features()`. This is
the natural hook point for downstream graph attention layers in the
full ChromaGraphNet model.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ChromaFoldConfig:
    """Configuration for the ChromaFold backbone."""

    # Branch 1: accessibility + CTCF
    acc_in_channels: int = 2          # ATAC + CTCF
    acc_hidden_channels: int = 32     # 1D conv stack width
    acc_kernel_size: int = 7
    acc_num_conv1d_layers: int = 15

    # Outer concat -> 2D conv layers
    conv2d_channels: int = 32
    conv2d_kernel_size: int = 3
    num_conv2d_layers: int = 3

    # Branch 2: co-accessibility
    coacc_in_channels: int = 40        # 40 anchor bins
    coacc_hidden_channels: int = 32
    coacc_kernel_size: int = 7
    coacc_num_residual_blocks: int = 2

    # Geometry
    receptive_field_bp: int = 4_010_000     # 4.01 Mb total context
    anchor_resolution_bp: int = 10_000      # 10 kb anchor bins
    fine_resolution_bp: int = 50            # 50 bp Conv1D bins
    coacc_resolution_bp: int = 500          # 500 bp co-accessibility bins
    n_anchor_bins: int = 401                # ceil(4.01Mb / 10kb)

    # Fused embedding
    fused_dim: int = 128

    # V-stripe output (per-anchor-bin contact vector spanning +/- 2 Mb)
    vstripe_length: int = 201               # +/- 100 bins x 10 kb = +/- 1 Mb
                                            # ChromaFold uses 201 in practice

    @property
    def n_fine_bins(self) -> int:
        return self.receptive_field_bp // self.fine_resolution_bp   # 80,200

    @property
    def n_coacc_bins(self) -> int:
        return self.receptive_field_bp // self.coacc_resolution_bp  # 8,020


class _Conv1dBNReLU(nn.Module):
    """One Conv1D -> BatchNorm1d -> ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size, padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class _ResidualBlock1d(nn.Module):
    """Simple 2-layer 1D residual block with BatchNorm."""

    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + x)


class AccessibilityBranch(nn.Module):
    """
    Branch 1: scATAC + CTCF -> pairwise (anchor, flank) features.

    Pipeline:
        Conv1D stack (15 layers) -> downsample to anchor resolution (10 kb)
        -> outer concatenation to make a pairwise tensor
        -> 3 Conv2D layers -> linear consolidator
    """

    def __init__(self, cfg: ChromaFoldConfig):
        super().__init__()
        self.cfg = cfg

        # Conv1D stack
        layers = []
        in_ch = cfg.acc_in_channels
        for i in range(cfg.acc_num_conv1d_layers):
            out_ch = cfg.acc_hidden_channels
            layers.append(_Conv1dBNReLU(in_ch, out_ch, cfg.acc_kernel_size))
            in_ch = out_ch
        self.conv1d_stack = nn.Sequential(*layers)

        # Downsample 50 bp -> 10 kb (factor of 200) via average pooling
        self.downsample = nn.AvgPool1d(
            kernel_size=cfg.anchor_resolution_bp // cfg.fine_resolution_bp,
            stride=cfg.anchor_resolution_bp // cfg.fine_resolution_bp,
        )

        # Outer-concat doubles the channel dim, so 2D convs see 2*hidden
        conv2d_in = 2 * cfg.acc_hidden_channels
        conv2d_layers = []
        for i in range(cfg.num_conv2d_layers):
            out_ch = cfg.conv2d_channels
            conv2d_layers.append(
                nn.Conv2d(conv2d_in, out_ch, cfg.conv2d_kernel_size,
                          padding=cfg.conv2d_kernel_size // 2)
            )
            conv2d_layers.append(nn.BatchNorm2d(out_ch))
            conv2d_layers.append(nn.ReLU(inplace=True))
            conv2d_in = out_ch
        self.conv2d_stack = nn.Sequential(*conv2d_layers)

        # Per-anchor-bin linear consolidator: (anchor, flank, C) -> (anchor, fused_dim)
        # We pool over the flank axis with a learnable 1x1 attention pool.
        self.flank_pool = nn.Conv2d(cfg.conv2d_channels, 1, kernel_size=1)
        self.consolidator = nn.Linear(cfg.conv2d_channels, cfg.fused_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, L_50bp) input.
        Returns:
            anchor_features: (B, n_anchor_bins, fused_dim)
        """
        h = self.conv1d_stack(x)              # (B, C, L_50bp)
        h = self.downsample(h)                # (B, C, n_anchor_bins)

        # Outer concatenation: pair every anchor bin with every flank bin.
        # Result: (B, 2C, n_anchor_bins, n_anchor_bins)
        B, C, L = h.shape
        a = h.unsqueeze(3).expand(B, C, L, L)            # anchor axis
        f = h.unsqueeze(2).expand(B, C, L, L)            # flank axis
        pair = torch.cat([a, f], dim=1)                  # (B, 2C, L, L)

        pair = self.conv2d_stack(pair)                   # (B, C', L, L)

        # Attention-pool across flank axis to get one vector per anchor bin.
        attn = self.flank_pool(pair)                     # (B, 1, L, L)
        attn = F.softmax(attn, dim=-1)                   # softmax over flanks
        anchor = (pair * attn).sum(dim=-1)               # (B, C', L)
        anchor = anchor.transpose(1, 2)                  # (B, L, C')
        anchor = self.consolidator(anchor)               # (B, L, fused_dim)
        return anchor


class CoaccessibilityBranch(nn.Module):
    """
    Branch 2: 500-bp Jaccard co-accessibility V-stripe.

    Pipeline:
        3 Conv1D -> 2 residual blocks -> 3 Conv1D -> linear -> per-bin features
    """

    def __init__(self, cfg: ChromaFoldConfig):
        super().__init__()
        self.cfg = cfg

        c = cfg.coacc_hidden_channels
        k = cfg.coacc_kernel_size

        self.entry = nn.Sequential(
            _Conv1dBNReLU(cfg.coacc_in_channels, c, k),
            _Conv1dBNReLU(c, c, k),
            _Conv1dBNReLU(c, c, k),
        )
        self.residual_stack = nn.Sequential(*[
            _ResidualBlock1d(c, k)
            for _ in range(cfg.coacc_num_residual_blocks)
        ])
        self.exit = nn.Sequential(
            _Conv1dBNReLU(c, c, k),
            _Conv1dBNReLU(c, c, k),
            _Conv1dBNReLU(c, c, k),
        )

        # Downsample 500 bp -> 10 kb (factor of 20)
        self.downsample = nn.AvgPool1d(
            kernel_size=cfg.anchor_resolution_bp // cfg.coacc_resolution_bp,
            stride=cfg.anchor_resolution_bp // cfg.coacc_resolution_bp,
        )
        self.consolidator = nn.Linear(c, cfg.fused_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 40, n_coacc_bins) co-accessibility V-stripe.
        Returns:
            features: (B, n_anchor_bins, fused_dim)
        """
        h = self.entry(x)
        h = self.residual_stack(h)
        h = self.exit(h)                              # (B, C, n_coacc_bins)
        h = self.downsample(h)                        # (B, C, n_anchor_bins)
        h = h.transpose(1, 2)                         # (B, L, C)
        h = self.consolidator(h)                      # (B, L, fused_dim)
        return h


class ChromaFoldBackbone(nn.Module):
    """
    Full ChromaFold backbone.

    Returns per-anchor-bin embeddings that the downstream graph and
    multi-modal fusion modules consume. The original ChromaFold model
    appends a small MLP head to predict a 201-d V-stripe; we expose
    that head as `predict_vstripe()` for backwards-compatible inference,
    but the natural extension point is `forward_features()`.
    """

    def __init__(self, cfg: ChromaFoldConfig | None = None):
        super().__init__()
        self.cfg = cfg or ChromaFoldConfig()

        self.acc_branch = AccessibilityBranch(self.cfg)
        self.coacc_branch = CoaccessibilityBranch(self.cfg)

        # Linear fusion of the two branches.
        self.fusion = nn.Linear(2 * self.cfg.fused_dim, self.cfg.fused_dim)

        # Backwards-compatible V-stripe head (per anchor bin -> 201 contact scores).
        self.vstripe_head = nn.Linear(self.cfg.fused_dim,
                                      self.cfg.vstripe_length)

    def forward_features(
        self,
        acc_ctcf: torch.Tensor,
        coacc: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns the fused latent immediately before the V-stripe predictor.
        This is the recommended hook point for downstream extensions.

        Args:
            acc_ctcf: (B, 2, n_fine_bins)
            coacc:    (B, 40, n_coacc_bins)

        Returns:
            features: (B, n_anchor_bins, fused_dim)
        """
        a = self.acc_branch(acc_ctcf)        # (B, L, D)
        c = self.coacc_branch(coacc)         # (B, L, D)
        fused = torch.cat([a, c], dim=-1)    # (B, L, 2D)
        return self.fusion(fused)            # (B, L, D)

    def predict_vstripe(
        self,
        acc_ctcf: torch.Tensor,
        coacc: torch.Tensor,
    ) -> torch.Tensor:
        """ChromaFold-style HiC-DC+ Z-score V-stripe prediction."""
        feats = self.forward_features(acc_ctcf, coacc)
        return self.vstripe_head(feats)      # (B, L, vstripe_length)

    def forward(
        self,
        acc_ctcf: torch.Tensor,
        coacc: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        if return_features:
            return self.forward_features(acc_ctcf, coacc)
        return self.predict_vstripe(acc_ctcf, coacc)
