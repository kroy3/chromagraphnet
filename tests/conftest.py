"""
Shared pytest fixtures for ChromaGraphNet tests.
"""
from __future__ import annotations

import pytest
import torch

from chromagraphnet import ChromaGraphNet, ChromaGraphNetConfig


def _small_config() -> ChromaGraphNetConfig:
    """A tiny config that exercises every code path in <2s."""
    cfg = ChromaGraphNetConfig()
    cfg.backbone.receptive_field_bp = 400_000
    cfg.backbone.n_anchor_bins = 40
    cfg.backbone.vstripe_length = 21
    cfg.backbone.acc_num_conv1d_layers = 4
    cfg.fusion.n_latents = 16
    cfg.fusion.n_layers = 1
    cfg.graph.n_layers = 2
    cfg.heads.n_anchor_bins = 40
    cfg.heads.vstripe_length = 21
    cfg.fusion.n_anchor_bins = 40
    cfg.modalities.n_anchor_bins = 40
    cfg.__post_init__()
    return cfg


@pytest.fixture(scope="session")
def small_config() -> ChromaGraphNetConfig:
    return _small_config()


@pytest.fixture(scope="session")
def small_model(small_config: ChromaGraphNetConfig) -> ChromaGraphNet:
    return ChromaGraphNet(small_config)


@pytest.fixture
def random_inputs(small_config: ChromaGraphNetConfig):
    cfg = small_config
    B = 2
    L = cfg.backbone.n_anchor_bins
    return {
        "acc_ctcf": torch.randn(B, 2, cfg.backbone.n_fine_bins),
        "coacc": torch.randn(B, 40, cfg.backbone.n_coacc_bins),
        "rna": torch.randn(B, L, 1),
        "chip": torch.randn(B, 5, cfg.backbone.n_fine_bins),
        "motif": torch.randn(B, L, 200),
    }
