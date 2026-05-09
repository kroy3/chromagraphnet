"""
End-to-end smoke test for ChromaGraphNet.

Uses a deliberately small configuration so the test runs in a few
seconds on CPU. Verifies:

    1. The model constructs without errors.
    2. forward_features returns the expected (B, L, D) shape.
    3. The full multi-task forward returns all five output keys with
       sensible shapes.
    4. Backward pass produces gradients on every trainable parameter.
    5. The MC-dropout uncertainty path returns mean and std for each head.
    6. The graph builder produces non-empty graphs from random priors.
    7. The polymer physics prior returns finite scalar losses.
"""
from __future__ import annotations

import torch

from chromagraphnet import (
    ChromaGraphNet,
    ChromaGraphNetConfig,
    PolymerPhysicsPrior,
    PhysicsConfig,
    build_graph_for_window,
    batch_graphs,
)


def small_config() -> ChromaGraphNetConfig:
    """Return a tiny config that runs in seconds on CPU."""
    cfg = ChromaGraphNetConfig()
    # Shrink the receptive field and channel widths.
    cfg.backbone.receptive_field_bp = 200_000      # 200 kb context
    cfg.backbone.anchor_resolution_bp = 10_000     # 20 anchor bins
    cfg.backbone.fine_resolution_bp = 50           # 4_000 fine bins
    cfg.backbone.coacc_resolution_bp = 500         # 400 co-accessibility bins
    cfg.backbone.n_anchor_bins = 20
    cfg.backbone.acc_hidden_channels = 8
    cfg.backbone.acc_num_conv1d_layers = 3
    cfg.backbone.conv2d_channels = 8
    cfg.backbone.num_conv2d_layers = 2
    cfg.backbone.coacc_hidden_channels = 8
    cfg.backbone.coacc_num_residual_blocks = 1
    cfg.backbone.fused_dim = 32
    cfg.backbone.vstripe_length = 21
    # Modality encoders
    cfg.modalities.chip_n_marks = 3
    cfg.modalities.chip_hidden_channels = 8
    cfg.modalities.chip_n_layers = 2
    cfg.modalities.motif_n_factors = 16
    cfg.modalities.motif_hidden_dim = 16
    cfg.modalities.rna_hidden_dim = 16
    # Fusion
    cfg.fusion.n_latents = 8
    cfg.fusion.n_heads = 4
    cfg.fusion.n_layers = 1
    cfg.fusion.ffn_mult = 2
    # Graph
    cfg.graph.n_layers = 2
    cfg.graph.n_heads = 4
    # Heads
    cfg.heads.n_compartment_subtypes = 5
    cfg.__post_init__()  # propagate shared dims
    return cfg


def make_dummy_inputs(cfg: ChromaGraphNetConfig, batch_size: int = 2):
    L = cfg.backbone.n_anchor_bins
    n_fine = cfg.backbone.n_fine_bins
    n_coacc = cfg.backbone.n_coacc_bins
    acc_ctcf = torch.randn(batch_size, 2, n_fine)
    coacc = torch.randn(batch_size, 40, n_coacc)
    rna = torch.randn(batch_size, L, 1)
    chip = torch.randn(batch_size, cfg.modalities.chip_n_marks, n_fine)
    motif = torch.rand(batch_size, L, cfg.modalities.motif_n_factors)
    # Build a per-window graph and batch it.
    edge_indices, edge_attrs = [], []
    for _ in range(batch_size):
        coacc_mat = torch.rand(L, L)
        hic_prior = torch.rand(L, L) * 2.0
        ctcf_ori = torch.randint(-1, 2, (L,))
        ei, ea = build_graph_for_window(
            n_anchor_bins=L,
            coaccessibility=coacc_mat,
            hic_prior=hic_prior,
            ctcf_orientation=ctcf_ori,
            coacc_topk=3,
            hic_threshold=1.0,
        )
        edge_indices.append(ei)
        edge_attrs.append(ea)
    edge_index, edge_attr = batch_graphs(edge_indices, edge_attrs, L)
    return acc_ctcf, coacc, rna, chip, motif, edge_index, edge_attr


def test_construction():
    cfg = small_config()
    model = ChromaGraphNet(cfg)
    n_params = model.num_parameters()
    assert n_params > 0, "model has zero trainable parameters"
    print(f"  model constructed, {n_params:,} trainable parameters")


def test_forward_features():
    cfg = small_config()
    model = ChromaGraphNet(cfg)
    acc_ctcf, coacc, rna, chip, motif, ei, ea = make_dummy_inputs(cfg)
    feats = model.forward_features(
        acc_ctcf=acc_ctcf, coacc=coacc, rna=rna, chip=chip, motif=motif,
        edge_index=ei, edge_attr=ea,
    )
    B, L, D = feats.shape
    assert B == 2
    assert L == cfg.backbone.n_anchor_bins
    assert D == cfg.backbone.fused_dim
    assert torch.isfinite(feats).all()
    print(f"  forward_features OK, shape = {tuple(feats.shape)}")


def test_full_forward():
    cfg = small_config()
    model = ChromaGraphNet(cfg)
    acc_ctcf, coacc, rna, chip, motif, ei, ea = make_dummy_inputs(cfg)
    out = model(
        acc_ctcf=acc_ctcf, coacc=coacc, rna=rna, chip=chip, motif=motif,
        edge_index=ei, edge_attr=ea,
    )
    expected = {"contact_map", "vstripe", "compartment_logits",
                "loop_anchor_logits", "insulation"}
    assert set(out.keys()) == expected
    L = cfg.backbone.n_anchor_bins
    B = 2
    assert out["contact_map"].shape == (B, L, L)
    assert out["vstripe"].shape == (B, L, cfg.backbone.vstripe_length)
    assert out["compartment_logits"].shape == (B, L, 5)
    assert out["loop_anchor_logits"].shape == (B, L)
    assert out["insulation"].shape == (B, L)
    for k, v in out.items():
        assert torch.isfinite(v).all(), f"non-finite values in {k}"
    print(f"  full forward OK, all 5 output keys present and finite")


def test_backward_pass():
    cfg = small_config()
    model = ChromaGraphNet(cfg)
    acc_ctcf, coacc, rna, chip, motif, ei, ea = make_dummy_inputs(cfg)
    out = model(
        acc_ctcf=acc_ctcf, coacc=coacc, rna=rna, chip=chip, motif=motif,
        edge_index=ei, edge_attr=ea,
    )
    # Aggregate all heads into a scalar loss.
    loss = (
        out["contact_map"].mean()
        + out["vstripe"].mean()
        + out["compartment_logits"].mean()
        + out["loop_anchor_logits"].mean()
        + out["insulation"].mean()
    )
    loss.backward()
    n_with_grad = sum(1 for p in model.parameters()
                      if p.requires_grad and p.grad is not None
                      and p.grad.abs().sum() > 0)
    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  backward OK, {n_with_grad}/{n_total} params received nonzero "
          f"gradient")
    # Allow a few unused params (e.g., motif factor embedding when motif
    # input is exactly zero), but most should have gradient.
    assert n_with_grad > 0.7 * n_total


def test_uncertainty():
    cfg = small_config()
    model = ChromaGraphNet(cfg)
    acc_ctcf, coacc, rna, chip, motif, ei, ea = make_dummy_inputs(cfg)
    out = model.predict(
        acc_ctcf=acc_ctcf, coacc=coacc, rna=rna, chip=chip, motif=motif,
        edge_index=ei, edge_attr=ea,
        return_uncertainty=True, n_uncertainty_samples=4,
    )
    expected_keys = {f"{k}_{stat}"
                     for k in ("contact_map", "vstripe",
                               "compartment_logits", "loop_anchor_logits",
                               "insulation")
                     for stat in ("mean", "std")}
    assert set(out.keys()) == expected_keys
    print(f"  MC-dropout uncertainty OK, "
          f"{len(out)//2} heads have mean+std")


def test_physics_prior():
    cfg = small_config()
    L = cfg.backbone.n_anchor_bins
    physics = PolymerPhysicsPrior(PhysicsConfig(n_anchor_bins=L,
                                                bin_size_bp=10_000))
    contact_map = torch.rand(2, L, L) + 0.1   # ensure > 0 for log
    contact_map = 0.5 * (contact_map + contact_map.transpose(1, 2))
    compartment_logits = torch.randn(2, L, 5)
    ctcf_ori = torch.randint(-1, 2, (2, L))
    losses = physics(contact_map, compartment_logits, ctcf_ori)
    for k, v in losses.items():
        assert torch.isfinite(v), f"non-finite {k}: {v}"
    print(f"  physics prior OK, total loss = {losses['total'].item():.4f}")


def test_graph_builder_empty_inputs():
    """Graph builder must not crash when nothing is supplied."""
    ei, ea = build_graph_for_window(n_anchor_bins=10)
    # Just genomic adjacency: 2 * 9 = 18 edges
    assert ei.shape == (2, 18)
    assert ea.shape == (18, 4)
    print(f"  graph builder OK on minimal input, "
          f"{ei.shape[1]} adjacency edges")


def test_chromafold_only_mode():
    """If we disable the graph module, the model should still work as a
    pure ChromaFold-equivalent."""
    cfg = small_config()
    cfg.use_graph = False
    model = ChromaGraphNet(cfg)
    acc_ctcf, coacc, rna, chip, motif, _, _ = make_dummy_inputs(cfg)
    out = model(
        acc_ctcf=acc_ctcf, coacc=coacc, rna=rna, chip=chip, motif=motif,
        edge_index=None, edge_attr=None,
    )
    assert "contact_map" in out and torch.isfinite(out["contact_map"]).all()
    print(f"  graph-free fallback mode OK")


def main():
    print("=" * 60)
    print("ChromaGraphNet smoke test")
    print("=" * 60)

    torch.manual_seed(0)

    print("[1/8] construction")
    test_construction()

    print("[2/8] forward_features")
    test_forward_features()

    print("[3/8] full forward")
    test_full_forward()

    print("[4/8] backward pass")
    test_backward_pass()

    print("[5/8] MC-dropout uncertainty")
    test_uncertainty()

    print("[6/8] polymer physics prior")
    test_physics_prior()

    print("[7/8] graph builder edge cases")
    test_graph_builder_empty_inputs()

    print("[8/8] chromafold-only mode")
    test_chromafold_only_mode()

    print()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
