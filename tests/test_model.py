"""End-to-end smoke tests for ChromaGraphNet."""
from __future__ import annotations

import torch

from chromagraphnet import (
    ChromaGraphNet,
    build_graph_for_window,
    batch_graphs,
)


class TestForward:
    def test_forward_pass_shapes(self, small_model, small_config, random_inputs):
        L = small_config.backbone.n_anchor_bins
        B = random_inputs["acc_ctcf"].shape[0]

        # Build per-window graphs and batch them.
        edge_indices, edge_attrs = [], []
        for _ in range(B):
            ei, ea = build_graph_for_window(
                n_anchor_bins=L,
                coaccessibility=torch.rand(L, L),
                hic_prior=torch.rand(L, L) * 2.0,
                ctcf_orientation=torch.randint(-1, 2, (L,)),
                coacc_topk=3,
                hic_threshold=1.0,
            )
            edge_indices.append(ei)
            edge_attrs.append(ea)
        edge_index, edge_attr = batch_graphs(edge_indices, edge_attrs, L)

        out = small_model(
            **random_inputs,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        # Verify output shapes
        assert out["contact_map"].shape == (B, L, L)
        assert out["vstripe"].shape == (B, L, small_config.backbone.vstripe_length)
        assert out["compartment_logits"].shape == (B, L, 5)
        assert out["loop_anchor_logits"].shape == (B, L)
        assert out["insulation"].shape == (B, L)

    def test_contact_map_is_symmetric(self, small_model, small_config, random_inputs):
        L = small_config.backbone.n_anchor_bins
        B = random_inputs["acc_ctcf"].shape[0]
        edge_indices, edge_attrs = [], []
        for _ in range(B):
            ei, ea = build_graph_for_window(
                n_anchor_bins=L, coaccessibility=torch.rand(L, L),
                hic_prior=torch.rand(L, L), coacc_topk=3,
            )
            edge_indices.append(ei)
            edge_attrs.append(ea)
        edge_index, edge_attr = batch_graphs(edge_indices, edge_attrs, L)

        out = small_model(**random_inputs, edge_index=edge_index, edge_attr=edge_attr)
        cm = out["contact_map"]
        assert torch.allclose(cm, cm.transpose(-1, -2), atol=1e-5)

    def test_backward_pass(self, small_model, small_config, random_inputs):
        L = small_config.backbone.n_anchor_bins
        B = random_inputs["acc_ctcf"].shape[0]
        edge_indices, edge_attrs = [], []
        for _ in range(B):
            ei, ea = build_graph_for_window(
                n_anchor_bins=L, coaccessibility=torch.rand(L, L),
                hic_prior=torch.rand(L, L), coacc_topk=3,
            )
            edge_indices.append(ei)
            edge_attrs.append(ea)
        edge_index, edge_attr = batch_graphs(edge_indices, edge_attrs, L)

        out = small_model(**random_inputs, edge_index=edge_index, edge_attr=edge_attr)
        loss = (out["contact_map"].mean()
                + out["vstripe"].mean()
                + out["compartment_logits"].mean())
        loss.backward()

        # At least one parameter should have a non-trivial gradient.
        any_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in small_model.parameters()
        )
        assert any_grad


class TestNoGraph:
    """Verify the lightweight no-graph mode."""

    def test_works_without_graph(self, small_config, random_inputs):
        from chromagraphnet import ChromaGraphNet, ChromaGraphNetConfig

        cfg = ChromaGraphNetConfig()
        cfg.backbone = small_config.backbone
        cfg.fusion = small_config.fusion
        cfg.heads = small_config.heads
        cfg.modalities = small_config.modalities
        cfg.use_graph = False
        cfg.__post_init__()

        model = ChromaGraphNet(cfg)
        out = model(**random_inputs)
        assert "contact_map" in out


class TestUncertainty:
    def test_mc_dropout(self, small_model, small_config, random_inputs):
        L = small_config.backbone.n_anchor_bins
        B = random_inputs["acc_ctcf"].shape[0]
        edge_indices, edge_attrs = [], []
        for _ in range(B):
            ei, ea = build_graph_for_window(
                n_anchor_bins=L, coaccessibility=torch.rand(L, L),
                hic_prior=torch.rand(L, L), coacc_topk=3,
            )
            edge_indices.append(ei)
            edge_attrs.append(ea)
        edge_index, edge_attr = batch_graphs(edge_indices, edge_attrs, L)

        out = small_model.predict(
            **random_inputs,
            edge_index=edge_index,
            edge_attr=edge_attr,
            return_uncertainty=True,
            n_uncertainty_samples=3,
        )
        assert "contact_map_mean" in out
        assert "contact_map_std" in out
        assert (out["contact_map_std"] >= 0).all()
