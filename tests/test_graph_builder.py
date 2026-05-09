"""Tests for graph construction utilities."""
from __future__ import annotations

import torch

from chromagraphnet import build_graph_for_window, batch_graphs


class TestGraphBuilder:
    def test_genomic_adjacency_only(self):
        ei, ea = build_graph_for_window(
            n_anchor_bins=10,
            coaccessibility=None,
            hic_prior=None,
            ctcf_orientation=None,
        )
        # Bidirectional adjacency edges: 2 * (L - 1)
        assert ei.shape == (2, 2 * 9)
        assert ea.shape == (2 * 9, 4)

    def test_with_coaccessibility(self):
        L = 20
        coacc = torch.rand(L, L)
        ei, ea = build_graph_for_window(
            n_anchor_bins=L,
            coaccessibility=coacc,
            coacc_topk=3,
        )
        # Should have at least the genomic adjacency edges.
        assert ei.shape[1] >= 2 * (L - 1)
        assert ea.shape[1] == 4

    def test_with_hic_prior_and_ctcf(self):
        L = 20
        hic = torch.rand(L, L) * 3.0
        ctcf = torch.randint(-1, 2, (L,))
        ei, ea = build_graph_for_window(
            n_anchor_bins=L,
            hic_prior=hic,
            ctcf_orientation=ctcf,
            hic_threshold=1.0,
        )
        assert ei.shape[0] == 2
        # Edge attribute 4 (CTCF convergence flag) should be 0 or 1.
        assert ((ea[:, 3] == 0) | (ea[:, 3] == 1)).all()

    def test_batch_graphs(self):
        L = 10
        graphs = []
        for _ in range(3):
            ei, ea = build_graph_for_window(
                n_anchor_bins=L,
                coaccessibility=torch.rand(L, L),
                coacc_topk=2,
            )
            graphs.append((ei, ea))
        eis = [g[0] for g in graphs]
        eas = [g[1] for g in graphs]
        big_ei, big_ea = batch_graphs(eis, eas, L)

        assert big_ei.shape[0] == 2
        assert big_ei.max() < 3 * L   # indices stay within the batched graph
        assert big_ea.shape[0] == big_ei.shape[1]
