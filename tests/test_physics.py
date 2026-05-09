"""Tests for the polymer physics prior."""
from __future__ import annotations

import torch

from chromagraphnet import PolymerPhysicsPrior, PhysicsConfig


class TestPhysicsPrior:
    def test_all_terms_finite(self):
        cfg = PhysicsConfig(n_anchor_bins=40)
        prior = PolymerPhysicsPrior(cfg)
        B, L = 2, 40

        contact = torch.randn(B, L, L).abs() + 0.1
        compartment_logits = torch.randn(B, L, 5)
        ctcf = torch.randint(-1, 2, (B, L))

        losses = prior(contact, compartment_logits, ctcf)
        assert torch.isfinite(losses["distance"])
        assert torch.isfinite(losses["ctcf"])
        assert torch.isfinite(losses["compartment"])
        assert torch.isfinite(losses["total"])
        assert losses["total"] >= 0

    def test_gradients_flow(self):
        cfg = PhysicsConfig(n_anchor_bins=40)
        prior = PolymerPhysicsPrior(cfg)
        contact = (torch.randn(1, 40, 40).abs() + 0.1).requires_grad_(True)
        compartment_logits = torch.randn(1, 40, 5).requires_grad_(True)
        ctcf = torch.randint(-1, 2, (1, 40))

        losses = prior(contact, compartment_logits, ctcf)
        losses["total"].backward()
        # Gradient should flow back to the contact map.
        assert contact.grad is not None
