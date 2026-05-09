"""
Polymer physics prior.

Acts as a soft regularizer during training. Implements three loss terms
that penalize physically implausible contact predictions:

    1. Distance-decay penalty:
       Predicted contacts should follow P(s) ~ s^(-alpha) where alpha is
       cell-type-specific (~0.75 for active compartments, ~1.0 for
       inactive). We compute the empirical scaling of the predicted map
       and penalize deviation from a target alpha.

    2. CTCF-convergence prior:
       Loop anchors should preferentially involve convergent CTCF pairs.
       Given a binary CTCF orientation track (+1 forward, -1 reverse, 0
       absent), we penalize predicted strong contacts at non-convergent
       sites.

    3. Compartment phase-separation prior:
       Same-type compartments (A-A, B-B) should contact each other more
       than across-type (A-B). We compute the within- vs. between-type
       contact ratio in the prediction and add a hinge loss if the
       within/between ratio falls below a threshold.

These terms are combined into a single scalar `physics_loss` that is
added to the supervised reconstruction loss with a tunable weight.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PhysicsConfig:
    target_alpha: float = 0.85          # P(s) ~ s^(-alpha)
    distance_weight: float = 1.0
    ctcf_weight: float = 0.5
    compartment_weight: float = 0.5
    min_separation_bins: int = 2        # ignore the diagonal+adjacent bins
    n_anchor_bins: int = 401
    bin_size_bp: int = 10_000


class PolymerPhysicsPrior(nn.Module):
    """Soft regularizer combining the three physics terms above."""

    def __init__(self, cfg: PhysicsConfig | None = None):
        super().__init__()
        self.cfg = cfg or PhysicsConfig()

        # Precompute genomic-distance matrix in units of bins.
        bins = torch.arange(self.cfg.n_anchor_bins)
        d = (bins.unsqueeze(0) - bins.unsqueeze(1)).abs().float()
        self.register_buffer("distance_bins", d)

    # ----- Term 1: distance decay -----

    def distance_decay_loss(self, contact_map: torch.Tensor) -> torch.Tensor:
        """
        contact_map: (B, L, L) predicted contact strength
                     (assumed log-scale or Z-score).

        We bin pairs by genomic distance, compute mean predicted contact
        per distance bin, fit log(P) vs log(s), and penalize deviation
        of the slope from -target_alpha.
        """
        B, L, _ = contact_map.shape
        d = self.distance_bins.to(contact_map.device)
        mask = d >= self.cfg.min_separation_bins   # (L, L)

        # Use log distance, log mean contact at each distance.
        max_d = int(d.max().item())
        log_d = []
        log_p = []
        for s in range(self.cfg.min_separation_bins, max_d + 1):
            sel = (d == s) & mask
            if not sel.any():
                continue
            mean_c = contact_map[:, sel].mean()
            # Skip pathological values (e.g., negative log-contacts).
            if torch.isfinite(mean_c) and mean_c > 0:
                log_d.append(torch.log(torch.tensor(float(s),
                                                    device=mean_c.device)))
                log_p.append(torch.log(mean_c + 1e-6))
        if len(log_d) < 5:
            return contact_map.new_zeros(())

        log_d_t = torch.stack(log_d)
        log_p_t = torch.stack(log_p)
        # Closed-form least-squares slope.
        x_mean = log_d_t.mean()
        y_mean = log_p_t.mean()
        num = ((log_d_t - x_mean) * (log_p_t - y_mean)).sum()
        den = ((log_d_t - x_mean) ** 2).sum() + 1e-8
        slope = num / den
        # Penalize squared deviation from the target slope.
        return (slope + self.cfg.target_alpha) ** 2

    # ----- Term 2: CTCF convergence -----

    def ctcf_convergence_loss(
        self,
        contact_map: torch.Tensor,
        ctcf_orientation: torch.Tensor,
    ) -> torch.Tensor:
        """
        contact_map:      (B, L, L)
        ctcf_orientation: (B, L) integer in {-1, 0, +1}

        A convergent pair is (i, j) with i < j, orientation[i]=+1,
        orientation[j]=-1. Strong predicted contacts at non-convergent
        CTCF-bound pairs incur a penalty.
        """
        B, L, _ = contact_map.shape
        f = (ctcf_orientation == 1).float()       # (B, L) forward
        r = (ctcf_orientation == -1).float()      # (B, L) reverse

        convergent = f.unsqueeze(2) * r.unsqueeze(1)            # (B, L, L)
        any_ctcf = (ctcf_orientation != 0).float()              # (B, L)
        any_pair = any_ctcf.unsqueeze(2) * any_ctcf.unsqueeze(1)  # (B, L, L)
        non_convergent = any_pair * (1.0 - convergent)

        if non_convergent.sum() < 1.0:
            return contact_map.new_zeros(())

        # Hinge: penalize positive contact predictions at non-convergent CTCF pairs.
        hinge = F.relu(contact_map) * non_convergent
        return hinge.sum() / (non_convergent.sum() + 1e-6)

    # ----- Term 3: compartment phase separation -----

    def compartment_loss(
        self,
        contact_map: torch.Tensor,
        compartment_logits: torch.Tensor,
        within_to_between_target: float = 1.5,
    ) -> torch.Tensor:
        """
        Encourage within-compartment contacts to exceed between-compartment
        contacts by `within_to_between_target` ratio.
        """
        B, L, _ = contact_map.shape
        # Hard-assign compartments by argmax (no gradient through argmax;
        # this is a structural prior, not a label loss).
        with torch.no_grad():
            compartments = compartment_logits.argmax(dim=-1)     # (B, L)
            same = (compartments.unsqueeze(2)
                    == compartments.unsqueeze(1)).float()        # (B, L, L)

        # Off-diagonal mask
        d = self.distance_bins.to(contact_map.device)
        mask = (d >= self.cfg.min_separation_bins).float()

        within_mean = ((contact_map * same * mask).sum()
                       / (same * mask).sum().clamp(min=1.0))
        between_mean = ((contact_map * (1 - same) * mask).sum()
                        / ((1 - same) * mask).sum().clamp(min=1.0))

        ratio = within_mean / (between_mean.abs() + 1e-6)
        return F.relu(within_to_between_target - ratio)

    # ----- Combined loss -----

    def forward(
        self,
        contact_map: torch.Tensor,
        compartment_logits: torch.Tensor | None = None,
        ctcf_orientation: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        out["distance"] = self.distance_decay_loss(contact_map)
        if ctcf_orientation is not None:
            out["ctcf"] = self.ctcf_convergence_loss(contact_map,
                                                    ctcf_orientation)
        else:
            out["ctcf"] = contact_map.new_zeros(())
        if compartment_logits is not None:
            out["compartment"] = self.compartment_loss(contact_map,
                                                      compartment_logits)
        else:
            out["compartment"] = contact_map.new_zeros(())

        out["total"] = (
            self.cfg.distance_weight * out["distance"]
            + self.cfg.ctcf_weight * out["ctcf"]
            + self.cfg.compartment_weight * out["compartment"]
        )
        return out
