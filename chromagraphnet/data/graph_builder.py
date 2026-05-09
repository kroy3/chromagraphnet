"""
Graph builder utilities.

Constructs a PyTorch-Geometric edge_index + edge_attr tensor pair from
three sources of evidence:

    1. Genomic adjacency: (i, i+1) edges for the linear genome.
    2. Co-accessibility: top-k Jaccard similarity between bins.
    3. Hi-C prior: pairs with HiC-DC+ Z-score above a threshold.

Edge features are 4-dimensional:

    [log1p(genomic distance in bins),
     log1p(co-accessibility score),
     log1p(Hi-C prior score),
     CTCF convergent-orientation flag]

For batches, edges across multiple windows are concatenated and node
indices are offset by `B_i * n_anchor_bins` for the i-th batch element,
following standard PyG conventions.
"""
from __future__ import annotations

import torch


def build_graph_for_window(
    n_anchor_bins: int,
    coaccessibility: torch.Tensor | None = None,    # (L, L) or None
    hic_prior: torch.Tensor | None = None,          # (L, L) or None
    ctcf_orientation: torch.Tensor | None = None,   # (L,) integer in {-1,0,+1}
    coacc_topk: int = 10,
    hic_threshold: float = 1.0,
    add_genomic_adjacency: bool = True,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a single-window PyG graph.

    Returns:
        edge_index: (2, E) long tensor.
        edge_attr:  (E, 4) float tensor.
    """
    L = n_anchor_bins
    src_list: list[torch.Tensor] = []
    dst_list: list[torch.Tensor] = []
    feat_list: list[torch.Tensor] = []

    # 1. Genomic adjacency
    if add_genomic_adjacency:
        idx = torch.arange(L - 1, device=device)
        src = torch.cat([idx, idx + 1])
        dst = torch.cat([idx + 1, idx])
        src_list.append(src)
        dst_list.append(dst)
        e_dist = torch.zeros_like(src, dtype=torch.float)   # log1p(1) = 0.69
        feats = torch.zeros((src.numel(), 4), device=device)
        feats[:, 0] = torch.log1p(torch.ones_like(e_dist))
        feat_list.append(feats)

    # 2. Co-accessibility top-k per node
    if coaccessibility is not None:
        coacc = coaccessibility.to(device)
        # Zero the diagonal so we don't pick self-edges.
        coacc = coacc - torch.diag(torch.diag(coacc))
        topv, topi = torch.topk(coacc, k=min(coacc_topk, L - 1), dim=-1)
        for i in range(L):
            for j_idx, j in enumerate(topi[i].tolist()):
                if topv[i, j_idx] <= 0:
                    continue
                src_list.append(torch.tensor([i], device=device))
                dst_list.append(torch.tensor([j], device=device))
                gd = abs(i - j)
                feats = torch.tensor([[
                    torch.log1p(torch.tensor(float(gd))).item(),
                    torch.log1p(topv[i, j_idx]).item(),
                    0.0,
                    0.0,
                ]], device=device)
                feat_list.append(feats)

    # 3. Hi-C prior
    if hic_prior is not None:
        hic = hic_prior.to(device)
        mask = hic >= hic_threshold
        # Drop the diagonal to avoid self-loops.
        diag_idx = torch.arange(L, device=device)
        mask[diag_idx, diag_idx] = False
        idx_pairs = mask.nonzero(as_tuple=False)
        if idx_pairs.numel() > 0:
            src = idx_pairs[:, 0]
            dst = idx_pairs[:, 1]
            src_list.append(src)
            dst_list.append(dst)
            gd = (src - dst).abs().float()
            feats = torch.zeros((src.numel(), 4), device=device)
            feats[:, 0] = torch.log1p(gd)
            feats[:, 2] = torch.log1p(hic[src, dst])
            if ctcf_orientation is not None:
                ori = ctcf_orientation.to(device)
                # Convergent if src has +1 and dst has -1 (or vice versa).
                conv = ((ori[src] == 1) & (ori[dst] == -1)) | (
                    (ori[src] == -1) & (ori[dst] == 1)
                )
                feats[:, 3] = conv.float()
            feat_list.append(feats)

    if not src_list:
        # Empty graph fallback.
        return (
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty((0, 4), dtype=torch.float, device=device),
        )

    src = torch.cat(src_list).long()
    dst = torch.cat(dst_list).long()
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = torch.cat(feat_list, dim=0)
    return edge_index, edge_attr


def batch_graphs(
    edge_indices: list[torch.Tensor],
    edge_attrs: list[torch.Tensor],
    n_anchor_bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Concatenate per-window graphs into a single batched graph by shifting
    node indices for each window.
    """
    shifted_indices = []
    for i, ei in enumerate(edge_indices):
        shifted_indices.append(ei + i * n_anchor_bins)
    return (
        torch.cat(shifted_indices, dim=1),
        torch.cat(edge_attrs, dim=0),
    )
