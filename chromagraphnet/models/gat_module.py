"""
Graph attention module.

Operates on a graph whose nodes are 10 kb anchor bins and whose edges
encode three relation types:

    1. Genomic-adjacency edges: (i, i+1) for all bins.
    2. Co-accessibility edges: top-k pairs by Jaccard similarity.
    3. Hi-C-prior edges: pairs with HiC-DC+ Z-score above a threshold.

Edge features are continuous and small:

    [log(genomic distance + 1),
     log(co-accessibility + 1),
     log(Hi-C prior + 1),
     CTCF convergent-orientation flag]

The module stacks GATv2Conv layers with residual connections and layer
normalization. We use GATv2 over GAT (Brody et al., ICLR 2022) for its
strictly more expressive dynamic-attention mechanism, and we pass edge
features through the `edge_dim` argument so the attention mechanism can
read distance and contact-prior information directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv
    _HAS_PYG = True
except ImportError:  # pragma: no cover
    GATv2Conv = None
    _HAS_PYG = False


@dataclass
class GraphConfig:
    embed_dim: int = 128
    n_layers: int = 3
    n_heads: int = 8
    head_dim: int = 16            # 8 heads * 16 = 128 (same as embed_dim)
    edge_dim: int = 4             # see module docstring
    dropout: float = 0.2
    use_residual: bool = True
    use_layernorm: bool = True


class GraphAttentionModule(nn.Module):
    """
    Stack of GATv2 layers refining per-bin embeddings using the
    chromatin contact graph.
    """

    def __init__(self, cfg: GraphConfig):
        super().__init__()
        if not _HAS_PYG:
            raise ImportError(
                "torch_geometric is required for GraphAttentionModule. "
                "Install with: pip install torch_geometric"
            )
        self.cfg = cfg

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_dim = cfg.embed_dim

        for i in range(cfg.n_layers):
            # Concatenate heads on all but the last layer.
            concat = (i < cfg.n_layers - 1)
            out_dim_per_head = cfg.head_dim
            self.layers.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=out_dim_per_head,
                    heads=cfg.n_heads,
                    concat=concat,
                    dropout=cfg.dropout,
                    edge_dim=cfg.edge_dim,
                    add_self_loops=True,
                )
            )
            in_dim = (out_dim_per_head * cfg.n_heads
                      if concat else out_dim_per_head)
            self.norms.append(nn.LayerNorm(in_dim))

        # Project final output back to embed_dim if necessary.
        if in_dim != cfg.embed_dim:
            self.out_proj = nn.Linear(in_dim, cfg.embed_dim)
        else:
            self.out_proj = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, embed_dim) node features (concatenated batch graph).
            edge_index: (2, E) edges in COO format.
            edge_attr:  (E, edge_dim) edge features.
        Returns:
            (N, embed_dim) refined node features.
        """
        for layer, norm in zip(self.layers, self.norms):
            h = layer(x, edge_index, edge_attr=edge_attr)
            h = F.elu(h)
            h = norm(h)
            if self.cfg.use_residual and h.shape == x.shape:
                h = h + x
            x = h
        return self.out_proj(x)
