"""
Cross-modal fusion module.

Combines the ChromaFold backbone embedding with the modality-specific
embeddings (RNA, ChIP, motif) using a two-stage strategy:

    Stage 1 (FiLM):
        Each modality produces a per-bin (gamma, beta) pair that
        elementwise-modulates the backbone embedding. This is the
        cheapest way to inject auxiliary signal without disturbing
        the pretrained ChromaFold weights.
        Reference: Perez et al., "FiLM" (AAAI 2018).

    Stage 2 (Bottleneck transformer):
        A small set of learnable latent tokens cross-attends to the
        full stack of (backbone + modality) embeddings, then
        self-attention runs across the latents. This is the
        Perceiver-IO trick: bounded compute irrespective of how
        many modalities you stack.
        Reference: Jaegle et al., "Perceiver IO" (ICLR 2022).

The output is a fused per-bin embedding ready for the graph attention
module.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FusionConfig:
    embed_dim: int = 128
    n_anchor_bins: int = 401

    # FiLM
    film_use: bool = True

    # Bottleneck transformer
    n_latents: int = 64
    n_heads: int = 8
    n_layers: int = 2
    ffn_mult: int = 2
    dropout: float = 0.1

    # Modality gating in the FiLM stage
    learnable_modality_gates: bool = True


class _FiLMBlock(nn.Module):
    """Per-modality FiLM generator: modality embedding -> (gamma, beta)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.to_gamma_beta = nn.Linear(embed_dim, 2 * embed_dim)
        # Initialize so that gamma starts near 1 and beta near 0
        # (i.e., modulation is the identity at initialization).
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        # x:   (B, L, D)  backbone features being modulated
        # mod: (B, L, D)  modality features producing gamma/beta
        gb = self.to_gamma_beta(mod)           # (B, L, 2D)
        gamma, beta = gb.chunk(2, dim=-1)
        return x * (1.0 + gamma) + beta        # identity at init


class _CrossAttentionBlock(nn.Module):
    """One block of (cross-attn + FFN) for the bottleneck transformer."""

    def __init__(self, embed_dim: int, n_heads: int, ffn_mult: int,
                 dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_ff = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_mult * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * embed_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv),
                         need_weights=False)
        q = q + self.dropout(h)
        q = q + self.ffn(self.norm_ff(q))
        return q


class _SelfAttentionBlock(nn.Module):
    """Self-attention block on the latent tokens."""

    def __init__(self, embed_dim: int, n_heads: int, ffn_mult: int,
                 dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_mult * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * embed_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                         need_weights=False)
        x = x + self.dropout(h)
        x = x + self.ffn(self.norm2(x))
        return x


class CrossModalFusion(nn.Module):
    """
    Combine the ChromaFold backbone embedding with optional modality
    embeddings (RNA, ChIP, motif).

    Forward signature:
        backbone:  (B, L, D)            required
        modalities: dict[str, (B, L, D)] optional
    Returns:
        (B, L, D) fused per-bin embedding.
    """

    def __init__(self, cfg: FusionConfig,
                 modality_names: tuple[str, ...] = ("rna", "chip", "motif")):
        super().__init__()
        self.cfg = cfg
        self.modality_names = modality_names

        # FiLM blocks: one per modality.
        if cfg.film_use:
            self.film_blocks = nn.ModuleDict({
                name: _FiLMBlock(cfg.embed_dim) for name in modality_names
            })
        else:
            self.film_blocks = None

        # Learnable per-modality gates (default 1.0 each), so the model
        # can soft-prune unhelpful modalities.
        if cfg.learnable_modality_gates:
            self.modality_gates = nn.ParameterDict({
                name: nn.Parameter(torch.ones(()))
                for name in modality_names
            })
        else:
            self.modality_gates = None

        # Bottleneck transformer: latent tokens cross-attend to the
        # concatenated (backbone + modality) sequence, then self-attend.
        self.latents = nn.Parameter(
            torch.randn(cfg.n_latents, cfg.embed_dim) * 0.02
        )
        self.cross_blocks = nn.ModuleList([
            _CrossAttentionBlock(cfg.embed_dim, cfg.n_heads,
                                 cfg.ffn_mult, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.self_blocks = nn.ModuleList([
            _SelfAttentionBlock(cfg.embed_dim, cfg.n_heads,
                                cfg.ffn_mult, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])

        # Decode latents back to per-bin features via cross-attention
        # whose queries are the original backbone embedding (preserving
        # spatial alignment).
        self.decoder = _CrossAttentionBlock(
            cfg.embed_dim, cfg.n_heads, cfg.ffn_mult, cfg.dropout
        )

        self.out_norm = nn.LayerNorm(cfg.embed_dim)

    def forward(
        self,
        backbone: torch.Tensor,
        modalities: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        modalities = modalities or {}
        x = backbone

        # ---- Stage 1: FiLM modulation per modality ----
        if self.film_blocks is not None:
            for name in self.modality_names:
                if name in modalities:
                    mod = modalities[name]
                    gate = (self.modality_gates[name]
                            if self.modality_gates is not None else 1.0)
                    x = x + gate * (self.film_blocks[name](x, mod) - x)

        # ---- Stage 2: bottleneck transformer ----
        # Build the key/value sequence by concatenating backbone + each
        # modality along the sequence dimension. (B, L * (1 + n_mod), D)
        kv_chunks = [x]
        for name in self.modality_names:
            if name in modalities:
                kv_chunks.append(modalities[name])
        kv = torch.cat(kv_chunks, dim=1)

        B = x.shape[0]
        latents = self.latents.unsqueeze(0).expand(B, -1, -1).contiguous()

        for cross, self_attn in zip(self.cross_blocks, self.self_blocks):
            latents = cross(latents, kv)
            latents = self_attn(latents)

        # Decode back to per-bin features using x as queries.
        decoded = self.decoder(x, latents)
        return self.out_norm(decoded + x)
