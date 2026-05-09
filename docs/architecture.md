# ChromaGraphNet architecture

![ChromaGraphNet architecture](figures/chromagraphnet_architecture.png)

This document describes each module of ChromaGraphNet in the order data flows through the model. For the high-level summary and rationale, see the [main README](../README.md).

## Module 1: ChromaFold backbone

**File:** `chromagraphnet/models/chromafold_backbone.py`

A faithful PyTorch reimplementation of the two-branch CNN from Gao et al. (2024). The backbone produces a per-anchor-bin embedding that downstream modules consume.

### Branch 1: accessibility + CTCF

```
Input:  (B, 2, n_fine_bins)            n_fine_bins = 80,200 (4.01 Mb / 50 bp)
        channels = [pseudobulk scATAC, CTCF motif/ChIP]

  -> 15 x [Conv1D(k=7) + BatchNorm + ReLU]
  -> AvgPool(downsample 50 bp -> 10 kb, factor 200)
  -> Outer concatenation -> (B, 2C, L, L)
  -> 3 x [Conv2D(k=3) + BatchNorm + ReLU]
  -> Attention pool over flank axis
  -> Linear consolidator

Output: (B, n_anchor_bins, fused_dim)   n_anchor_bins = 401
```

The outer-concatenation trick (also used in Akita) lets the model represent every anchor-flank pair without quadratically growing the 1D feature map.

### Branch 2: co-accessibility V-stripe

```
Input:  (B, 40, n_coacc_bins)          n_coacc_bins = 8,020 (4.01 Mb / 500 bp)
        500 bp Jaccard co-accessibility V-stripe slice

  -> 3 x [Conv1D + BN + ReLU]
  -> 2 x ResidualBlock1d
  -> 3 x [Conv1D + BN + ReLU]
  -> AvgPool(downsample 500 bp -> 10 kb)
  -> Linear consolidator

Output: (B, n_anchor_bins, fused_dim)
```

### Linear fusion

```
fused = Linear([branch1; branch2])     # (B, L, 2D) -> (B, L, D)
```

This is the natural extension hook: `model.backbone.forward_features(acc_ctcf, coacc)` returns these fused features, which downstream modules refine.

## Module 2: Modality encoders

**File:** `chromagraphnet/models/modality_encoders.py`

Each modality has its own encoder that produces a `(B, n_anchor_bins, embed_dim)` tensor in the same embedding space as the backbone.

### scRNA encoder

```
Input:  (B, L, rna_in_dim)             per-bin pseudobulk expression
  -> Per-bin MLP (Linear -> GELU -> Dropout -> Linear)
  -> Conv1D(k=5) for local context
  -> LayerNorm
Output: (B, L, embed_dim)
```

### Histone ChIP encoder

```
Input:  (B, n_marks, n_fine_bins)      e.g. 5 histone tracks at 50 bp
  -> 4 x [Conv1D(k=7) + BN + GELU]
  -> AvgPool(50 bp -> 10 kb)
  -> Linear -> LayerNorm
Output: (B, L, embed_dim)
```

### TF motif encoder

```
Input:  (B, L, n_factors)              e.g. 200 TF motif scores per bin
  -> Linear projection
  -> Soft attention over a learnable per-factor embedding matrix
  -> GELU + Linear -> LayerNorm
Output: (B, L, embed_dim)
```

The factor-embedding trick lets related TFs (e.g. NPAS4 and FOS for activity-dependent loops) share representational capacity without being collapsed into a single channel.

## Module 3: Cross-modal fusion

**File:** `chromagraphnet/models/fusion.py`

Two-stage fusion:

### Stage 1: FiLM modulation

For each modality, generate per-bin (gamma, beta) and elementwise-modulate the backbone features:

```
gamma, beta = Linear(modality_embedding)    # initialized to zero, so identity at init
backbone' = backbone * (1 + gate * gamma) + (gate * beta)
```

The per-modality `gate` is a learnable scalar that allows the model to soft-prune unhelpful modalities without retraining.

### Stage 2: Bottleneck transformer (Perceiver-IO style)

```
latents = learnable parameter of shape (n_latents, D)         # default n_latents = 64

for layer in cross_blocks, self_blocks:
    latents = CrossAttention(q=latents,
                             kv=concat([backbone', rna, chip, motif]))
    latents = SelfAttention(latents)

decoded = CrossAttention(q=backbone', kv=latents)
output = LayerNorm(decoded + backbone')
```

Compute is bounded by the number of latents, irrespective of how many modalities are stacked. Adding a new modality is O(L * D) rather than O(L * D * n_modalities).

## Module 4: Graph attention

**File:** `chromagraphnet/models/gat_module.py`

The fused embeddings become node features on a chromatin graph.

### Graph construction

`chromagraphnet/data/graph_builder.py` constructs a graph with three edge types:

| Edge type             | Source                                 | Density per window |
|-----------------------|----------------------------------------|--------------------|
| Genomic adjacency    | (i, i+1) for all bins                  | 2 * (L - 1)        |
| Co-accessibility     | top-k Jaccard similarity per bin       | k * L              |
| Hi-C prior           | bins above HiC-DC+ Z-score threshold   | variable           |

Edge features are 4-dimensional: `[log1p(distance), log1p(coacc), log1p(hic), ctcf_convergent]`.

### GATv2 stack

```
for layer in 1..N:
    x = GATv2Conv(in=D, out=D//heads, heads=8, edge_dim=4, dropout=0.2)(x)
    x = ELU(x)
    x = LayerNorm(x)
    if x.shape == prev_x.shape: x = x + prev_x   # residual
```

We use GATv2 over GAT (Brody et al., 2022) for its strictly more expressive dynamic-attention mechanism, and pass continuous edge features through `edge_dim` so attention weights can read distance and contact-prior information directly.

## Module 5: Multi-task heads

**File:** `chromagraphnet/models/output_heads.py`

Five heads share the same fused embedding:

| Head                   | Output                                | Loss (recommended)     |
|------------------------|---------------------------------------|------------------------|
| Contact map            | (B, L, L) symmetric                   | MSE / Pearson loss     |
| V-stripe               | (B, L, vstripe_length) HiC-DC+ Z      | MSE                    |
| Compartments           | (B, L, 5) logits (A1/A2/B1/B2/B3)     | Cross-entropy          |
| Loop anchors           | (B, L) logits                         | Binary cross-entropy   |
| Insulation             | (B, L) continuous                     | MSE                    |

The contact and compartment heads use a Dropout layer that doubles as the MC-dropout estimator at inference time.

## Module 6: Polymer physics prior

**File:** `chromagraphnet/models/physics_prior.py`

Three soft regularizer terms added to the supervised loss during training:

1. **Distance-decay penalty.** Empirical contact P(s) should follow s^(-alpha) with alpha around 0.85. We compute the slope of log(P) vs log(s) and penalize squared deviation from -alpha.

2. **CTCF-convergence prior.** CTCF-bound bins only form loops with convergent partners (i forward, j reverse). We penalize predicted strong contacts at non-convergent CTCF pairs.

3. **Compartment phase-separation prior.** Within-compartment (A-A, B-B) contacts should exceed between-compartment (A-B) contacts by a configurable ratio.

These act as inductive biases, especially valuable when training data is sparse (the NeuroHi-C cfChromatin scenario).

## Putting it together

The full forward pass:

```python
def forward(acc_ctcf, coacc, rna, chip, motif, edge_index, edge_attr):
    # 1. Backbone
    backbone_feats = chromafold_backbone.forward_features(acc_ctcf, coacc)

    # 2. Modality encoders
    modalities = modality_bank(rna=rna, chip=chip, motif=motif)

    # 3. Cross-modal fusion
    fused = cross_modal_fusion(backbone_feats, modalities)

    # 4. Graph attention
    if use_graph:
        fused_flat = fused.reshape(B*L, D)
        refined = gat_module(fused_flat, edge_index, edge_attr)
        fused = refined.reshape(B, L, D)

    # 5. Multi-task heads
    return multi_task_heads(fused)
```

## Design choices and alternatives

| Decision                              | Choice                | Considered alternatives        |
|---------------------------------------|-----------------------|--------------------------------|
| Backbone reuse strategy               | Full fine-tune        | Freeze, partial fine-tune      |
| Graph attention type                  | GATv2                 | GAT, Graphormer, TransformerConv |
| Fusion mechanism                      | FiLM + bottleneck TF  | Concatenation only             |
| Compartment classes                   | 5 (Rao et al. 2014)   | 2 (A/B), continuous            |
| Resolution                            | 10 kb (5 kb head)     | 5 kb native, 25 kb (ChromaFold) |
| Uncertainty                           | MC dropout            | Deep ensembles, Bayesian NN    |

See [docs/design_decisions.md](design_decisions.md) for the full rationale (in preparation).
