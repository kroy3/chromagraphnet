"""
ChromaGraphNet: the full multi-modal 3D-genome model.

Pipeline:

    Inputs
        scATAC + CTCF, co-accessibility, scRNA, histone ChIP, motifs,
        genomic graph (edge_index, edge_attr).

    1. ChromaFold backbone -> per-bin embedding (B, L, D)
    2. Modality encoders   -> {rna: (B,L,D), chip: ..., motif: ...}
    3. Cross-modal fusion  -> per-bin fused embedding (B, L, D)
    4. Graph attention     -> refined embedding (B, L, D)
    5. Multi-task heads    -> contact map, compartments, loops, insulation

The class supports three forward modes:
    forward(...)                  -> dict of outputs
    forward_features(...)         -> per-bin fused embedding
    predict(...)                  -> outputs in eval mode
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .chromafold_backbone import ChromaFoldBackbone, ChromaFoldConfig
from .modality_encoders import ModalityEncoderBank, ModalityEncoderConfig
from .fusion import CrossModalFusion, FusionConfig
from .gat_module import GraphAttentionModule, GraphConfig
from .output_heads import MultiTaskHeads, HeadsConfig


@dataclass
class ChromaGraphNetConfig:
    backbone: ChromaFoldConfig = field(default_factory=ChromaFoldConfig)
    modalities: ModalityEncoderConfig = field(
        default_factory=ModalityEncoderConfig
    )
    fusion: FusionConfig = field(default_factory=FusionConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    heads: HeadsConfig = field(default_factory=HeadsConfig)

    # Whether the GAT module is included. Inference can disable it
    # for a lightweight ChromaFold-equivalent forward.
    use_graph: bool = True

    def __post_init__(self):
        # Propagate shared dimensions so the user only has to set them once.
        D = self.backbone.fused_dim
        L = self.backbone.n_anchor_bins
        self.modalities.embed_dim = D
        self.modalities.n_anchor_bins = L
        self.fusion.embed_dim = D
        self.fusion.n_anchor_bins = L
        self.graph.embed_dim = D
        # The graph layer's per-head dim should evenly divide embed_dim.
        if D % self.graph.n_heads != 0:
            raise ValueError(
                f"graph.n_heads ({self.graph.n_heads}) must divide "
                f"embed_dim ({D})."
            )
        self.graph.head_dim = D // self.graph.n_heads
        self.heads.embed_dim = D
        self.heads.n_anchor_bins = L
        self.heads.vstripe_length = self.backbone.vstripe_length


class ChromaGraphNet(nn.Module):
    """The full ChromaGraphNet model."""

    def __init__(self, cfg: ChromaGraphNetConfig | None = None):
        super().__init__()
        self.cfg = cfg or ChromaGraphNetConfig()

        self.backbone = ChromaFoldBackbone(self.cfg.backbone)
        self.modality_bank = ModalityEncoderBank(self.cfg.modalities)
        self.fusion = CrossModalFusion(self.cfg.fusion)
        if self.cfg.use_graph:
            self.graph = GraphAttentionModule(self.cfg.graph)
        else:
            self.graph = None
        self.heads = MultiTaskHeads(self.cfg.heads)

    # ---- Weight loading helpers ----

    def load_chromafold_weights(self, state_dict_path: str,
                                strict: bool = False) -> list[str]:
        """
        Partial-load ChromaFold pretrained weights into the backbone.

        Returns a list of unexpected keys (from the saved checkpoint) so
        the user can verify the mapping. Setting `strict=False` is the
        default because (a) the original weights live under different
        attribute names, and (b) we add a fused linear layer that is
        not present in ChromaFold.
        """
        sd = torch.load(state_dict_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

        # Translate ChromaFold attribute names -> ChromaGraphNet names.
        # The exact mapping depends on the upstream model definition;
        # users should run the included `tools/inspect_chromafold.py`
        # script first to print the keys and adjust.
        out = self.backbone.load_state_dict(sd, strict=strict)
        return list(out.unexpected_keys)

    # ---- Forward passes ----

    def forward_features(
        self,
        acc_ctcf: torch.Tensor,
        coacc: torch.Tensor,
        rna: torch.Tensor | None = None,
        chip: torch.Tensor | None = None,
        motif: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. backbone
        backbone_feats = self.backbone.forward_features(acc_ctcf, coacc)

        # 2. modality encoders
        modalities = self.modality_bank(rna=rna, chip=chip, motif=motif)

        # 3. cross-modal fusion
        fused = self.fusion(backbone_feats, modalities)

        # 4. graph attention (optional)
        if self.graph is not None:
            if edge_index is None:
                raise ValueError(
                    "edge_index must be provided when use_graph=True. "
                    "Build it with chromagraphnet.data.graph_builder."
                )
            B, L, D = fused.shape
            # Flatten to (B*L, D) and shift edge indices per batch element.
            x_flat = fused.reshape(B * L, D)
            offsets = torch.arange(B, device=fused.device).repeat_interleave(
                edge_index.shape[1] // B if edge_index.shape[1] % B == 0
                else 0
            )
            # The user is expected to provide a single batched edge_index
            # already shifted with PyG conventions, so just call the layer.
            refined = self.graph(x_flat, edge_index, edge_attr=edge_attr)
            fused = refined.reshape(B, L, D)

        return fused

    def forward(
        self,
        acc_ctcf: torch.Tensor,
        coacc: torch.Tensor,
        rna: torch.Tensor | None = None,
        chip: torch.Tensor | None = None,
        motif: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        feats = self.forward_features(
            acc_ctcf=acc_ctcf, coacc=coacc, rna=rna, chip=chip,
            motif=motif, edge_index=edge_index, edge_attr=edge_attr,
        )
        return self.heads(feats)

    @torch.no_grad()
    def predict(
        self,
        acc_ctcf: torch.Tensor,
        coacc: torch.Tensor,
        rna: torch.Tensor | None = None,
        chip: torch.Tensor | None = None,
        motif: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        return_uncertainty: bool = False,
        n_uncertainty_samples: int = 20,
    ) -> dict[str, torch.Tensor]:
        self.eval()
        feats = self.forward_features(
            acc_ctcf=acc_ctcf, coacc=coacc, rna=rna, chip=chip,
            motif=motif, edge_index=edge_index, edge_attr=edge_attr,
        )
        if return_uncertainty:
            return self.heads.predict_with_uncertainty(
                feats, n_samples=n_uncertainty_samples
            )
        return self.heads(feats)

    # ---- Useful introspection ----

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters()
                       if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
