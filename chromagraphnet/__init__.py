"""
ChromaGraphNet: a multi-modal graph-attention model for 3D genome
architecture prediction, built on the ChromaFold backbone.

Quick start:

    from chromagraphnet import ChromaGraphNet, ChromaGraphNetConfig

    model = ChromaGraphNet()
    out = model(acc_ctcf, coacc, rna=rna, chip=chip, motif=motif,
                edge_index=edge_index, edge_attr=edge_attr)

The output dict contains five tensors: contact_map, vstripe,
compartment_logits, loop_anchor_logits, insulation.

For inference with pretrained weights:

    from chromagraphnet.inference import load_model, predict_window
    model = load_model("checkpoints/chromagraphnet_v1.pt")
    out = predict_window(model, acc_ctcf, coacc, ...)
"""

__version__ = "0.1.1"

from .models.chromagraphnet import ChromaGraphNet, ChromaGraphNetConfig
from .models.chromafold_backbone import ChromaFoldBackbone, ChromaFoldConfig
from .models.modality_encoders import (
    ModalityEncoderBank,
    ModalityEncoderConfig,
)
from .models.fusion import CrossModalFusion, FusionConfig
from .models.gat_module import GraphAttentionModule, GraphConfig
from .models.output_heads import MultiTaskHeads, HeadsConfig
from .models.physics_prior import PolymerPhysicsPrior, PhysicsConfig
from .data.graph_builder import build_graph_for_window, batch_graphs
from .inference.predict import load_model, predict_window

__all__ = [
    "__version__",
    "ChromaGraphNet",
    "ChromaGraphNetConfig",
    "ChromaFoldBackbone",
    "ChromaFoldConfig",
    "ModalityEncoderBank",
    "ModalityEncoderConfig",
    "CrossModalFusion",
    "FusionConfig",
    "GraphAttentionModule",
    "GraphConfig",
    "MultiTaskHeads",
    "HeadsConfig",
    "PolymerPhysicsPrior",
    "PhysicsConfig",
    "build_graph_for_window",
    "batch_graphs",
    "load_model",
    "predict_window",
]
