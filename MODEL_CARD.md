# Model Card: ChromaGraphNet

## Model Details

**Model name:** ChromaGraphNet
**Version:** 0.1.0
**Date:** 2026
**Type:** Multi-modal graph attention network for 3D genome architecture prediction
**Architecture:** ChromaFold backbone (frozen or fine-tuned) + modality encoders (RNA, ChIP, motif) + cross-modal fusion (FiLM + bottleneck transformer) + graph attention (GATv2 x 3) + multi-task heads
**Parameters:** ~3.5M at default config (4.01 Mb context, 401 anchor bins, embed_dim=128)
**License:** MIT
**Repository:** https://github.com/USERNAME/chromagraphnet

## Intended Use

### Primary intended uses

- **Research on 3D genome architecture** in cell types where Hi-C is unavailable or sparse, given scATAC-seq and optionally other modalities.
- **Cross-cell-type generalization studies**: predicting compartments, loops, and contact maps in held-out cell types using only accessibility and complementary data.
- **Hypothesis generation** for activity-dependent loop dynamics in neurons (with the forthcoming neuron-finetuned checkpoint).

### Primary intended users

- Computational biologists working on chromatin organization.
- Researchers building benchmarks against existing 3D genome predictors.
- Method developers extending the architecture for new modalities or species.

### Out-of-scope use cases

- **Clinical decision-making.** ChromaGraphNet is a research tool. Predictions must not be used to inform patient diagnosis, treatment, or genetic counseling.
- **Variant effect prediction without explicit fine-tuning.** The model has not been validated on personal genome inputs or rare-variant scenarios.
- **Reproducing experimental Hi-C in absentia.** Predictions are aggregations over the cell types in the training data; they cannot replace primary experimental measurements.

## Training Data

### v0.1 release: random initialization

The v0.1 checkpoint shipped with this repository is **randomly initialized**. It exists for testing the loading and inference pipeline, not for prediction.

### v0.2 release (in preparation)

The v0.2 release will include weights trained on:

- **Cell types:** IMR-90, GM12878, HUVEC (hg38), and downstream zero-shot evaluation on K562, HCT116, mouse cortical neurons (mm10).
- **Hi-C ground truth:** publicly available 4DN consortium high-resolution Hi-C, processed with HiC-DC+ to obtain Z-scores at 10 kb resolution.
- **scATAC ground truth:** matched scATAC-seq from the same cell types.
- **Optional modalities:** matched scRNA-seq, ENCODE ChIP-seq for the histone panel, and JASPAR-derived motif scores.

Training data will be enumerated with GEO accession numbers in the model card update accompanying v0.2.

### Neuron-finetuned checkpoint (in preparation)

A separate `chromagraphnet-neuro-v1` checkpoint will be fine-tuned on Bonev et al. (2017) cortical neuron Hi-C and matched epigenomic data, intended specifically for neuronal applications.

## Training Procedure

### Loss function

Multi-task weighted loss:

```
L_total = w_contact * MSE(contact_map, hic_dc_plus_z)
        + w_vstripe * MSE(vstripe, hic_dc_plus_z_vstripe)
        + w_compart * CE(compartment_logits, A1/A2/B1/B2/B3 labels)
        + w_loop    * BCE(loop_anchor_logits, loop_anchor_labels)
        + w_insul   * MSE(insulation, insulation_score)
        + lambda    * physics_prior
```

### Hyperparameters (default)

- Optimizer: AdamW, lr=1e-4, weight_decay=1e-4
- Scheduler: cosine annealing with linear warmup
- Batch size: 4 (4.01 Mb windows; effective batch size 16 with gradient accumulation)
- Training compute: ~1 V100 GPU-day for the base model

### Train / val / test split

Following ChromaFold conventions:

- Train: chr1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19
- Validation: chr3, chr15
- Test: chr5, chr18, chr20, chr21

This avoids genomic neighborhood leakage between sets.

## Evaluation

### Metrics

- **Distance-stratified Pearson correlation** between predicted and observed contact maps, binned by genomic distance.
- **AUROC / AUPRC on top-10% Z-score interactions** for loop detection.
- **Insulation-score Pearson correlation** for TAD boundary prediction.
- **Compartment classification accuracy and macro-F1** for the 5-way subtype prediction.

### Baselines

ChromaFold, C.Origami, Epiphany, GraphReg, HiCDiffusion (where applicable to the task).

### Quantitative results

Pending v0.2 release. The reproducible benchmark suite will be in `scripts/benchmark/`.

## Limitations

1. **Training data bias.** The base model is trained primarily on K562/IMR-90/GM12878-class cell lines. Predictions on highly divergent cell types (e.g., germ-line cells, non-mammalian species) are extrapolations.
2. **10 kb resolution ceiling.** Sub-loop-anchor resolution (1 kb) is not natively supported; the 5 kb head is upsampled from the 10 kb anchor grid using the GAT module's learned refinement.
3. **No allele-specific predictions.** The model predicts cell-type-average contact maps and cannot resolve allele-specific or single-cell heterogeneity.
4. **Modality dropout sensitivity.** Performance degrades when optional modalities are missing relative to training. Use the no-graph mode and the `chromafold` baseline path for fair comparisons under partial data.
5. **CTCF orientation requires external annotation.** The model expects user-provided CTCF orientation labels. Mis-oriented inputs will degrade the convergence prior.

## Bias and Risks

### Risks

- **Misuse for clinical inference.** As stated above, the model is not validated for clinical use. We include this warning prominently in the README and CLI output.
- **Misinterpretation of confidence.** MC-dropout uncertainty estimates are *epistemic* (model uncertainty), not aleatoric (data noise). Low std does not guarantee correctness.

### Bias

- **Cell-type representation bias.** Training data oversamples cell lines available on ENCODE/4DN. Underrepresented tissue types (placenta, tumor microenvironment cells, etc.) may yield worse predictions.
- **Species bias.** Primary training is on hg38; mm10 generalization is reasonable but degrades for non-mammalian targets.
- **Active vs. inactive compartment imbalance.** Heterochromatin contact prediction is harder than euchromatin due to sparse Hi-C signal; loss reweighting can partially mitigate.

## Ethical Considerations

ChromaGraphNet outputs predictions about endogenous biological structure, not human-attributed traits. The model does not predict disease risk, behavioral attributes, or any phenotype on individuals.

We follow the [AlphaGenome](https://deepmind.google/blog/alphagenome-ai-for-better-understanding-the-genome/) precedent of explicitly disclaiming clinical use and recommending validation by orthogonal experimental methods.

## Citation

```bibtex
@software{chromagraphnet2026,
  title  = {ChromaGraphNet: multi-modal graph-attention prediction of 3D genome architecture},
  author = {Roy and contributors},
  year   = {2026},
  url    = {https://github.com/USERNAME/chromagraphnet},
}
```

## Contact

Questions and issues: please open a GitHub issue at https://github.com/USERNAME/chromagraphnet/issues.

## License

MIT License (see [LICENSE](LICENSE)).
