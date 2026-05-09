# Changelog

All notable changes to ChromaGraphNet will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned for v0.2.0

- Pretrained weights on IMR-90, GM12878, HUVEC.
- Neuron-finetuned checkpoint (Bonev 2017 ground truth).
- Full training pipeline with Hydra config groups.
- Reproducible benchmark suite vs. ChromaFold, C.Origami, Epiphany.
- NeuroHi-C small-fragment inference mode.
- Reference data preprocessing scripts.

## [0.1.1] - 2026-05

### Changed

- Updated author metadata: corresponding author Roy (University of Houston, Department of Biology and Biochemistry) and second author Juboraj Roy Pavel (Old Dominion University, Department of Computer Science).
- Added `AUTHORS.md` with full affiliations and contribution roles.
- Updated `LICENSE` copyright statement, `CITATION.cff`, `pyproject.toml`, and `README.md` accordingly.

### Verified

- All 19 unit tests pass on the v0.1.1 codebase.
- CLI tools (`chromagraphnet-info`, `chromagraphnet-predict`) function correctly.
- Random-init checkpoint loading verified end-to-end.

## [0.1.0] - 2026-04

### Added

- Full ChromaGraphNet architecture: ChromaFold backbone, modality encoders (RNA, ChIP, motif), cross-modal fusion (FiLM + bottleneck transformer), graph attention (GATv2), multi-task heads, polymer physics prior.
- Random-initialized v0.1 checkpoint for testing the inference pipeline.
- CLI entry points: `chromagraphnet-predict`, `chromagraphnet-info`.
- Comprehensive test suite (19 tests covering forward, backward, MC-dropout uncertainty, graph construction, physics prior, and ChromaFold-only fallback).
- README, model card, architecture and data-format documentation.
- pyproject.toml with optional extras (genomics, viz, dev, docs).
- Conda environment file and Dockerfile.
- GitHub Actions CI workflow.
