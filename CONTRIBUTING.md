# Contributing to ChromaGraphNet

Thanks for your interest in contributing. This document outlines the conventions used in the project.

## Setting up a development environment

```bash
git clone https://github.com/USERNAME/chromagraphnet.git
cd chromagraphnet
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
pip install -e ".[dev]"
pre-commit install
```

## Workflow

1. **Open an issue** before starting non-trivial work. This avoids parallel effort and lets us discuss the approach.
2. **Branch from `main`** with a descriptive name: `feat/neurohic-mode`, `fix/graph-builder-empty-input`, `docs/model-card-update`.
3. **Write tests** for new functionality. We aim for >85% coverage on `chromagraphnet/`.
4. **Run the full test suite** before submitting:
   ```bash
   ruff check .
   ruff format --check .
   pytest --cov=chromagraphnet
   ```
5. **Update the changelog** under `## [Unreleased]` describing your change.
6. **Open a pull request**. Reference the issue, describe the change, and include any relevant benchmark numbers.

## Commit messages

Use the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
feat(fusion): add learnable modality gates with sigmoid activation
fix(graph): handle empty hic_prior tensor without crashing
docs(readme): clarify CUDA install path
test(physics): add gradient flow check for compartment loss
chore(deps): bump torch_geometric to 2.7
```

Sign your commits: `git commit -s` (we follow the Developer Certificate of Origin).

## Code style

- **Formatter:** `black` (line length 88).
- **Linter:** `ruff` with the rules in `pyproject.toml`.
- **Type hints:** required on all new public functions; encouraged elsewhere.
- **Docstrings:** Google or NumPy style; explain *what* and *why*, not just *how*.

## Adding a new modality encoder

1. Add a new class in `chromagraphnet/models/modality_encoders.py` that follows the `(B, n_anchor_bins, embed_dim)` output convention.
2. Register it in `ModalityEncoderBank.__init__` and the `forward()` dispatch.
3. Add the corresponding flag to `ModalityEncoderConfig`.
4. Update `chromagraphnet/cli.py` to accept the new input key.
5. Add tests in `tests/test_modality_encoders.py`.
6. Document the input format in `docs/data_format.md`.

## Reporting bugs

Please include:

- Python and PyTorch versions.
- The full traceback.
- A minimal reproducible example (preferably with shapes printed).
- Whether the issue reproduces with `cfg.use_graph = False`.

## Code of Conduct

Be respectful and constructive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

## Questions

Open a GitHub Discussion or email the maintainers (see `pyproject.toml`).
