# Multi-stage Dockerfile for reproducible ChromaGraphNet inference.
# Default build uses CPU PyTorch. For GPU, pass --build-arg PYTORCH_VARIANT=cu121.

ARG PYTORCH_VARIANT=cpu
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch first to avoid solver issues.
ARG PYTORCH_VARIANT
RUN pip install --upgrade pip && \
    if [ "$PYTORCH_VARIANT" = "cpu" ]; then \
        pip install torch --index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip install torch --index-url https://download.pytorch.org/whl/$PYTORCH_VARIANT; \
    fi

RUN pip install torch_geometric

# Copy and install the project.
COPY pyproject.toml README.md LICENSE ./
COPY chromagraphnet ./chromagraphnet
RUN pip install -e .

# Default to printing model info; users can override the command.
ENTRYPOINT ["chromagraphnet-info"]
