#!/bin/bash
# Minimal setup for emotion-concepts-in-llms.
# Requires: uv (https://docs.astral.sh/uv/) and an NVIDIA GPU for local activation extraction.
set -euo pipefail

# Install core dependencies (CPU/GPU torch via the pinned CUDA index in pyproject.toml).
uv sync

# Copy the env template if you haven't already, then fill in your API keys.
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example — edit it to add your API keys."
fi

echo "Done. Core install ready."
echo "Optional extras:"
echo "  uv sync --extra local-inference   # local vLLM story generation"
