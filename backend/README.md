# MoCap Studio Backend

Motion capture backend using SAM 3D Body for multi-person tracking with temporal consistency.

## Requirements

- **No conda needed!** Just uv and a CUDA-capable GPU
- CUDA 12.8+ (for GPU acceleration)
- [uv](https://github.com/astral-sh/uv) package manager (installs Python automatically)

## Quick Start
```bash
# 1. Install uv (it will handle Python 3.12 automatically)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and setup
git clone <your-repo>
cd mocap-studio/backend

# 3. One command to install everything (including Python 3.12!)
uv sync

# 4. Run the server
uv run uvicorn main:app --host 0.0.0.0 --port 8001
```

That's it! No conda, no manual Python installation needed.

## What uv Does For You

- ✅ Installs Python 3.12 automatically
- ✅ Creates isolated virtual environment
- ✅ Installs all dependencies including PyTorch, detectron2, pymomentum-gpu
- ✅ Handles CUDA packages correctly
- ✅ 10-100x faster than pip/conda

## Development
```bash
# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Format code
uv run black .

# Add a new dependency
uv add <package-name>
```

## Accessing HuggingFace Models

SAM 3D Body models are hosted on HuggingFace. You need to:
1. Request access on the model repos
2. Authenticate: `uv run huggingface-cli login`

Available models:
- `facebook/sam-3d-body-dinov3`
- `facebook/sam-3d-body-vith`

## Troubleshooting

**CUDA version mismatch?**
```bash
# Check your CUDA version
nvidia-smi

# For CUDA 12.4, update pyproject.toml extra-index-url to cu124
# For CUDA 12.8, it's already configured
```

**Need different Python version?**
```bash
# uv can install any Python version
uv python install 3.11
uv sync --python 3.11
```
