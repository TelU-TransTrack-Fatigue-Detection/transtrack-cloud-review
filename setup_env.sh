#!/bin/bash
set -e

ENV_NAME="transtrack_review"
PYTHON_VERSION="3.11"

echo "============================================="
echo " TransTrack Cloud Review — Environment Setup"
echo "============================================="
echo

read -p "Do you have an NVIDIA GPU with CUDA? (y/n): " GPU_CHOICE
echo

echo "[1/4] Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)..."
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

echo
echo "[2/4] Installing core dependencies..."
conda run -n "$ENV_NAME" pip install -r requirements.txt

echo
echo "[3/4] Installing PyTorch..."
if [[ "$GPU_CHOICE" == "y" || "$GPU_CHOICE" == "Y" ]]; then
    echo "Installing GPU version (CUDA 12.1)..."
    conda run -n "$ENV_NAME" pip install -r requirements-torch-gpu.txt
else
    echo "Installing CPU version..."
    conda run -n "$ENV_NAME" pip install -r requirements-torch-cpu.txt
fi

echo
echo "[4/4] Installing dev dependencies (pytest)..."
conda run -n "$ENV_NAME" pip install -r requirements-dev.txt

echo
echo "============================================="
echo " Setup complete!"
echo
echo " Activate with:"
echo "   conda activate $ENV_NAME"
echo
echo " Then verify model:"
echo "   python scripts/check_model.py"
echo
echo " Run tests:"
echo "   pytest -v --ignore=tests/test_integration.py"
echo "============================================="
