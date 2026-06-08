#!/bin/bash
set -e

echo "==> Cloning repo..."
git clone https://github.com/Kanigzzz/finetune-medical.git
cd finetune-medical

echo "==> Fixing numpy..."
pip install "numpy<2" --force-reinstall -q

echo "==> Installing mlflow..."
pip install mlflow --ignore-installed blinker -q

echo "==> Installing ML libraries..."
pip install transformers peft datasets accelerate pyyaml --force-reinstall -q

echo "==> Upgrading PyTorch for CUDA..."
pip install "torch>=2.4" --index-url https://download.pytorch.org/whl/cu121 -q

echo "==> Preparing dataset..."
python -m src.data.prepare

echo "==> Starting training..."
python -m src.training.train

echo "==> Uploading mlruns to HuggingFace..."
  hf upload Kamil123456789/tinyllama-medical mlruns mlruns

echo "==> Done!"
