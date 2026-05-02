#!/bin/bash
# setup.bash

set -e  # exit on any error

echo "============================================"
echo "  GNR Project 1 — Environment Setup"
echo "============================================"

REPO_URL="https://github.com/vdsathvikreddy/Group_21_F_GNR/tree/main/GNR_Project1"

echo ""
echo "[1/4] Fetching repository files into current directory..."

git clone "$REPO_URL" .tmp_repo

cp -r .tmp_repo/* ./

rm -rf .tmp_repo

# 2. Create conda environment
echo ""
echo "[2/4] Creating conda env: gnr_project_env (Python 3.11)..."
conda create -n gnr_project_env python=3.11 -y

# Initialize conda for the bash script
eval "$(conda shell.bash hook)"
conda activate gnr_project_env

# 3. Install dependencies
echo ""
echo "[3/4] Installing dependencies..."
python -m pip install --upgrade pip

# Install PyTorch specifically configured for the L40s GPU (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install the rest of the pipeline dependencies
pip install -r requirements.txt

# Pre-download PaddleOCR models into ./weights/paddleocr/
echo "Downloading PaddleOCR models..."
python - <<'PYEOF'
from paddleocr import PaddleOCR
import os

# This triggers the download and caches all 3 models locally
os.makedirs("./weights/paddleocr", exist_ok=True)
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    use_gpu=False,   # CPU fine for download
    det_model_dir="./weights/paddleocr/det",
    rec_model_dir="./weights/paddleocr/rec",
    cls_model_dir="./weights/paddleocr/cls",
    show_log=False
)
print("PaddleOCR models downloaded.")
PYEOF

# 4. Download Qwen2-VL-7B weights (~15GB safetensors)
echo ""
echo "[4/4] Downloading Qwen2-VL-7B-Instruct weights..."
mkdir -p weights/qwen2-vl-7b

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen2-VL-7B-Instruct',
    local_dir='./weights/qwen2-vl-7b',
    ignore_patterns=['*.bin']
)
print('Weights downloaded successfully.')
"

echo ""
echo "============================================"
echo "  Setup complete! Ready for offline inference."
echo "============================================"