#!/bin/bash
# setup.bash

set -e  # exit on any error

echo "============================================"
echo "  GNR Project 1 — Environment Setup"
echo "============================================"

REPO_URL="https://github.com/vdsathvikreddy/Group_21_F_GNR.git"

# 1. Fetch repository into the CURRENT directory
echo ""
echo "[1/4] Fetching specific project folder into current directory..."

# Tell Git to completely ignore massive LFS files
export GIT_LFS_SKIP_SMUDGE=1

# Clone ONLY the skeleton of the repository (no files, no history)
git clone --no-checkout --depth 1 --sparse --filter=blob:none "$REPO_URL" .tmp_repo

cd .tmp_repo
# Tell Git we ONLY want the GNR_Project1 folder
git sparse-checkout set GNR_Project1
# Now actually download the files for just that folder
git checkout
cd ..

# Delete setup.bash from the temp folder so it doesn't overwrite the running script
rm -f .tmp_repo/GNR_Project1/setup.bash

# Copy ONLY the contents of the GNR_Project1 folder into the current directory
cp -r .tmp_repo/GNR_Project1/* ./

# Clean up the temporary folder
rm -rf .tmp_repo

echo "  Files successfully loaded into $(pwd)"

# 2. Create conda environment
echo ""
echo "[2/4] Creating conda env: gnr_project_env (Python 3.11)..."

# Accept Anaconda Terms of Service to prevent non-interactive crashes
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# Clean up the broken environment from the previous failed run
conda env remove -n gnr_project_env -y 2>/dev/null || true

# Install Python and pip via conda-forge
conda create --override-channels -c conda-forge -n gnr_project_env python=3.11 pip -y

# Initialize conda for the bash script
eval "$(conda shell.bash hook)"
conda activate gnr_project_env

# 3. Install dependencies
echo ""
echo "[3/4] Installing dependencies..."
python -m pip install --upgrade pip

# Install PyTorch specifically configured for the L40s GPU (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install paddlepaddle-gpu==2.6.1 paddleocr==2.8.1

# Install the rest of the pipeline dependencies (excluding paddle, which is now handled above)
pip install -r requirements.txt

# 4. Download Models
echo ""
echo "[4/4] Downloading offline model weights..."

echo "Downloading PaddleOCR models..."
python - <<'PYEOF'
import logging
from paddleocr import PaddleOCR

# Suppress verbose logging
logging.getLogger('ppocr').setLevel(logging.ERROR)

# Initialize with bare minimum. PaddleOCR will automatically download 
# the English models to ~/.paddleocr.
ocr = PaddleOCR(use_angle_cls=True, lang="en")
print("  -> PaddleOCR models downloaded to cache.")
PYEOF

# Download Qwen2-VL-7B weights (~15GB safetensors)
echo "Downloading Qwen2-VL-7B-Instruct weights..."
mkdir -p weights/qwen2-vl-7b

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen2-VL-7B-Instruct',
    local_dir='./weights/qwen2-vl-7b',
    ignore_patterns=['*.bin']
)
print('  -> Qwen weights downloaded successfully.')
"

echo ""
echo "============================================"
echo "  Setup complete! Ready for offline inference."
echo "============================================"