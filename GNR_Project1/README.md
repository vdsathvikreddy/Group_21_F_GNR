# GNR Project 1 — Map Stitching & Spatial MCQ Answering

## Team
- Vijaya Raghavendra S, 23B1042
- V D Sathvik, 23B0906
- Ajay Pudi, 23B1010

## Pipeline Overview
This project solves the spatial map reconstruction and evaluation task using a 3-phase hybrid architecture:
1. **Computer Vision (Stitching):** Utilizes Perceptual Hashing and a Directed Acyclic Graph (DAG) to dynamically reconstruct the shredded map geometry without relying on arbitrary resizing.
2. **Spatial Indexing (OCR):** Passes the stitched canvas through PaddleOCR to extract text bounding boxes, creating a mathematical coordinate grid of place names.
3. **Hybrid VQA Engine:** Analyzes questions dynamically. It uses mathematical coordinate calculations for spatial questions (e.g., "closest to", "northernmost") and routes complex visual reasoning tasks to a localized instance of `Qwen2-VL-7B-Instruct`. A strict confidence threshold (0.20) is enforced to minimize the -0.25 incorrect answer penalty.

**Expected Runtime:** ~XX minutes on an NVIDIA L40s GPU (well within the 1-hour competition limit).

## File Structure
```
gnr_project1/
├── inference.py     # main script — run this
├── stitch.py        # Phase 1: patch stitching
├── ocr_index.py     # Phase 2: PaddleOCR spatial index
├── answering.py     # Phase 3: hybrid MCQ answering
├── setup.bash       # environment setup
├── requirements.txt 
├── README.md
└── weights/
    └── qwen2-vl-7b/ # downloaded by setup.bash (~15GB)
```

## Setup (run once, internet required)
Open your terminal in the root directory of this project and run:

```bash
bash setup.bash
```

## Run Inference (no internet required)
```bash
conda activate gnr_project_env
python inference.py --test_dir /absolute/path/to/test_dir
```
Note: The `--test_dir` argument should point to a folder containing both the `patches/` directory and the `test.csv` file.

Output: `./submission.csv`

## Dependencies
- Python 3.11
- opencv-python-headless, pillow, imagehash, numpy
- paddlepaddle-gpu, paddleocr
- torch, transformers>=4.45.0, accelerate, qwen-vl-utils
- pandas, rapidfuzz, huggingface_hub

## Citations
- Qwen2-VL: https://github.com/QwenLM/Qwen2-VL
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR
- imagehash: https://github.com/JohannesBuchner/imagehash
