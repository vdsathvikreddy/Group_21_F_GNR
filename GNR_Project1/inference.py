"""
inference.py — Main entry point for GNR Project 1.

Usage:
    python inference.py --test_dir <absolute_path_to_test_dir>

Reads:
    <test_dir>/patches/patch_*.png
    <test_dir>/test.csv

Writes:
    ./submission.csv  (current working directory, NOT inside test_dir)
"""
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"   # hard blocks any HF network call
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"
import argparse, os, time
import pandas as pd
import cv2

from stitch    import stitch_patches
from ocr_index import build_ocr_index
from answering import answer_question, load_qwen

# ── Paths ─────────────────────────────────────────────────────────────────────
# Model weights are downloaded by setup.bash into ./weights/qwen2-vl-7b
MODEL_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)),"weights", "qwen2-vl-7b")

OUTPUT_CSV = "./submission.csv"   # spec: must be in cwd, not test_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir", type=str, required=True,help="Absolute path to test directory")
    return p.parse_args()


def main():
    args = parse_args()
    test_dir    = args.test_dir
    patches_dir = os.path.join(test_dir, "patches")
    test_csv    = os.path.join(test_dir, "test.csv")

    print("="*60)
    print("GNR Project 1 — Map Stitching + MCQ Answering")
    print(f"test_dir  : {test_dir}")
    print(f"model_dir : {MODEL_DIR}")
    print("="*60)

    # Validate
    assert os.path.isdir(patches_dir), f"patches/ not found in {test_dir}"
    assert os.path.isfile(test_csv),   f"test.csv not found in {test_dir}"
    assert os.path.isdir(MODEL_DIR), (
        f"Model not found at {MODEL_DIR}\n"
        "Did you run setup.bash? It downloads weights to ./weights/qwen2-vl-7b/")

    # Phase 1: Stitch
    print("\n[Phase 1] Stitching patches...")
    t0 = time.time()
    canvas = stitch_patches(patches_dir, output_path="./stitched_map.png")
    print(f"Stitching done in {time.time()-t0:.1f}s  canvas={canvas.shape}")

    # Phase 2: OCR index
    print("\n[Phase 2] Building OCR spatial index...")
    t0 = time.time()
    ocr_df = build_ocr_index(canvas)
    print(f"OCR done in {time.time()-t0:.1f}s  ({len(ocr_df)} text regions)")

    # Phase 3: Load Qwen
    print("\n[Phase 3] Loading Qwen2-VL-7B...")
    model, processor = load_qwen(MODEL_DIR)

    # Phase 4: Answer all questions
    print("\n[Phase 4] Answering questions...")
    test_df = pd.read_csv(test_csv)
    print(f"Total questions: {len(test_df)}")

    results = []
    for i, row in test_df.iterrows():
        qid      = row.get("question_id",row.get("id"))
        question = row["question"]
        options  = [row["option_1"], row["option_2"],
                    row["option_3"], row["option_4"]]

        print(f"\n[{i+1}/{len(test_df)}] {qid}")
        print(f"  Q: {question[:80]}")

        final, conf, route = answer_question(
            question, options, canvas, ocr_df, model, processor)

        results.append({"id": qid, "question_num": qid, "option": final})

    # Phase 5: Write submission.csv
    sub_df = pd.DataFrame(results)[["id","question_num","option"]]
    sub_df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "="*60)
    print(f"submission.csv saved → {OUTPUT_CSV}")
    print(f"Attempted : {(sub_df['option']!=5).sum()}")
    print(f"Skipped   : {(sub_df['option']==5).sum()}")
    print()
    print(sub_df.to_string(index=False))
    print("="*60)


if __name__ == "__main__":
    main()
