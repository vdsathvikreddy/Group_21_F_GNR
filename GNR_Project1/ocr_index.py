"""
ocr_index.py
Runs PaddleOCR on the stitched map canvas.
Builds a spatial DataFrame: text | bbox | center | confidence
Used by answering.py for lookup and spatial routing.
"""
import os
import cv2
import numpy as np
import pandas as pd
import logging

def build_ocr_index(canvas_bgr, verbose=True):
    """
    Run PaddleOCR on the stitched map.
    Returns DataFrame with columns:
        text, text_lower, x_min, y_min, x_max, y_max, cx, cy, conf
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        raise ImportError("Run: pip install paddleocr")

    if verbose:
        print("Running PaddleOCR...")

    # Suppress PaddleOCR's verbose logging
    logging.getLogger('ppocr').setLevel(logging.ERROR)

    canvas_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
    
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang="en",
        use_gpu=False
    )
    
    results = ocr.ocr(canvas_rgb, cls=True)

    rows = []
    if results and results[0]:
        for line in results[0]:
            box, (text, conf) = line[0], line[1]
            xs = [pt[0] for pt in box]; ys = [pt[1] for pt in box]
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))
            rows.append({
                "text":       text,
                "text_lower": text.lower().strip(),
                "x_min": x_min, "y_min": y_min,
                "x_max": x_max, "y_max": y_max,
                "cx":  (x_min+x_max)//2,
                "cy":  (y_min+y_max)//2,
                "conf": float(conf)
            })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["text","text_lower","x_min","y_min","x_max","y_max","cx","cy","conf"])
    if verbose:
        print(f"OCR found {len(df)} text regions")
    return df


def find_text(ocr_df, query, top_k=3, min_conf=0.4):
    """Fuzzy search for a place name. Returns top-k matching rows."""
    if ocr_df is None or len(ocr_df) == 0:
        return pd.DataFrame()
    try:
        from rapidfuzz import fuzz
    except ImportError:
        raise ImportError("Run: pip install rapidfuzz")
    q = query.lower().strip()
    scores = ocr_df["text_lower"].apply(lambda t: fuzz.partial_ratio(q, t)/100.0)
    mask = (scores > 0.4) & (ocr_df["conf"] >= min_conf)
    result = ocr_df[mask].copy()
    result["match_score"] = scores[mask]
    return result.sort_values("match_score", ascending=False).head(top_k)


def get_location(ocr_df, name, min_conf=0.4):
    """
    Returns (cx, cy, match_score) for a named place, or None if not found.
    """
    hits = find_text(ocr_df, name, top_k=1, min_conf=min_conf)
    if len(hits) == 0:
        return None
    row = hits.iloc[0]
    return int(row["cx"]), int(row["cy"]), float(row["match_score"])