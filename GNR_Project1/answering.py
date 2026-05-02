"""
answering.py
Hybrid MCQ answering engine with 3 routes:
  LOOKUP  — option text found directly in OCR index
  SPATIAL — coordinate math (near/north/south/east/west)
  VISUAL  — Qwen2-VL-7B on a cropped map region

Confidence threshold = 0.20 (break-even for -0.25 penalty scoring).
"""

import re, json
import numpy as np
import cv2
import torch
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.20
CROP_SIZE            = 600

# Keywords that indicate a RELATIVE spatial reference (e.g. "north of X")
# These ALWAYS override LOOKUP — a question can say "shown north of X"
# and it must be treated as SPATIAL not LOOKUP
SPATIAL_REF_PATTERNS = [
    r"near\s+\w",           # "near the airport", "near Powai"
    r"closest to",
    r"nearest to",
    r"north of",            # "north of Powai Lake" → relative
    r"south of",
    r"east of",
    r"west of",
    r"adjacent to",
    r"next to",
    r"in the north",        # "shown in the north of the map"
    r"in the south",
    r"in the east",
    r"in the west",
]

# General spatial keywords (no reference location)
SPATIAL_KEYWORDS = [
    "furthest north", "farthest north", "northernmost", "most north",
    "furthest south", "farthest south", "southernmost", "most south",
    "furthest east",  "farthest east",  "easternmost",  "most east",
    "furthest west",  "farthest west",  "westernmost",  "most west",
    "furthest", "farthest", "distance", "km", "kilometre", "kilometer",
]

# LOOKUP keywords — only used when NO spatial pattern detected
LOOKUP_KEYWORDS = [
    "visible", "shown", "present", "exist", "found in", "seen",
    "appears", "marked", "indicated", "labeled", "labelled"
]


# ── Routing ───────────────────────────────────────────────────────────────────
def route_question(question):
    """
    Priority order:
      1. Spatial with reference ("near X", "north of X") → SPATIAL always wins
      2. General spatial superlatives ("furthest north") → SPATIAL
      3. Lookup ("visible", "shown") → LOOKUP
      4. Everything else → VISUAL
    """
    q = question.lower()

    # Priority 1: spatial reference patterns beat everything including LOOKUP
    if any(re.search(p, q) for p in SPATIAL_REF_PATTERNS):
        return "spatial"

    # Priority 2: superlative spatial (no reference needed)
    if any(k in q for k in SPATIAL_KEYWORDS):
        return "spatial"

    # Priority 3: pure lookup (visible/shown with no spatial reference)
    if any(re.search(k, q) for k in LOOKUP_KEYWORDS):
        return "lookup"

    return "visual"


# ── Route 1: LOOKUP ───────────────────────────────────────────────────────────
def answer_lookup(options, ocr_df):
    from ocr_index import find_text
    best_opt, best_conf = None, 0.0
    for i, opt in enumerate(options):
        hits = find_text(ocr_df, opt, top_k=1, min_conf=0.3)
        if len(hits) > 0:
            score = float(hits.iloc[0]["match_score"])
            if score > best_conf:
                best_conf = score; best_opt = i+1
    return (best_opt, best_conf) if best_opt else (5, 0.0)


# ── Route 2: SPATIAL ──────────────────────────────────────────────────────────
def answer_spatial(question, options, ocr_df):
    from ocr_index import get_location
    q = question.lower()

    # Locate each option in the OCR index
    locs = {}
    for i, opt in enumerate(options):
        loc = get_location(ocr_df, opt)
        if loc:
            locs[i+1] = {"cx": loc[0], "cy": loc[1], "match": loc[2]}

    if not locs:
        return 5, 0.0

    # ── "near [reference]" / "closest to [reference]" ──────────────────────
    near_match = re.search(
        r"(?:near|closest to|nearest to|adjacent to|next to)\s+(?:the\s+)?(.+?)(?:\?|,|\.|$)",
        q)
    if near_match:
        ref_name = near_match.group(1).strip()
        ref_loc  = get_location(ocr_df, ref_name)
        if ref_loc:
            ref_cx, ref_cy = ref_loc[0], ref_loc[1]
            best = min(locs, key=lambda k:
                       ((locs[k]["cx"]-ref_cx)**2 + (locs[k]["cy"]-ref_cy)**2)**0.5)
            return best, min(locs[best]["match"], ref_loc[2])

    # ── "[direction] of [reference]" — RELATIVE positioning ────────────────
    # e.g. "north of Powai Lake", "west of Saki Naka"
    # Fix: pick option that is actually IN that direction relative to reference,
    #      not just the absolute extreme on the map
    dir_of_match = re.search(
        r"(north|south|east|west)\s+of\s+(.+?)(?:\?|,|\.|$)", q)
    if dir_of_match:
        direction = dir_of_match.group(1)
        ref_name  = dir_of_match.group(2).strip()
        ref_loc   = get_location(ocr_df, ref_name)
        if ref_loc:
            ref_cx, ref_cy = ref_loc[0], ref_loc[1]
            # Filter to only options that are genuinely in that direction
            valid = {}
            for k, v in locs.items():
                if   direction == "north" and v["cy"] < ref_cy:  valid[k] = v
                elif direction == "south" and v["cy"] > ref_cy:  valid[k] = v
                elif direction == "east"  and v["cx"] > ref_cx:  valid[k] = v
                elif direction == "west"  and v["cx"] < ref_cx:  valid[k] = v
            if valid:
                # Among valid, pick closest to the reference
                best = min(valid, key=lambda k:
                           ((valid[k]["cx"]-ref_cx)**2 + (valid[k]["cy"]-ref_cy)**2)**0.5)
                return best, valid[best]["match"]
            # No option found in that direction → return highest OCR match
            if locs:
                best = max(locs, key=lambda k: locs[k]["match"])
                return best, locs[best]["match"] * 0.5  # lower confidence

    # ── "in the north/south/east/west of the map" — absolute positioning ───
    # e.g. "shown in the north of the map"
    in_dir_match = re.search(r"in the\s+(north|south|east|west)\b", q)
    if in_dir_match:
        direction = in_dir_match.group(1)
        if   direction == "north": best = min(locs, key=lambda k: locs[k]["cy"])
        elif direction == "south": best = max(locs, key=lambda k: locs[k]["cy"])
        elif direction == "east":  best = max(locs, key=lambda k: locs[k]["cx"])
        elif direction == "west":  best = min(locs, key=lambda k: locs[k]["cx"])
        return best, locs[best]["match"]

    # ── Absolute superlatives (no reference) ───────────────────────────────
    if re.search(r"\b(northernmost|furthest north|farthest north|most north)\b", q):
        best = min(locs, key=lambda k: locs[k]["cy"])
        return best, locs[best]["match"]
    if re.search(r"\b(southernmost|furthest south|farthest south|most south)\b", q):
        best = max(locs, key=lambda k: locs[k]["cy"])
        return best, locs[best]["match"]
    if re.search(r"\b(easternmost|furthest east|farthest east|most east)\b", q):
        best = max(locs, key=lambda k: locs[k]["cx"])
        return best, locs[best]["match"]
    if re.search(r"\b(westernmost|furthest west|farthest west|most west)\b", q):
        best = min(locs, key=lambda k: locs[k]["cx"])
        return best, locs[best]["match"]

    # ── Fallback: highest OCR match score ──────────────────────────────────
    best = max(locs, key=lambda k: locs[k]["match"])
    return best, locs[best]["match"]


# ── Route 3: VISUAL (Qwen2-VL) ───────────────────────────────────────────────
def load_qwen(model_dir):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    print("Loading Qwen2-VL-7B...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16,
        device_map="auto", local_files_only=True)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    print("Qwen2-VL loaded.")
    return model, processor


def _crop_around(canvas_bgr, cx, cy, size=CROP_SIZE):
    H, W = canvas_bgr.shape[:2]; half = size//2
    x1,x2 = max(0,cx-half), min(W,cx+half)
    y1,y2 = max(0,cy-half), min(H,cy+half)
    crop = canvas_bgr[y1:y2, x1:x2]
    return crop if crop.size > 0 else canvas_bgr


def answer_visual(question, options, canvas_bgr, ocr_df, model, processor):
    from ocr_index import get_location
    from qwen_vl_utils import process_vision_info

    # Try to find relevant crop from named entities in question
    crop = canvas_bgr
    for entity in re.findall(r"[A-Z][a-zA-Z\s]{2,}", question):
        loc = get_location(ocr_df, entity.strip())
        if loc:
            region = _crop_around(canvas_bgr, loc[0], loc[1])
            if region.size > 0: crop = region; break

    # Resize if too large
    H, W = crop.shape[:2]
    if max(H, W) > 1024:
        scale = 1024 / max(H, W)
        crop  = cv2.resize(crop, (int(W*scale), int(H*scale)))

    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    opts_str = "\n".join([f"Option {i+1}: {o}" for i, o in enumerate(options)])
    prompt = (
        f"Look carefully at this map. Read ALL visible text labels.\n\n"
        f"Question: {question}\n\n{opts_str}\n\n"
        f"Which option (1,2,3,4) is correct? Output 5 if unsure.\n"
        f"Respond ONLY with JSON: {{\"answer\":<1-5>,\"confidence\":<0.0-1.0>}}"
    )
    messages = [{"role": "user", "content": [
        {"type": "image", "image": crop_pil},
        {"type": "text",  "text": prompt}
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=img_inputs, videos=vid_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=50,
            return_dict_in_generate=True, output_scores=True)
    new_tokens = outputs.sequences[:, inputs["input_ids"].shape[1]:]
    raw = processor.decode(new_tokens[0], skip_special_tokens=True).strip()
    return _parse_vlm(raw, outputs, processor)


def _parse_vlm(raw, outputs, processor):
    m = re.search(r"\{.*?\}", raw, re.DOTALL)
    if m:
        try:
            d    = json.loads(m.group())
            ans  = int(d.get("answer", 5))
            conf = float(d.get("confidence", 0.5))
            if ans not in [1, 2, 3, 4, 5]: ans = 5
            return ans, conf
        except Exception: pass
    if outputs and outputs.scores:
        probs = torch.softmax(outputs.scores[0][0], dim=0)
        digit_tokens = {}
        for d in ["1", "2", "3", "4"]:
            tid = processor.tokenizer.encode(d, add_special_tokens=False)
            if tid: digit_tokens[int(d)] = probs[tid[0]].item()
        if digit_tokens:
            best = max(digit_tokens, key=digit_tokens.get)
            return best, digit_tokens[best]
    d = re.search(r"\b([1-4])\b", raw)
    return (int(d.group(1)), 0.3) if d else (5, 0.0)


# ── Master function ───────────────────────────────────────────────────────────
def answer_question(question, options, canvas_bgr, ocr_df,
                    model, processor, verbose=True):
    route = route_question(question)

    if route == "lookup":
        ans, conf = answer_lookup(options, ocr_df)
    elif route == "spatial":
        ans, conf = answer_spatial(question, options, ocr_df)
        if ans == 5:
            route = "visual"
            ans, conf = answer_visual(
                question, options, canvas_bgr, ocr_df, model, processor)
    else:
        ans, conf = answer_visual(
            question, options, canvas_bgr, ocr_df, model, processor)

    final = ans if conf >= CONFIDENCE_THRESHOLD else 5

    if verbose:
        tag = "SKIP ⏭" if final == 5 else f"→ Option {final}"
        print(f"  [{route.upper()}] raw={ans} conf={conf:.2f} {tag}")

    return final, conf, route
