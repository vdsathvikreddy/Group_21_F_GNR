"""
answering.py
Hybrid MCQ answering engine with 3 routes:
  LOOKUP  — option text found directly in OCR index
  SPATIAL — coordinate math only when ≥1 options confirmed on-map by OCR
  VISUAL  — Qwen2-VL-7B on map image (fallback for everything else)

Confidence threshold = 0.20 (break-even for -0.25 penalty scoring).
"""

import re, json
import numpy as np
import cv2
import torch
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD  = 0.20   
CROP_SIZE             = 800    
OCR_ON_MAP_THRESHOLD  = 0.82

# FIX 1: Lowered to 1. If only the correct answer is on the map, allow SPATIAL to pick it.
MIN_OPTIONS_ON_MAP    = 1

SPATIAL_REF_PATTERNS = [
    r"near\s+\w",
    r"closest to", r"nearest to",
    r"north of", r"south of", r"east of", r"west of",
    r"adjacent to", r"next to",
    r"in the north", r"in the south", r"in the east", r"in the west",
    r"in the bottom", r"in the top",
    r"bottom left", r"bottom right", r"top left", r"top right",
    r"northeast", r"northwest", r"southeast", r"southwest",
]

SPATIAL_KEYWORDS = [
    "furthest north", "farthest north", "northernmost", "most north", "further north",
    "furthest south", "farthest south", "southernmost", "most south", "further south",
    "furthest east",  "farthest east",  "easternmost",  "most east",  "further east",
    "furthest west",  "farthest west",  "westernmost",  "most west",  "further west",
    "furthest", "farthest",
]

LOOKUP_KEYWORDS = [
    "visible", "shown", "present", "exist", "found in", "seen",
    "appears", "marked", "indicated", "labeled", "labelled"
]


# ── Routing ───────────────────────────────────────────────────────────────────
def route_question(question):
    q = question.lower()
    if any(re.search(p, q) for p in SPATIAL_REF_PATTERNS): return "spatial"
    if any(k in q for k in SPATIAL_KEYWORDS): return "spatial"
    if any(re.search(k, q) for k in LOOKUP_KEYWORDS): return "lookup"
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
                best_conf = score
                best_opt  = i + 1
    if best_opt and best_conf >= 0.82:
        return best_opt, best_conf
    return 5, 0.0   


# ── Route 2: SPATIAL ──────────────────────────────────────────────────────────
def answer_spatial(question, options, ocr_df):
    from ocr_index import find_text, get_location
    
    # FIX 2: Pre-clean to prevent "the map" from acting as a spatial landmark
    q = question.lower()
    q = re.sub(r"\b(?:of|on|in)\s+the\s+map\b", "", q).strip()

    locs = {}
    for i, opt in enumerate(options):
        hits = find_text(ocr_df, opt, top_k=1, min_conf=0.3)
        if len(hits) > 0 and float(hits.iloc[0]["match_score"]) >= OCR_ON_MAP_THRESHOLD:
            loc = get_location(ocr_df, opt)
            if loc:
                locs[i+1] = {"cx": loc[0], "cy": loc[1], "match": float(hits.iloc[0]["match_score"])}

    if len(locs) < MIN_OPTIONS_ON_MAP:
        print(f"    [SPATIAL→VISUAL] only {len(locs)}/4 options found in OCR → falling back to Qwen")
        return 5, 0.0

    # ── "near [reference]" ──────────────────────────────────────────────────
    near_m = re.search(r"(?:near|closest to|nearest to|adjacent to|next to)\s+(?:the\s+)?(.+?)(?:\?|,|\.|$)", q)
    if near_m:
        ref_name = near_m.group(1).strip()
        ref_loc  = get_location(ocr_df, ref_name)
        if ref_loc:
            best = min(locs, key=lambda k: ((locs[k]["cx"]-ref_loc[0])**2 + (locs[k]["cy"]-ref_loc[1])**2)**0.5)
            return best, min(locs[best]["match"], ref_loc[2])

    # ── "[direction] of [reference]" — RELATIVE ─────────────────────────────
    dir_of_m = re.search(r"(north|south|east|west)\s+of\s+(.+?)(?:\?|,|\.|and\b|$)", q)
    if dir_of_m:
        direction = dir_of_m.group(1)
        ref_name  = dir_of_m.group(2).strip()
        if ref_name: 
            ref_loc = get_location(ocr_df, ref_name)
            if ref_loc:
                ref_cx, ref_cy = ref_loc[0], ref_loc[1]
                valid = {}
                for k, v in locs.items():
                    if   direction == "north" and v["cy"] < ref_cy: valid[k] = v
                    elif direction == "south" and v["cy"] > ref_cy: valid[k] = v
                    elif direction == "east"  and v["cx"] > ref_cx: valid[k] = v
                    elif direction == "west"  and v["cx"] < ref_cx: valid[k] = v
                if valid:
                    best = min(valid, key=lambda k: ((valid[k]["cx"]-ref_cx)**2 + (valid[k]["cy"]-ref_cy)**2)**0.5)
                    return best, valid[best]["match"]
                else:
                    # FIX 3: Do not guess blindly with 0.50 if math fails. Send to VISUAL.
                    return 5, 0.0

    # ── "in the north/south/east/west" — ABSOLUTE ────────────────
    in_dir_m = re.search(r"in the\s+(north|south|east|west)\b", q)
    if in_dir_m:
        direction = in_dir_m.group(1)
        if   direction == "north": best = min(locs, key=lambda k: locs[k]["cy"])
        elif direction == "south": best = max(locs, key=lambda k: locs[k]["cy"])
        elif direction == "east":  best = max(locs, key=lambda k: locs[k]["cx"])
        elif direction == "west":  best = min(locs, key=lambda k: locs[k]["cx"])
        return best, locs[best]["match"]

    # ── Corner/quadrant ──────────────────────────────────────────────────────
    if re.search(r"bottom.?left|southwest", q):
        best = max(locs, key=lambda k: locs[k]["cy"] - locs[k]["cx"])
        return best, locs[best]["match"]
    if re.search(r"bottom.?right|southeast", q):
        best = max(locs, key=lambda k: locs[k]["cy"] + locs[k]["cx"])
        return best, locs[best]["match"]
    if re.search(r"top.?left|northwest", q):
        best = min(locs, key=lambda k: locs[k]["cy"] + locs[k]["cx"])
        return best, locs[best]["match"]
    if re.search(r"top.?right|northeast", q):
        best = max(locs, key=lambda k: locs[k]["cx"] - locs[k]["cy"])
        return best, locs[best]["match"]

    # ── Absolute superlatives ────────────────────────────────────────────────
    if re.search(r"\b(northernmost|furthest north|farthest north|most north|further north)\b", q):
        best = min(locs, key=lambda k: locs[k]["cy"])
        return best, locs[best]["match"]
    if re.search(r"\b(southernmost|furthest south|farthest south|most south|further south)\b", q):
        best = max(locs, key=lambda k: locs[k]["cy"])
        return best, locs[best]["match"]
    if re.search(r"\b(easternmost|furthest east|farthest east|most east|further east)\b", q):
        best = max(locs, key=lambda k: locs[k]["cx"])
        return best, locs[best]["match"]
    if re.search(r"\b(westernmost|furthest west|farthest west|most west|further west)\b", q):
        best = min(locs, key=lambda k: locs[k]["cx"])
        return best, locs[best]["match"]

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
    H, W = canvas_bgr.shape[:2]; half = size // 2
    x1, x2 = max(0, cx-half), min(W, cx+half)
    y1, y2 = max(0, cy-half), min(H, cy+half)
    crop = canvas_bgr[y1:y2, x1:x2]
    return crop if crop.size > 0 else canvas_bgr


def answer_visual(question, options, canvas_bgr, ocr_df, model, processor):
    from ocr_index import get_location
    from qwen_vl_utils import process_vision_info

    crop = None
    named_entities = re.findall(r"[A-Z][a-zA-Z\s&]{2,}", question)
    for entity in named_entities:
        loc = get_location(ocr_df, entity.strip())
        if loc:
            region = _crop_around(canvas_bgr, loc[0], loc[1])
            if region.size > 0:
                crop = region
                break

    if crop is None:
        for opt in options:
            loc = get_location(ocr_df, opt)
            if loc:
                crop = _crop_around(canvas_bgr, loc[0], loc[1])
                break

    if crop is None:
        crop = canvas_bgr

    H, W = crop.shape[:2]
    if max(H, W) > 1024:
        scale = 1024 / max(H, W)
        crop  = cv2.resize(crop, (int(W*scale), int(H*scale)))

    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    opts_str = "\n".join([f"Option {i+1}: {o}" for i, o in enumerate(options)])

    prompt = (
        f"You are analyzing a detailed street map of Mumbai, India.\n"
        f"Read ALL visible text labels carefully including road names, "
        f"area names, landmarks, lakes, and institutions.\n\n"
        f"Question: {question}\n\n"
        f"{opts_str}\n\n"
        f"Based ONLY on what is visible in this map image, which option number "
        f"(1, 2, 3, or 4) is correct?\n"
        f"If you cannot determine the answer confidently, output 5.\n"
        f"Respond ONLY with this JSON: "
        f"{{\"answer\": <1-5>, \"confidence\": <0.0-1.0>, \"reason\": \"<brief>\"}}"
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
            **inputs, max_new_tokens=80,
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
        except Exception:
            pass
    if outputs and outputs.scores:
        probs = torch.softmax(outputs.scores[0][0], dim=0)
        digit_tokens = {}
        for d in ["1", "2", "3", "4"]:
            tid = processor.tokenizer.encode(d, add_special_tokens=False)
            if tid:
                digit_tokens[int(d)] = probs[tid[0]].item()
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
        if ans == 5:               
            route = "visual"
            ans, conf = answer_visual(
                question, options, canvas_bgr, ocr_df, model, processor)

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
