"""
Unified Map Stitcher (Dynamic Module)
Assumes all patches are correctly oriented (no rotations).
Dynamically calculates grid size and exports a BGR array for downstream inference.
"""

from pathlib import Path
from collections import defaultdict
from itertools import combinations
import math
import numpy as np
import cv2
from PIL import Image
import imagehash

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
PATCH_DIR      = Path("patches")
OUTPUT_PATH    = Path("final_stitched.png")
VALID_EXTS     = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
THREE_PART_MSE = 300.0


# ─────────────────────────────────────────────
# UTILITIES & MATH
# ─────────────────────────────────────────────
def _estimate_row_len(n):
    """Find most-square factorization, return cols (wider side)."""
    best = (1, n)
    for r in range(int(math.isqrt(n)), 0, -1):
        if n % r == 0:
            c = n // r
            if abs(r - c) < abs(best[0] - best[1]):
                best = (r, c)
    return max(best)

def list_images(folder):
    """Returns a sorted list of valid image paths from the folder."""
    return sorted([p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in VALID_EXTS])

def arr_mse(a, b):
    """Calculates Mean Squared Error between two numpy pixel arrays."""
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    if h == 0 or w == 0: return float("inf")
    return float(np.mean((a[:h,:w].astype(np.float32) - b[:h,:w].astype(np.float32))**2))

def has_path(start, target, out_map):
    """Checks for a path in the Directed Acyclic Graph (DAG) to prevent cycles."""
    cur, seen = start, set()
    while cur in out_map and cur not in seen:
        if cur == target: return True
        seen.add(cur); cur = out_map[cur]
    return cur == target

def three_part_ok(src_strip, dst_strip, max_mse=THREE_PART_MSE):
    """Splits overlapping regions into thirds; all three must pass the MSE threshold."""
    h = min(src_strip.shape[0], dst_strip.shape[0])
    w = min(src_strip.shape[1], dst_strip.shape[1])
    if h == 0 or w == 0: return False
    s, d = src_strip[:h, :w], dst_strip[:h, :w]
    a, b = w // 3, (2 * w) // 3
    return all(arr_mse(s[:, l:r], d[:, l:r]) <= max_mse
               for l, r in [(0, a), (a, b), (b, w)] if r > l)


# ─────────────────────────────────────────────
# PREPROCESSING: STRIPS & HASHES
# ─────────────────────────────────────────────
def build_strip_cache(images, axis):
    """Caches 1/4 width/height boundary strips as raw pixel arrays."""
    cache = {}
    for idx, im in enumerate(images):
        arr = np.array(im.convert("L"), dtype=np.float32)
        h, w = arr.shape
        if axis == "h":
            q = max(1, w // 4)
            cache[idx] = {"left": arr[:, :q], "right": arr[:, -q:]}
        else:
            q = max(1, h // 4)
            cache[idx] = {"top": arr[:q, :], "bottom": arr[-q:, :]}
    return cache

def build_hash_records(images, axis):
    """Computes perceptual hashes for boundary strips to quickly group matches."""
    records = []
    for idx, im in enumerate(images):
        gray = im.convert("L")
        w, h = gray.size
        if axis == "h":
            q = max(1, w // 4)
            strips = [(0, gray.crop((0, 0, q, h))), (1, gray.crop((w-q, 0, w, h)))]
        else:
            q = max(1, h // 4)
            strips = [(0, gray.crop((0, 0, w, q))), (1, gray.crop((0, h-q, w, h)))]
        for label, strip in strips:
            records.append({"strip_idx": len(records), "image_idx": idx,
                            "label": label, "hash": str(imagehash.phash(strip))})
    return records


# ─────────────────────────────────────────────
# GRAPH BUILDING & SOLVING
# ─────────────────────────────────────────────
def compute_flag_info(records):
    """Flags unambiguous edges (no competing matches) for strict DAG inclusion."""
    hmap = defaultdict(list)
    for r in records: hmap[r["hash"]].append(r)
    left_nb  = defaultdict(set)
    right_nb = defaultdict(set)
    for recs in hmap.values():
        for i, j in combinations(recs, 2):
            ia, ib = i["image_idx"], j["image_idx"]
            if ia == ib: continue
            la, lb = i["label"], j["label"]
            if la == 0: left_nb[ia].add(ib)
            if la == 1: right_nb[ia].add(ib)
            if lb == 0: left_nb[ib].add(ia)
            if lb == 1: right_nb[ib].add(ia)
    info = {}
    for idx in {r["image_idx"] for r in records}:
        ln, rn = len(left_nb[idx]), len(right_nb[idx])
        info[idx] = {"left_nb": ln, "right_nb": rn, "flag": (ln <= 1 and rn <= 1)}
    return info

def build_candidate_edges(records, cache, axis):
    """Finds matching perceptual hashes and ranks them mathematically via MSE."""
    hmap = defaultdict(list)
    for r in records: hmap[r["hash"]].append(r["strip_idx"])
    matched = set()
    for idxs in hmap.values():
        for i, j in combinations(idxs, 2):
            ra, rb = records[i], records[j]
            if ra["image_idx"] != rb["image_idx"]:
                matched.add((i,j) if i < j else (j,i))
    best = defaultdict(lambda: float("inf"))
    for i, j in matched:
        ra, rb = records[i], records[j]
        ia, ib = ra["image_idx"], rb["image_idx"]
        la, lb = ra["label"],     rb["label"]
        if axis == "h":
            if   la == 1 and lb == 0: src, dst = ia, ib
            elif la == 0 and lb == 1: src, dst = ib, ia
            else: continue
            score = arr_mse(cache[src]["right"], cache[dst]["left"])
        else:
            if   la == 1 and lb == 0: src, dst = ia, ib
            elif la == 0 and lb == 1: src, dst = ib, ia
            else: continue
            score = arr_mse(cache[src]["bottom"], cache[dst]["top"])
        key = (src, dst)
        if score < best[key]: best[key] = score
    edges = [{"src": s, "dst": d, "mse": v} for (s,d),v in best.items()]
    edges.sort(key=lambda e: e["mse"])
    return edges

def build_dag(nodes, edges, flag_info):
    """Constructs the trusted graph skeleton using strictly unambiguous edges."""
    out_map, in_map = {}, {}
    for e in edges:
        src, dst = e["src"], e["dst"]
        if not (flag_info.get(src, {}).get("flag") and
                flag_info.get(dst, {}).get("flag")):
            continue
        if src in out_map or dst in in_map: continue
        if src == dst or has_path(dst, src, out_map): continue
        out_map[src] = dst
        in_map[dst]  = src
    return out_map, in_map

def extract_partial_chains(nodes, out_map, in_map):
    """Extracts solid segments from the DAG to be joined later."""
    chains, visited = [], set()
    for s in sorted(n for n in nodes if n not in in_map):
        chain = [s]; visited.add(s); cur = s
        while cur in out_map:
            nxt = out_map[cur]
            if nxt in visited: break
            chain.append(nxt); visited.add(nxt); cur = nxt
        chains.append(chain)
    for n in sorted(nodes):
        if n not in visited: chains.append([n])
    return chains

def join_partial_chains(partial_chains, all_edges, h_cache, flag_info, target):
    """Exhaustive DFS to bridge the gaps between partial chains until length = target."""
    node_to_chain = {}
    for ci, chain in enumerate(partial_chains):
        for n in chain:
            node_to_chain[n] = ci

    join_edges = defaultdict(list)
    dag_used   = set()

    for ci, chain in enumerate(partial_chains):
        for k in range(len(chain) - 1):
            dag_used.add((chain[k], chain[k+1]))

    for e in all_edges:
        src, dst = e["src"], e["dst"]
        if (src, dst) in dag_used: continue
        ci_src = node_to_chain.get(src)
        ci_dst = node_to_chain.get(dst)
        if ci_src is None or ci_dst is None: continue
        if ci_src == ci_dst: continue
        chain_src = partial_chains[ci_src]
        chain_dst = partial_chains[ci_dst]
        if chain_src[-1] != src: continue
        if chain_dst[0]  != dst: continue
        join_edges[src].append((dst, e["mse"]))

    for src in join_edges:
        join_edges[src].sort(key=lambda x: x[1])

    def boundary_ok(left_chain, right_chain):
        return three_part_ok(h_cache[left_chain[-1]]["right"],
                             h_cache[right_chain[0]]["left"])

    def structural_ok(full_chain):
        n = len(full_chain)
        if n < 3: return True
        a, b = n // 3, (2 * n) // 3
        for src, dst in [(full_chain[a-1], full_chain[a]),
                         (full_chain[b-1], full_chain[b])]:
            if not three_part_ok(h_cache[src]["right"], h_cache[dst]["left"]):
                return False
        return True

    def dfs(current_nodes, used_chains):
        if len(current_nodes) == target:
            return current_nodes if structural_ok(current_nodes) else None
        if len(current_nodes) > target:
            return None
        tail = current_nodes[-1]
        for head, _ in join_edges.get(tail, []):
            ci_next = node_to_chain[head]
            if ci_next in used_chains: continue
            next_chain = partial_chains[ci_next]
            if len(current_nodes) + len(next_chain) > target: continue
            if not boundary_ok(current_nodes, next_chain): continue
            result = dfs(current_nodes + next_chain, used_chains | {ci_next})
            if result is not None: return result
        return None

    for ci, chain in enumerate(partial_chains):
        result = dfs(list(chain), {ci})
        if result is not None:
            return result

    raise RuntimeError(f"Could not join chains into target length {target}.")

def build_all_rows(all_nodes, h_edges, h_cache, h_flag_info, target):
    """Processes horizontal links to build perfect rows."""
    remaining = set(all_nodes)
    rows = []
    while remaining:
        nodes = sorted(remaining)
        relevant_edges = [e for e in h_edges
                          if e["src"] in remaining and e["dst"] in remaining]
        out_map, in_map = build_dag(nodes, relevant_edges, h_flag_info)
        partial_chains  = extract_partial_chains(nodes, out_map, in_map)

        print(f"  remaining={len(remaining):<3} partial_chains={len(partial_chains):<2} "
              f"sizes={sorted(len(c) for c in partial_chains)}")

        row = join_partial_chains(partial_chains, relevant_edges,
                                  h_cache, h_flag_info, target=target)
        rows.append(row)
        for idx in row:
            remaining.remove(idx)
    return rows

def order_rows_vertically_from_patches(h_rows, images):
    """
    Orders rows vertically using ORIGINAL patch images, not stitched rows.
    Uses all patches in each row and votes on vertical adjacency.
    Much more reliable than hashing wide stitched row images.
    """
    n_rows = len(h_rows)

    # Step 1: Hash top and bottom of every original patch
    patch_top_hash    = {}
    patch_bottom_hash = {}
    for row_idx, row_patches in enumerate(h_rows):
        for patch_idx in row_patches:
            img  = images[patch_idx].convert("L")
            w, h = img.size
            q    = max(1, h // 4)
            patch_top_hash[patch_idx]    = str(imagehash.phash(img.crop((0, 0, w, q))))
            patch_bottom_hash[patch_idx] = str(imagehash.phash(img.crop((0, h-q, w, h))))

    # Step 2: For each hash value, find which rows have it as bottom vs top
    # votes[row_a][row_b] = how many patches say row_b is below row_a
    votes = defaultdict(lambda: defaultdict(int))
    for row_a, row_patches_a in enumerate(h_rows):
        for patch_a in row_patches_a:
            h_bottom = patch_bottom_hash[patch_a]
            for row_b, row_patches_b in enumerate(h_rows):
                if row_a == row_b: continue
                for patch_b in row_patches_b:
                    if patch_top_hash[patch_b] == h_bottom:
                        votes[row_a][row_b] += 1

    # Step 3: Build DAG from votes — highest vote = most confident edge
    all_edges = sorted(
        [(cnt, ra, rb) for ra, nbrs in votes.items() for rb, cnt in nbrs.items()],
        reverse=True
    )
    below_map = {}   # row_a → row_b  (row_b is directly below row_a)
    above_map = {}   # row_b → row_a

    for cnt, row_a, row_b in all_edges:
        if row_a in below_map or row_b in above_map: continue
        if has_path(row_b, row_a, below_map): continue
        below_map[row_a] = row_b
        above_map[row_b] = row_a

    # Step 4: Extract ordered chain from top (no above_map entry)
    tops = [r for r in range(n_rows) if r not in above_map]
    for start in tops:
        chain = [start]; visited = {start}
        while chain[-1] in below_map:
            nxt = below_map[chain[-1]]
            if nxt in visited: break
            chain.append(nxt); visited.add(nxt)
        if len(chain) == n_rows:
            print(f"  Vertical order found via patch voting: {chain}")
            return chain

    # Step 5: Fallback — MSE on first patch bottom vs top
    print("  WARNING: Hash voting incomplete, falling back to MSE on first patches...")
    return _vertical_order_mse_fallback(h_rows, images, n_rows)


def _vertical_order_mse_fallback(h_rows, images, n_rows):
    """Fallback: order rows by MSE between bottom of row A and top of row B."""
    row_top    = {}
    row_bottom = {}
    for row_idx, row_patches in enumerate(h_rows):
        # Use first patch as representative
        arr = np.array(images[row_patches[0]].convert("L"), dtype=np.float32)
        h, w = arr.shape; q = max(1, h // 4)
        row_top[row_idx]    = arr[:q, :]
        row_bottom[row_idx] = arr[-q:, :]

    mse_edges = sorted(
        [(arr_mse(row_bottom[a], row_top[b]), a, b)
         for a in range(n_rows) for b in range(n_rows) if a != b]
    )
    below_map = {}; above_map = {}
    for mse, a, b in mse_edges:
        if a in below_map or b in above_map: continue
        if has_path(b, a, below_map): continue
        below_map[a] = b; above_map[b] = a

    tops  = [r for r in range(n_rows) if r not in above_map]
    chain = [tops[0]]
    while chain[-1] in below_map:
        chain.append(below_map[chain[-1]])
    print(f"  MSE fallback vertical order: {chain}")
    return chain

def order_rows_vertically(row_images, v_records, v_cache, v_flag_info, v_edges):
    """Processes vertical links to order the stitched rows from top to bottom."""
    n_rows = len(row_images)
    v_nodes = list(range(n_rows))
    out_map, in_map = build_dag(v_nodes, v_edges, v_flag_info)
    v_partial = extract_partial_chains(v_nodes, out_map, in_map)
    print(f"  vertical partial chains: {len(v_partial)}  "
          f"sizes={sorted(len(c) for c in v_partial)}")
    if len(v_partial) == 1 and len(v_partial[0]) == n_rows:
        return v_partial[0]
    return join_partial_chains(v_partial, v_edges, v_cache, v_flag_info, target=n_rows)

def verify_vertical_alignment(v_chain, v_cache, v_records):
    """Performs a final validation pass ensuring all rows connect flawlessly."""
    hash_by_img = {}
    for r in v_records:
        idx = r["image_idx"]
        if idx not in hash_by_img: hash_by_img[idx] = {}
        hash_by_img[idx]["top" if r["label"] == 0 else "bottom"] = r["hash"]

    failures = []
    for i in range(len(v_chain) - 1):
        a, b = v_chain[i], v_chain[i+1]
        h_a = hash_by_img.get(a, {}).get("bottom")
        h_b = hash_by_img.get(b, {}).get("top")
        hash_match = (h_a is not None and h_a == h_b)
        pixel_ok   = three_part_ok(v_cache[a]["bottom"], v_cache[b]["top"])
        if not hash_match or not pixel_ok:
            failures.append(f"  row {i}->{i+1}: hash_match={hash_match} pixel_ok={pixel_ok}")
    if failures:
        raise ValueError("Vertical alignment failures:\n" + "\n".join(failures))
    print(f"  All {len(v_chain)-1} vertical transitions verified OK.")


def detect_overlap(img_a, img_b, axis, max_overlap_ratio=0.6):
    """
    Find exact overlap between two adjacent patches by exhaustive MSE search.
    Tries every possible overlap from 1px to max and picks the minimum MSE.
    Works reliably even on uniform regions (water, parks).
    
    axis='h': img_b is to the RIGHT of img_a
    axis='v': img_b is BELOW img_a
    Returns: overlap in pixels
    """
    arr_a = np.array(img_a.convert("L"), dtype=np.float32)
    arr_b = np.array(img_b.convert("L"), dtype=np.float32)

    if axis == "h":
        dim      = min(arr_a.shape[1], arr_b.shape[1])
        max_ov   = max(1, int(dim * max_overlap_ratio))
        best_ov, best_mse = 1, float("inf")
        for ov in range(1, max_ov + 1):
            # last `ov` cols of A vs first `ov` cols of B
            strip_a = arr_a[:, -ov:]
            strip_b = arr_b[:,  :ov]
            mse = arr_mse(strip_a, strip_b)
            if mse < best_mse:
                best_mse = mse
                best_ov  = ov

    else:  # axis == "v"
        dim      = min(arr_a.shape[0], arr_b.shape[0])
        max_ov   = max(1, int(dim * max_overlap_ratio))
        best_ov, best_mse = 1, float("inf")
        for ov in range(1, max_ov + 1):
            # last `ov` rows of A vs first `ov` rows of B
            strip_a = arr_a[-ov:, :]
            strip_b = arr_b[ :ov, :]
            mse = arr_mse(strip_a, strip_b)
            if mse < best_mse:
                best_mse = mse
                best_ov  = ov

    return best_ov

# ─────────────────────────────────────────────
# FINAL STITCH RENDERING (Fixed Geometric Cropping)
# ─────────────────────────────────────────────
def stitch_h(images):
    """Horizontally stitches map patches with auto-detected overlap per pair."""
    min_h = min(im.height for im in images)

    normalized = []
    for im in images:
        rgb = im.convert("RGB")
        if rgb.height != min_h:
            rgb = rgb.crop((0, 0, rgb.width, min_h))
        normalized.append(rgb)

    parts = []
    for i, im in enumerate(normalized):
        if i == 0:
            parts.append(im)
            continue
        # Detect actual overlap between this pair instead of assuming 25%
        ov = detect_overlap(normalized[i-1], im, axis="h")
        ov = min(ov, im.width - 1)
        parts.append(im.crop((ov, 0, im.width, im.height)))

    canvas = Image.new("RGB", (sum(p.width for p in parts), min_h))
    x = 0
    for p in parts:
        canvas.paste(p, (x, 0))
        x += p.width
    return canvas

def stitch_v(images):
    """Vertically stitches full rows using overlap cropping to preserve geometry."""
    min_w = min(im.width for im in images)

    normalized = []
    for im in images:
        rgb = im.convert("RGB")
        if rgb.width != min_w:
            rgb = rgb.crop((0, 0, min_w, rgb.height))
        normalized.append(rgb)

    parts = []
    for i, im in enumerate(normalized):
        if i == 0:
            parts.append(im)
            continue
        # Detect actual vertical overlap between this pair
        ov = detect_overlap(normalized[i-1], im, axis="v")
        ov = min(ov, im.height - 1)
        parts.append(im.crop((0, ov, im.width, im.height)))

    canvas = Image.new("RGB", (min_w, sum(p.height for p in parts)))
    y = 0
    for p in parts:
        canvas.paste(p, (0, y))
        y += p.height
    return canvas


# ─────────────────────────────────────────────
# MODULAR ENTRY POINT
# ─────────────────────────────────────────────
def stitch_patches(patches_dir, output_path=None):
    """
    Called by inference.py.
    patches_dir : path to folder containing patch_*.png
    output_path : optional path to save stitched PNG
    Returns     : numpy BGR array (OpenCV format)
    """
    folder      = Path(patches_dir)
    image_paths = list_images(folder)
    N           = len(image_paths)
    
    if N == 0:
        raise FileNotFoundError(f"No images found in {folder.resolve()}")

    row_len       = _estimate_row_len(N)
    perfect_edges = row_len * (row_len - 1)
    print(f"Images={N}  ROW_LEN={row_len}  PERFECT_EDGES={perfect_edges}")

    images    = [Image.open(p) for p in image_paths]
    all_nodes = list(range(N))

    # Horizontal pass
    print("\n── Horizontal pass ──")
    h_records   = build_hash_records(images, "h")
    h_cache     = build_strip_cache(images,  "h")
    h_flag_info = compute_flag_info(h_records)
    h_edges     = build_candidate_edges(h_records, h_cache, "h")

    n_flag  = sum(1 for v in h_flag_info.values() if     v["flag"])
    n_ambig = sum(1 for v in h_flag_info.values() if not v["flag"])
    print(f"  candidate edges={len(h_edges)}  (perfect={perfect_edges})")
    print(f"  unambiguous={n_flag}  ambiguous={n_ambig}\n")

    h_rows = build_all_rows(all_nodes, h_edges, h_cache, h_flag_info, target=row_len)

    bad = [r for r in h_rows if len(r) != row_len]
    if bad:
        raise RuntimeError(f"{len(bad)} row(s) with wrong length != {row_len}")

    print(f"\n  {len(h_rows)} rows of length {row_len} ✓")
    row_images = [stitch_h([images[idx] for idx in row]) for row in h_rows]

    # ── VERTICAL PASS ──
    # Use original patch images for vertical ordering — stitched rows are
    # too wide for reliable phash matching
    print("\n── Vertical pass ──")
    flat_v = order_rows_vertically_from_patches(h_rows, images)
    print(f"  vertical order: {flat_v}")

    # Final render
    print("\n── Final stitch ──")
    final_pil = stitch_v([row_images[idx] for idx in flat_v])

    if output_path:
        final_pil.save(output_path)
        print(f"Saved → {output_path}  size={final_pil.size}")

    # Return as numpy BGR for OCR + Qwen pipeline
    return cv2.cvtColor(np.array(final_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

# ─────────────────────────────────────────────
# CLI EXECUTION
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    patch_folder = Path(sys.argv[1]) if len(sys.argv) > 1 else PATCH_DIR
    canvas = stitch_patches(patch_folder, OUTPUT_PATH)
    print(f"Done. Canvas shape: {canvas.shape}")