"""
Unified Map Stitcher (Dynamic Module)
- Overlap constrained to 24–64 pixels
- Patches must be square (enforced at load time)
- Final image can be rectangular (rows × cols)
"""

from pathlib import Path
from collections import defaultdict
from itertools import combinations
import math
import numpy as np
import cv2
from PIL import Image
import imagehash


# CONFIGURATION

PATCH_DIR      = Path("patches")
OUTPUT_PATH    = Path("final_stitched.png")
VALID_EXTS     = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
THREE_PART_MSE = 300.0

# Overlap search range (pixels) — in [24, 64]
OVERLAP_MIN = 24
OVERLAP_MAX = 64


# UTILITIES & MATH

def _factorizations(n):
    """Return all (rows, cols) pairs where rows*cols == n, rows <= cols."""
    pairs = []
    for r in range(1, int(math.isqrt(n)) + 1):
        if n % r == 0:
            pairs.append((r, n // r))
    return pairs


def _estimate_grid(n):
    """
    Return (n_rows, n_cols) for a rectangular grid.
    Prefers the most-square factorization but keeps rows <= cols so the
    final image can be wider than it is tall (typical map layout).
    """
    pairs = _factorizations(n)
    best = min(pairs, key=lambda rc: abs(rc[0] - rc[1]))
    return best


def list_images(folder):
    """Returns a sorted list of valid image paths from the folder."""
    return sorted([p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in VALID_EXTS])


def enforce_square_patches(images, paths):
    """
    Assert every loaded patch is square.
    Raises ValueError listing all offending files if any are not square.
    """
    bad = []
    for p, im in zip(paths, images):
        if im.width != im.height:
            bad.append(f"  {p.name}: {im.width}x{im.height}")
    if bad:
        raise ValueError(
            "All patches must be square, but the following are not:\n"
            + "\n".join(bad)
        )


def arr_mse(a, b):
    """Calculates Mean Squared Error between two numpy pixel arrays."""
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    if h == 0 or w == 0:
        return float("inf")
    return float(np.mean(
        (a[:h, :w].astype(np.float32) - b[:h, :w].astype(np.float32)) ** 2
    ))


def has_path(start, target, out_map):
    """Checks for a path in the DAG to prevent cycles."""
    cur, seen = start, set()
    while cur in out_map and cur not in seen:
        if cur == target:
            return True
        seen.add(cur)
        cur = out_map[cur]
    return cur == target


def three_part_ok(src_strip, dst_strip, max_mse=THREE_PART_MSE):
    """Splits overlapping regions into thirds; all three must pass MSE threshold."""
    h = min(src_strip.shape[0], dst_strip.shape[0])
    w = min(src_strip.shape[1], dst_strip.shape[1])
    if h == 0 or w == 0:
        return False
    s, d = src_strip[:h, :w], dst_strip[:h, :w]
    a, b = w // 3, (2 * w) // 3
    return all(
        arr_mse(s[:, l:r], d[:, l:r]) <= max_mse
        for l, r in [(0, a), (a, b), (b, w)]
        if r > l
    )


 
# PREPROCESSING: STRIPS & HASHES
 
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
            strips = [(0, gray.crop((0, 0, q, h))),
                      (1, gray.crop((w - q, 0, w, h)))]
        else:
            q = max(1, h // 4)
            strips = [(0, gray.crop((0, 0, w, q))),
                      (1, gray.crop((0, h - q, w, h)))]
        for label, strip in strips:
            records.append({
                "strip_idx":  len(records),
                "image_idx":  idx,
                "label":      label,
                "hash":       str(imagehash.phash(strip)),
            })
    return records


 
# GRAPH BUILDING & SOLVING
 
def compute_flag_info(records):
    """Flags unambiguous edges (no competing matches) for strict DAG inclusion."""
    hmap = defaultdict(list)
    for r in records:
        hmap[r["hash"]].append(r)
    left_nb  = defaultdict(set)
    right_nb = defaultdict(set)
    for recs in hmap.values():
        for i, j in combinations(recs, 2):
            ia, ib = i["image_idx"], j["image_idx"]
            if ia == ib:
                continue
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
    """Finds matching perceptual hashes and ranks them via MSE."""
    hmap = defaultdict(list)
    for r in records:
        hmap[r["hash"]].append(r["strip_idx"])
    matched = set()
    for idxs in hmap.values():
        for i, j in combinations(idxs, 2):
            ra, rb = records[i], records[j]
            if ra["image_idx"] != rb["image_idx"]:
                matched.add((i, j) if i < j else (j, i))
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
        if score < best[key]:
            best[key] = score
    edges = [{"src": s, "dst": d, "mse": v} for (s, d), v in best.items()]
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
        if src in out_map or dst in in_map:
            continue
        if src == dst or has_path(dst, src, out_map):
            continue
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
            if nxt in visited:
                break
            chain.append(nxt); visited.add(nxt); cur = nxt
        chains.append(chain)
    for n in sorted(nodes):
        if n not in visited:
            chains.append([n])
    return chains


def join_partial_chains(partial_chains, all_edges, h_cache, flag_info, target):
    """Exhaustive DFS to bridge gaps between partial chains until length = target."""
    node_to_chain = {}
    for ci, chain in enumerate(partial_chains):
        for n in chain:
            node_to_chain[n] = ci

    join_edges = defaultdict(list)
    dag_used   = set()

    for ci, chain in enumerate(partial_chains):
        for k in range(len(chain) - 1):
            dag_used.add((chain[k], chain[k + 1]))

    for e in all_edges:
        src, dst = e["src"], e["dst"]
        if (src, dst) in dag_used:
            continue
        ci_src = node_to_chain.get(src)
        ci_dst = node_to_chain.get(dst)
        if ci_src is None or ci_dst is None:
            continue
        if ci_src == ci_dst:
            continue
        chain_src = partial_chains[ci_src]
        chain_dst = partial_chains[ci_dst]
        if chain_src[-1] != src:
            continue
        if chain_dst[0]  != dst:
            continue
        join_edges[src].append((dst, e["mse"]))

    for src in join_edges:
        join_edges[src].sort(key=lambda x: x[1])

    def boundary_ok(left_chain, right_chain):
        return three_part_ok(h_cache[left_chain[-1]]["right"],
                             h_cache[right_chain[0]]["left"])

    def structural_ok(full_chain):
        n = len(full_chain)
        if n < 3:
            return True
        a, b = n // 3, (2 * n) // 3
        for src, dst in [(full_chain[a - 1], full_chain[a]),
                         (full_chain[b - 1], full_chain[b])]:
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
            if ci_next in used_chains:
                continue
            next_chain = partial_chains[ci_next]
            if len(current_nodes) + len(next_chain) > target:
                continue
            if not boundary_ok(current_nodes, next_chain):
                continue
            result = dfs(current_nodes + next_chain, used_chains | {ci_next})
            if result is not None:
                return result
        return None

    for ci, chain in enumerate(partial_chains):
        result = dfs(list(chain), {ci})
        if result is not None:
            return result

    raise RuntimeError(f"Could not join chains into target length {target}.")


def build_all_rows(all_nodes, h_edges, h_cache, h_flag_info, target):
    """Processes horizontal links to build perfect rows of length `target`."""
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
    """
    n_rows = len(h_rows)

    patch_top_hash    = {}
    patch_bottom_hash = {}
    for row_idx, row_patches in enumerate(h_rows):
        for patch_idx in row_patches:
            img  = images[patch_idx].convert("L")
            w, h = img.size
            q    = max(1, h // 4)
            patch_top_hash[patch_idx]    = str(imagehash.phash(img.crop((0, 0, w, q))))
            patch_bottom_hash[patch_idx] = str(imagehash.phash(img.crop((0, h - q, w, h))))

    votes = defaultdict(lambda: defaultdict(int))
    for row_a, row_patches_a in enumerate(h_rows):
        for patch_a in row_patches_a:
            h_bottom = patch_bottom_hash[patch_a]
            for row_b, row_patches_b in enumerate(h_rows):
                if row_a == row_b:
                    continue
                for patch_b in row_patches_b:
                    if patch_top_hash[patch_b] == h_bottom:
                        votes[row_a][row_b] += 1

    all_edges = sorted(
        [(cnt, ra, rb)
         for ra, nbrs in votes.items()
         for rb, cnt in nbrs.items()],
        reverse=True,
    )
    below_map = {}
    above_map = {}

    for cnt, row_a, row_b in all_edges:
        if row_a in below_map or row_b in above_map:
            continue
        if has_path(row_b, row_a, below_map):
            continue
        below_map[row_a] = row_b
        above_map[row_b] = row_a

    tops = [r for r in range(n_rows) if r not in above_map]
    for start in tops:
        chain = [start]; visited = {start}
        while chain[-1] in below_map:
            nxt = below_map[chain[-1]]
            if nxt in visited:
                break
            chain.append(nxt); visited.add(nxt)
        if len(chain) == n_rows:
            print(f"  Vertical order found via patch voting: {chain}")
            return chain

    print("  WARNING: Hash voting incomplete, falling back to MSE on first patches...")
    return _vertical_order_mse_fallback(h_rows, images, n_rows)


def _vertical_order_mse_fallback(h_rows, images, n_rows):
    """Fallback: order rows by MSE between bottom of row A and top of row B."""
    row_top    = {}
    row_bottom = {}
    for row_idx, row_patches in enumerate(h_rows):
        arr = np.array(images[row_patches[0]].convert("L"), dtype=np.float32)
        h, w = arr.shape
        q = max(1, h // 4)
        row_top[row_idx]    = arr[:q, :]
        row_bottom[row_idx] = arr[-q:, :]

    mse_edges = sorted(
        [(arr_mse(row_bottom[a], row_top[b]), a, b)
         for a in range(n_rows)
         for b in range(n_rows) if a != b]
    )
    below_map = {}
    above_map = {}
    for mse, a, b in mse_edges:
        if a in below_map or b in above_map:
            continue
        if has_path(b, a, below_map):
            continue
        below_map[a] = b
        above_map[b] = a

    tops  = [r for r in range(n_rows) if r not in above_map]
    chain = [tops[0]]
    while chain[-1] in below_map:
        chain.append(below_map[chain[-1]])
    print(f"  MSE fallback vertical order: {chain}")
    return chain


 
# OVERLAP DETECTION  (constrained 24–64 px)
 
def detect_overlap(img_a: Image.Image, img_b: Image.Image, axis: str) -> int:
    """
    Find exact overlap between two adjacent patches by exhaustive MSE search,
    restricted to the range [OVERLAP_MIN, OVERLAP_MAX] pixels.

    axis='h': img_b is to the RIGHT of img_a  -> compare right edge of A vs left edge of B
    axis='v': img_b is BELOW img_a            -> compare bottom edge of A vs top edge of B

    Returns: overlap in pixels (always in [OVERLAP_MIN, OVERLAP_MAX])
    """
    arr_a = np.array(img_a.convert("L"), dtype=np.float32)
    arr_b = np.array(img_b.convert("L"), dtype=np.float32)

    if axis == "h":
        max_possible = min(arr_a.shape[1], arr_b.shape[1]) - 1
    else:
        max_possible = min(arr_a.shape[0], arr_b.shape[0]) - 1

    lo = OVERLAP_MIN
    hi = min(OVERLAP_MAX, max_possible)

    if lo > hi:
        # Patch too small to honour the range; use whatever fits
        hi = max(1, max_possible)
        lo = min(lo, hi)

    best_ov, best_mse = lo, float("inf")

    if axis == "h":
        for ov in range(lo, hi + 1):
            mse = arr_mse(arr_a[:, -ov:], arr_b[:, :ov])
            if mse < best_mse:
                best_mse = mse
                best_ov  = ov
    else:
        for ov in range(lo, hi + 1):
            mse = arr_mse(arr_a[-ov:, :], arr_b[:ov, :])
            if mse < best_mse:
                best_mse = mse
                best_ov  = ov

    return best_ov


 
# FINAL STITCH RENDERING
 
def stitch_h(images: list[Image.Image]) -> Image.Image:
    """Horizontally stitch patches with per-pair overlap detection (24-64 px)."""
    min_h = min(im.height for im in images)
    normalized = []
    for im in images:
        rgb = im.convert("RGB")
        if rgb.height != min_h:
            rgb = rgb.crop((0, 0, rgb.width, min_h))
        normalized.append(rgb)

    stitched_so_far = normalized[0]
    parts = [stitched_so_far]

    for i in range(1, len(normalized)):
        patch = normalized[i]

        ov = detect_overlap(stitched_so_far, patch, axis="h")
        ov = min(ov, patch.width - 1)
        cropped = patch.crop((ov, 0, patch.width, patch.height))
        parts.append(cropped)

        # Rebuild stitched_so_far so each subsequent patch is compared against
        # the correct right edge (important when patches differ in width).
        canvas_w = sum(p.width for p in parts)
        tmp = Image.new("RGB", (canvas_w, min_h))
        x = 0
        for p in parts:
            tmp.paste(p, (x, 0))
            x += p.width
        stitched_so_far = tmp

    return stitched_so_far


def stitch_v(images: list[Image.Image]) -> Image.Image:
    """Vertically stitch full rows with per-pair overlap detection (24-64 px)."""
    min_w = min(im.width for im in images)
    normalized = []
    for im in images:
        rgb = im.convert("RGB")
        if rgb.width != min_w:
            rgb = rgb.crop((0, 0, min_w, rgb.height))
        normalized.append(rgb)

    stitched_so_far = normalized[0]
    parts = [stitched_so_far]

    for i in range(1, len(normalized)):
        row_img = normalized[i]

        ov = detect_overlap(stitched_so_far, row_img, axis="v")
        ov = min(ov, row_img.height - 1)
        cropped = row_img.crop((0, ov, row_img.width, row_img.height))
        parts.append(cropped)

        canvas_h = sum(p.height for p in parts)
        tmp = Image.new("RGB", (min_w, canvas_h))
        y = 0
        for p in parts:
            tmp.paste(p, (0, y))
            y += p.height
        stitched_so_far = tmp

    return stitched_so_far


 
# MODULAR ENTRY POINT
 
def stitch_patches(patches_dir, output_path=None):
    """
    Called by inference.py.
    patches_dir : path to folder containing patch_*.png  (must be square patches)
    output_path : optional path to save stitched PNG
    Returns     : numpy BGR array (OpenCV format)
    """
    folder      = Path(patches_dir)
    image_paths = list_images(folder)
    N           = len(image_paths)

    if N == 0:
        raise FileNotFoundError(f"No images found in {folder.resolve()}")

    images = [Image.open(p) for p in image_paths]

    # Enforce square patches
    enforce_square_patches(images, image_paths)

    # Determine rectangular grid layout
    n_rows, n_cols = _estimate_grid(N)
    print(f"Images={N}  GRID={n_rows}x{n_cols}  "
          f"(overlap range: {OVERLAP_MIN}-{OVERLAP_MAX} px)")

    all_nodes = list(range(N))

    # Horizontal pass
    print("\n-- Horizontal pass --")
    h_records   = build_hash_records(images, "h")
    h_cache     = build_strip_cache(images,  "h")
    h_flag_info = compute_flag_info(h_records)
    h_edges     = build_candidate_edges(h_records, h_cache, "h")

    n_flag  = sum(1 for v in h_flag_info.values() if     v["flag"])
    n_ambig = sum(1 for v in h_flag_info.values() if not v["flag"])
    print(f"  candidate edges={len(h_edges)}")
    print(f"  unambiguous={n_flag}  ambiguous={n_ambig}\n")

    h_rows = build_all_rows(all_nodes, h_edges, h_cache, h_flag_info, target=n_cols)

    bad = [r for r in h_rows if len(r) != n_cols]
    if bad:
        raise RuntimeError(f"{len(bad)} row(s) with wrong length != {n_cols}")

    if len(h_rows) != n_rows:
        raise RuntimeError(
            f"Expected {n_rows} rows but got {len(h_rows)}. "
            f"Check that N={N} patches match the {n_rows}x{n_cols} grid."
        )

    print(f"\n  {len(h_rows)} rows of length {n_cols} OK")
    row_images = [stitch_h([images[idx] for idx in row]) for row in h_rows]

    # Vertical pass
    print("\n-- Vertical pass --")
    flat_v = order_rows_vertically_from_patches(h_rows, images)
    print(f"  vertical order: {flat_v}")

    # Final render
    print("\n-- Final stitch --")
    final_pil = stitch_v([row_images[idx] for idx in flat_v])

    if output_path:
        final_pil.save(output_path)
        print(f"Saved -> {output_path}  size={final_pil.size}")

    return cv2.cvtColor(np.array(final_pil.convert("RGB")), cv2.COLOR_RGB2BGR)


 
# CLI EXECUTION
 
if __name__ == "__main__":
    import sys
    patch_folder = Path(sys.argv[1]) if len(sys.argv) > 1 else PATCH_DIR
    canvas = stitch_patches(patch_folder, OUTPUT_PATH)
    print(f"Done. Canvas shape: {canvas.shape}")