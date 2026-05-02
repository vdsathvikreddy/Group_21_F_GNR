import sys
import math
import numpy as np
import cv2
import imagehash
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from PIL import Image

TARGET_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MSE_LIMIT = 300.0

def get_grid_dims(n):
    best = (1, n)
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            j = n // i
            if abs(i - j) < abs(best[0] - best[1]):
                best = (i, j)
    return best

def get_files(directory):
    return sorted([f for f in Path(directory).iterdir() if f.is_file() and f.suffix.lower() in TARGET_EXTS])

def get_mse(arr1, arr2):
    h = min(arr1.shape[0], arr2.shape[0])
    w = min(arr1.shape[1], arr2.shape[1])
    if h == 0 or w == 0: return float("inf")
    return float(np.mean((arr1[:h, :w].astype(np.float32) - arr2[:h, :w].astype(np.float32)) ** 2))

def detect_cycle(start, end, graph):
    curr, seen = start, set()
    while curr in graph and curr not in seen:
        if curr == end: return True
        seen.add(curr)
        curr = graph[curr]
    return curr == end

def check_boundary_thirds(s1, s2, limit=MSE_LIMIT):
    h = min(s1.shape[0], s2.shape[0])
    w = min(s1.shape[1], s2.shape[1])
    if h == 0 or w == 0: return False
    c1, c2 = s1[:h, :w], s2[:h, :w]
    p1, p2 = w // 3, (2 * w) // 3
    return all(get_mse(c1[:, l:r], c2[:, l:r]) <= limit for l, r in [(0, p1), (p1, p2), (p2, w)] if r > l)

def extract_edges(imgs, axis, size):
    res = {}
    for i, im in enumerate(imgs):
        mat = np.array(im.convert("L"), dtype=np.float32)
        h, w = mat.shape
        if axis == "h":
            val = min(size, w)
            res[i] = {"L": mat[:, :val], "R": mat[:, -val:]}
        else:
            val = min(size, h)
            res[i] = {"T": mat[:val, :], "B": mat[-val:, :]}
    return res

def compute_phash_data(imgs, axis, size):
    out = []
    for i, im in enumerate(imgs):
        g = im.convert("L")
        w, h = g.size
        if axis == "h":
            val = min(size, w)
            sections = [(0, g.crop((0, 0, val, h))), (1, g.crop((w - val, 0, w, h)))]
        else:
            val = min(size, h)
            sections = [(0, g.crop((0, 0, w, val))), (1, g.crop((0, h - val, w, h)))]
        for code, section in sections:
            out.append({"s_idx": len(out), "img_id": i, "side": code, "phash": str(imagehash.phash(section))})
    return out

def find_best_overlap(imgs, axis, required_edges, start=24, end=64):
    results = []
    selected = None
    for sz in range(start, end + 1):
        h_data = compute_phash_data(imgs, axis, sz)
        e_cache = extract_edges(imgs, axis, sz)
        c_edges = evaluate_pairs(h_data, e_cache, axis)
        count = len(c_edges)
        results.append((sz, count))
        if selected is None and count >= required_edges:
            selected = sz
    
    if selected is None:
        selected = max(results, key=lambda x: (x[1], -x[0]))[0]
        msg = "fallback to max candidates"
    else:
        msg = "met target threshold"

    print(f"  Sweep {axis} ({start}-{end}):")
    print("    " + ", ".join(f"{s}:{c}" for s, c in results))
    print(f"  Selected overlap={selected} (target={required_edges}, {msg})")
    return selected

def generate_flags(h_data):
    group = defaultdict(list)
    for item in h_data: group[item["phash"]].append(item)
    l_adj, r_adj = defaultdict(set), defaultdict(set)
    for items in group.values():
        for a, b in combinations(items, 2):
            id1, id2 = a["img_id"], b["img_id"]
            if id1 == id2: continue
            s1, s2 = a["side"], b["side"]
            if s1 == 0: l_adj[id1].add(id2)
            if s1 == 1: r_adj[id1].add(id2)
            if s2 == 0: l_adj[id2].add(id1)
            if s2 == 1: r_adj[id2].add(id1)
    res = {}
    for x in {r["img_id"] for r in h_data}:
        res[x] = {"l_cnt": len(l_adj[x]), "r_cnt": len(r_adj[x]), "valid": (len(l_adj[x]) <= 1 and len(r_adj[x]) <= 1)}
    return res

def refresh_flags(h_data, active_nodes):
    filtered = [x for x in h_data if x["img_id"] in active_nodes]
    group = defaultdict(list)
    for item in filtered: group[item["phash"]].append(item)
    l_adj, r_adj = defaultdict(set), defaultdict(set)
    for items in group.values():
        for a, b in combinations(items, 2):
            id1, id2 = a["img_id"], b["img_id"]
            if id1 == id2: continue
            s1, s2 = a["side"], b["side"]
            if s1 == 0: l_adj[id1].add(id2)
            if s1 == 1: r_adj[id1].add(id2)
            if s2 == 0: l_adj[id2].add(id1)
            if s2 == 1: r_adj[id2].add(id1)
    res = {}
    for x in active_nodes:
        res[x] = {"l_cnt": len(l_adj[x]), "r_cnt": len(r_adj[x]), "valid": (len(l_adj[x]) <= 1 and len(r_adj[x]) <= 1)}
    return res

def evaluate_pairs(h_data, e_cache, axis, tolerance=8):
    count = len(h_data)
    pairs = set()
    hash_vals = [imagehash.hex_to_hash(r["phash"]) for r in h_data]

    for i in range(count):
        for j in range(i + 1, count):
            if h_data[i]["img_id"] == h_data[j]["img_id"]: continue
            if hash_vals[i] - hash_vals[j] <= tolerance:
                pairs.add((i, j))

    optimal = defaultdict(lambda: float("inf"))
    for i, j in pairs:
        r1, r2 = h_data[i], h_data[j]
        i1, i2 = r1["img_id"], r2["img_id"]
        s1, s2 = r1["side"], r2["side"]
        if axis == "h":
            if s1 == 1 and s2 == 0: src, dst = i1, i2
            elif s1 == 0 and s2 == 1: src, dst = i2, i1
            else: continue
            err = get_mse(e_cache[src]["R"], e_cache[dst]["L"])
        else:
            if s1 == 1 and s2 == 0: src, dst = i1, i2
            elif s1 == 0 and s2 == 1: src, dst = i2, i1
            else: continue
            err = get_mse(e_cache[src]["B"], e_cache[dst]["T"])
        
        pair_k = (src, dst)
        if err < optimal[pair_k]: optimal[pair_k] = err

    final_edges = [{"src": s, "dst": d, "err": v} for (s, d), v in optimal.items()]
    final_edges.sort(key=lambda x: x["err"])
    return final_edges

def build_structure(nodes, edges, flags):
    fwd, bwd = {}, {}
    for edge in edges:
        s, d = edge["src"], edge["dst"]
        if not (flags.get(s, {}).get("valid") and flags.get(d, {}).get("valid")): continue
        if s in fwd or d in bwd: continue
        if s == d or detect_cycle(d, s, fwd): continue
        fwd[s] = d
        bwd[d] = s
    return fwd, bwd

def get_blocks(nodes, fwd, bwd):
    blocks, visited = [], set()
    for start in sorted(n for n in nodes if n not in bwd):
        blk = [start]
        visited.add(start)
        curr = start
        while curr in fwd:
            nxt = fwd[curr]
            if nxt in visited: break
            blk.append(nxt)
            visited.add(nxt)
            curr = nxt
        blocks.append(blk)
    for n in sorted(nodes):
        if n not in visited: blocks.append([n])
    return blocks

def connect_blocks(blocks, all_edges, e_cache, flags, expected_size):
    lookup = {n: i for i, blk in enumerate(blocks) for n in blk}
    options = defaultdict(list)
    used_links = set()

    for blk in blocks:
        for i in range(len(blk) - 1):
            used_links.add((blk[i], blk[i + 1]))

    for edge in all_edges:
        s, d = edge["src"], edge["dst"]
        if (s, d) in used_links: continue
        idx_s, idx_d = lookup.get(s), lookup.get(d)
        if idx_s is None or idx_d is None or idx_s == idx_d: continue
        if blocks[idx_s][-1] != s or blocks[idx_d][0] != d: continue
        options[s].append((d, edge["err"]))

    for k in options: options[k].sort(key=lambda x: x[1])

    def test_boundary(b1, b2):
        return check_boundary_thirds(e_cache[b1[-1]]["R"], e_cache[b2[0]]["L"])

    def test_full(arr):
        n = len(arr)
        if n < 3: return True
        p1, p2 = n // 3, (2 * n) // 3
        for src, dst in [(arr[p1 - 1], arr[p1]), (arr[p2 - 1], arr[p2])]:
            if not check_boundary_thirds(e_cache[src]["R"], e_cache[dst]["L"]): return False
        return True

    def explore(path, used_idx):
        if len(path) == expected_size:
            return path if test_full(path) else None
        if len(path) > expected_size: return None
        
        for head, _ in options.get(path[-1], []):
            n_idx = lookup[head]
            if n_idx in used_idx: continue
            cand = blocks[n_idx]
            if len(path) + len(cand) > expected_size: continue
            if not test_boundary(path, cand): continue
            res = explore(path + cand, used_idx | {n_idx})
            if res: return res
        return None

    for i, blk in enumerate(blocks):
        res = explore(list(blk), {i})
        if res: return res

    raise RuntimeError(f"Failed to assemble blocks into length {expected_size}.")

def assemble_grid(nodes, h_data, h_edges, e_cache, expected_size):
    pool = set(nodes)
    grid = []

    while len(pool) >= expected_size:
        active = sorted(pool)
        valid_edges = [e for e in h_edges if e["src"] in pool and e["dst"] in pool]

        dyn_flags = refresh_flags(h_data, pool)
        fwd, bwd = build_structure(active, valid_edges, dyn_flags)
        blocks = get_blocks(active, fwd, bwd)

        print(f"  Pool={len(pool):<3} Blocks={len(blocks):<2} Sizes={[len(b) for b in blocks]}")

        ready = [b for b in blocks if len(b) == expected_size]
        if ready:
            r = ready[0]
            grid.append(r)
            for x in r: pool.discard(x)
            continue

        try:
            r = connect_blocks(blocks, valid_edges, e_cache, dyn_flags, expected_size)
            grid.append(r)
            for x in r: pool.discard(x)
        except RuntimeError as e:
            print(f"  ERR: {e}")
            biggest = max(blocks, key=len)
            print(f"  Using best match of size {len(biggest)}")
            grid.append(biggest)
            for x in biggest: pool.discard(x)

    return grid

def sort_vertical(row_imgs, v_data, e_cache, flags, v_edges):
    total = len(row_imgs)
    v_nodes = list(range(total))
    fwd, bwd = build_structure(v_nodes, v_edges, flags)
    blocks = get_blocks(v_nodes, fwd, bwd)
    print(f"  V-blocks: {len(blocks)} Sizes={[len(b) for b in blocks]}")
    if len(blocks) == 1 and len(blocks[0]) == total:
        return blocks[0]
    return connect_blocks(blocks, v_edges, e_cache, flags, total)

def validate_vertical(seq, e_cache, v_data):
    h_map = {}
    for r in v_data:
        i = r["img_id"]
        if i not in h_map: h_map[i] = {}
        h_map[i]["T" if r["side"] == 0 else "B"] = r["phash"]

    errs = []
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i + 1]
        ha, hb = h_map.get(a, {}).get("B"), h_map.get(b, {}).get("T")
        h_ok = (ha is not None and ha == hb)
        p_ok = check_boundary_thirds(e_cache[a]["B"], e_cache[b]["T"])
        if not h_ok or not p_ok:
            errs.append(f"  {i}->{i+1}: hash_ok={h_ok} px_ok={p_ok}")
    if errs:
        print(f"  WARN: V-alignment issues:\n" + "\n".join(errs))
    else:
        print(f"  All {len(seq)-1} V-links verified.")

def combine_h(imgs, overlap):
    min_ht = min(i.height for i in imgs)
    std = [i.convert("RGB").crop((0, 0, i.width, min_ht)) if i.height != min_ht else i.convert("RGB") for i in imgs]

    chunks = []
    for i, im in enumerate(std):
        if i == 0:
            chunks.append(im)
        else:
            o = min(overlap, im.width - 1)
            chunks.append(im.crop((o, 0, im.width, im.height)))

    out = Image.new("RGB", (sum(p.width for p in chunks), min_ht))
    offset = 0
    for p in chunks:
        out.paste(p, (offset, 0))
        offset += p.width
    return out

def combine_v(imgs, overlap):
    min_wd = min(i.width for i in imgs)
    std = [i.convert("RGB").crop((0, 0, min_wd, i.height)) if i.width != min_wd else i.convert("RGB") for i in imgs]

    chunks = []
    for i, im in enumerate(std):
        if i == 0:
            chunks.append(im)
        else:
            o = min(overlap, im.height - 1)
            chunks.append(im.crop((0, o, im.width, im.height)))

    out = Image.new("RGB", (min_wd, sum(p.height for p in chunks)))
    offset = 0
    for p in chunks:
        out.paste(p, (0, offset))
        offset += p.height
    return out

def stitch_patches(src_folder, save_path=None):
    files = get_files(src_folder)
    count = len(files)
    if count == 0: raise FileNotFoundError("Directory is empty or lacks images.")

    rows, cols = get_grid_dims(count)
    req_h_edges = rows * cols * (cols - 1) // rows
    print(f"Found {count} files. Matrix={rows}x{cols} TargetH={req_h_edges}")

    img_list = [Image.open(f) for f in files]
    all_ids = list(range(count))

    print("\n--- H-Pass ---")
    h_ov = find_best_overlap(img_list, "h", req_h_edges, 24, 64)
    h_data = compute_phash_data(img_list, "h", h_ov)
    h_cache = extract_edges(img_list, "h", h_ov)
    h_flags = generate_flags(h_data)
    h_edges = evaluate_pairs(h_data, h_cache, "h")

    safe = sum(1 for v in h_flags.values() if v["valid"])
    print(f"  Edges={len(h_edges)} Safe={safe} Ambiguous={count - safe}\n")

    grid = assemble_grid(all_ids, h_data, h_edges, h_cache, cols)

    if any(len(r) != cols for r in grid):
        raise RuntimeError("Grid dimension failure.")

    print(f"\n  {len(grid)} rows validated.")
    merged_rows = [combine_h([img_list[i] for i in r], h_ov) for r in grid]

    print("\n--- V-Pass ---")
    req_v_edges = len(merged_rows) * (len(merged_rows) - 1)
    v_ov = find_best_overlap(merged_rows, "v", req_v_edges, 24, 64)
    v_data = compute_phash_data(merged_rows, "v", v_ov)
    v_cache = extract_edges(merged_rows, "v", v_ov)
    v_flags = generate_flags(v_data)
    v_edges = evaluate_pairs(v_data, v_cache, "v")

    print(f"  Edges={len(v_edges)} (Target={req_v_edges})")

    v_order = sort_vertical(merged_rows, v_data, v_cache, v_flags, v_edges)
    print(f"  Order: {v_order}")

    validate_vertical(v_order, v_cache, v_data)

    print("\n--- Export ---")
    final_img = combine_v([merged_rows[i] for i in v_order], v_ov)

    if save_path:
        final_img.save(save_path)
        print(f"Saved: {save_path} Size={final_img.size}")

    return cv2.cvtColor(np.array(final_img.convert("RGB")), cv2.COLOR_RGB2BGR)

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "patches_out"
    out_file = "final_stitched.png"
    matrix_bgr = stitch_patches(target_dir, out_file)
    print(f"Finished. BGR shape: {matrix_bgr.shape}")
