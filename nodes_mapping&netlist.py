"""
Robust wire-connection detection for circuit diagrams.

Features:
- Heuristic detection of coordinate format (normalized vs pixels, center vs top-left)
- Conservative text detection (less destructive) and masking that preserves wire pixels
- Creates binary wire image, small morphological closing, optional skeletonization (thinning)
- Finds nearest wire pixel to each component by BFS from bbox perimeter
- Finds a path along wire pixels between components (BFS). Draws actual routed wire paths.
- Builds nets using union-find and writes readable + SPICE-like netlists
- Saves debug images for inspection

Requirements:
- opencv-python
- numpy
- (optional) opencv-contrib-python for cv2.ximgproc.thinning (fallback uses simple morphological thinning)
"""

import cv2
import numpy as np
import json
from collections import deque, defaultdict
import os
import math

# CONFIG
IMG_PATH = "data/cleaned/circuit_1199.jpg"  # change as needed
DETECTIONS_JSON = "detections.json"  # must contain {"predictions": [ {x,y,width,height,class_name,...}, ... ] }
OUT_DIR = "debug_out"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# Utilities for coordinate handling
# -----------------------
def load_components_with_coord_fix(components_raw, image_shape):
    """
    Heuristics to determine whether detections are:
      - normalized (x,y,width,height between 0..1) OR absolute pixels
      - coordinates given as center x,y OR top-left x,y

    Returns a list of components with fields:
      - 'x_center','y_center','width','height','bbox' (x1,y1,x2,y2), plus original fields
    """
    h, w = image_shape[:2]
    comps = []
    xs = [c["x"] for c in components_raw]
    ws = [c["width"] for c in components_raw]

    # Heuristic: if many xs <= 1.0 -> normalized
    maybe_normalized = (max(xs) <= 1.01) or (np.mean(ws) <= 1.01)
    # Additional check if values look like pixel top-left: many x + width <= image width
    count_top_left_like = 0
    for c in components_raw:
        if c["x"] + c["width"] <= w + 1 and c["y"] + c["height"] <= h + 1:
            count_top_left_like += 1
    maybe_top_left = (count_top_left_like >= 0.6 * len(components_raw)) and not maybe_normalized

    for c in components_raw:
        x = c["x"]
        y = c["y"]
        cw = c["width"]
        ch = c["height"]

        if maybe_normalized:
            x = float(x) * w
            y = float(y) * h
            cw = float(cw) * w
            ch = float(ch) * h

        # If top-left coordinates were detected -> convert to center
        if maybe_top_left:
            x_center = x + cw / 2.0
            y_center = y + ch / 2.0
        else:
            # treat as center by default
            x_center = x
            y_center = y

        x1 = int(round(x_center - cw / 2.0))
        y1 = int(round(y_center - ch / 2.0))
        x2 = int(round(x_center + cw / 2.0))
        y2 = int(round(y_center + ch / 2.0))

        # clamp
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        comp = dict(c)  # copy original
        comp.update({
            "x_center": int(round(x_center)),
            "y_center": int(round(y_center)),
            "width_px": int(round(cw)),
            "height_px": int(round(ch)),
            "bbox": (x1, y1, x2, y2)
        })
        comps.append(comp)
    return comps

# -----------------------
# Text detection (conservative)
# -----------------------
def detect_text_regions(image):
    """Detect text regions with conservative dilation and return boxes."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use adaptive thresholding + invert to pick up strokes reliably
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 25, 10)
    # smaller horizontal dilation to connect letters, but not too large
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(thr, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    H, W = gray.shape
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = ww * hh
        aspect = ww / hh if hh > 0 else 0
        # permissive but exclude very large non-text blobs and tiny specks
        if 6 < hh < 80 and 10 < ww < 400 and area < 15000:
            margin = 4
            boxes.append((max(0, x - margin), max(0, y - margin),
                          min(W, x + ww + margin), min(H, y + hh + margin)))
    # debug visualization
    dbg = image.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.imwrite(os.path.join(OUT_DIR, "debug_text_boxes.jpg"), dbg)
    print(f"[INFO] detected {len(boxes)} text regions")
    return boxes

# -----------------------
# Create wire mask preserving wire pixels under text
# -----------------------
def create_wire_mask(image, components, text_boxes):
    """Return a mask (255 = keep) that excludes components but does not obliterate wire pixels under text."""
    H, W = image.shape[:2]
    mask = np.ones((H, W), dtype=np.uint8) * 255

    # Exclude components by filling bbox area (with small margin)
    for comp in components:
        x1, y1, x2, y2 = comp["bbox"]
        margin = 4
        x1m = max(0, x1 - margin)
        y1m = max(0, y1 - margin)
        x2m = min(W - 1, x2 + margin)
        y2m = min(H - 1, y2 + margin)
        cv2.rectangle(mask, (x1m, y1m), (x2m, y2m), 0, -1)

    # Mask text areas but preserve underlying white pixels that look like wires:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    for tx1, ty1, tx2, ty2 in text_boxes:
        tx1c, ty1c = tx1, ty1
        tx2c, ty2c = tx2, ty2
        region = binary[ty1c:ty2c, tx1c:tx2c]  # text-like strokes (white in binary)
        if region.size == 0:
            continue
        # Only mask the strong stroke pixels (assumed text); preserve other pixels (possible wire pixels)
        stroke_mask = (region > 128).astype(np.uint8) * 255
        # invert stroke_mask to apply: 255 where we keep, 0 where we mask
        keep = 255 - stroke_mask
        mask[ty1c:ty2c, tx1c:tx2c] = cv2.bitwise_and(mask[ty1c:ty2c, tx1c:tx2c], keep)
    cv2.imwrite(os.path.join(OUT_DIR, "debug_wire_mask.jpg"), mask)
    return mask

# -----------------------
# Binary wires extraction + small morphology + thinning
# -----------------------
def extract_wires_binary(image, mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # apply mask: keep only areas allowed
    binary_wires = cv2.bitwise_and(binary_inv, binary_inv, mask=mask)

    # small morphological closing to bridge tiny gaps; do horizontal and vertical small closes separately
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    binary_wires = cv2.morphologyEx(binary_wires, cv2.MORPH_CLOSE, h_kernel, iterations=1)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    binary_wires = cv2.morphologyEx(binary_wires, cv2.MORPH_CLOSE, v_kernel, iterations=1)

    # optional gaussian blur + threshold to remove tiny specks (tunable)
    # binary_wires = cv2.GaussianBlur(binary_wires, (3,3), 0)
    _, binary_wires = cv2.threshold(binary_wires, 128, 255, cv2.THRESH_BINARY)

    cv2.imwrite(os.path.join(OUT_DIR, "binary_wires_prethin.jpg"), binary_wires)

    # Try cv2.ximgproc thinning if available; otherwise fallback to a simple morphological thinning-ish approach
    skeleton = None
    try:
        import cv2.ximgproc as ximg
        skeleton = ximg.thinning(binary_wires)
        # ensure binary format
        _, skeleton = cv2.threshold(skeleton, 128, 255, cv2.THRESH_BINARY)
        print("[INFO] used ximgproc.thinning for skeletonization")
    except Exception:
        # Zhang-Suen or simple iterative morphological thinning approximation
        skel = np.zeros_like(binary_wires)
        temp = binary_wires.copy()
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            open_img = cv2.morphologyEx(temp, cv2.MORPH_OPEN, element)
            temp2 = cv2.subtract(temp, open_img)
            eroded = cv2.erode(temp, element)
            skel = cv2.bitwise_or(skel, temp2)
            temp = eroded.copy()
            if cv2.countNonZero(temp) == 0:
                break
        skeleton = skel
        print("[INFO] used fallback morphological skeletonization")

    cv2.imwrite(os.path.join(OUT_DIR, "binary_wires_skeleton.jpg"), skeleton)
    return binary_wires, skeleton

# -----------------------
# Find nearest wire pixel to component bbox (BFS outward from bbox perimeter)
# -----------------------
def find_nearest_wire_pixel(binary_img, bbox, max_search=60):
    """
    BFS from bbox perimeter outward to find the nearest white pixel in binary_img.
    Returns (x,y) or None.
    """
    H, W = binary_img.shape
    x1, y1, x2, y2 = bbox
    q = deque()
    visited = set()
    # add perimeter points
    for x in range(x1, x2 + 1):
        q.append((x, y1, 0))
        q.append((x, y2, 0))
        visited.add((x, y1)); visited.add((x, y2))
    for y in range(y1, y2 + 1):
        q.append((x1, y, 0))
        q.append((x2, y, 0))
        visited.add((x1, y)); visited.add((x2, y))
    dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]
    while q:
        x, y, d = q.popleft()
        if d > max_search:
            continue
        if 0 <= x < W and 0 <= y < H:
            if binary_img[y, x] > 128:
                return (x, y)
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny, d + 1))
    return None

# -----------------------
# Pathfinding on the skeleton/binary wires (BFS). Returns list of pixels from start->goal
# -----------------------
def find_wire_path(binary_img, start, goal, max_nodes=200000):
    """
    BFS on binary image to find an actual path along white pixels.
    start, goal: (x,y) coordinates (must be on white pixels)
    """
    H, W = binary_img.shape
    sx, sy = start; gx, gy = goal
    if not (0 <= sx < W and 0 <= sy < H and 0 <= gx < W and 0 <= gy < H):
        return None
    if binary_img[sy, sx] <= 128 or binary_img[gy, gx] <= 128:
        return None

    q = deque()
    q.append((sx, sy))
    parent = { (sx, sy): None }
    dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]
    nodes = 0
    max_allowed = max_nodes
    while q:
        x, y = q.popleft()
        nodes += 1
        if nodes > max_allowed:
            return None
        if (x, y) == (gx, gy):
            # reconstruct path
            path = []
            cur = (x, y)
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and binary_img[ny, nx] > 128 and (nx, ny) not in parent:
                parent[(nx, ny)] = (x, y)
                q.append((nx, ny))
    return None

# -----------------------
# Main detection pipeline
# -----------------------
def detect_wires_and_connections(image, components_raw):
    # 1) fix coordinates / compute bbox
    components = load_components_with_coord_fix(components_raw, image.shape)
    print(f"[INFO] normalized components: {len(components)}")

    # 2) text detection
    text_boxes = detect_text_regions(image)

    # 3) create mask that excludes components and text stroke pixels but preserves wires
    mask = create_wire_mask(image, components, text_boxes)

    # 4) binary wires + skeleton
    binary_wires, skeleton = extract_wires_binary(image, mask)

    # select map to pathfind on (prefer skeleton if it contains pixels)
    path_img = skeleton if np.count_nonzero(skeleton) > 0 else binary_wires

    # 5) find nearest wire pixel for each component
    comp_wire_points = {}
    for idx, comp in enumerate(components):
        pt = find_nearest_wire_pixel(path_img, comp["bbox"], max_search=80)
        # if no point found on skeleton, try fallback to binary_wires
        if pt is None and path_img is skeleton:
            pt = find_nearest_wire_pixel(binary_wires, comp["bbox"], max_search=80)
        comp_wire_points[idx] = pt
        print(f"[INFO] comp {idx} wire point: {pt}")

    # 6) attempt to find paths between component wire points
    connections = []  # list of (compA_idx, compB_idx, path_pixels)
    checked_pairs = set()
    N = len(components)
    for i in range(N):
        for j in range(i + 1, N):
            if (i,j) in checked_pairs:
                continue
            p1 = comp_wire_points.get(i)
            p2 = comp_wire_points.get(j)
            if p1 is None or p2 is None:
                continue
            # guard: skip if Euclidean distance too large (tunable)
            dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
            euclid = math.hypot(dx, dy)
            if euclid > 700:  # skip extremely far pairs
                continue
            path = find_wire_path(path_img, p1, p2)
            if path is None:
                # as a last resort, try on binary_wires (non-skeleton)
                if path_img is not binary_wires:
                    path = find_wire_path(binary_wires, p1, p2)
            if path is not None:
                # optional sanity: path length should not be extremely larger than euclid
                if len(path) <= 15 * max(1.0, euclid):
                    connections.append((i, j, path))
                    print(f"[OK] connection found {i} <-> {j}, path len {len(path)}")
                else:
                    print(f"[WARN] path found but likely spurious (too long): {i} <-> {j}, len {len(path)}, euclid {euclid}")
            checked_pairs.add((i,j))

    print(f"[INFO] total connections found: {len(connections)}")

    # return detailed results
    return {
        "components": components,
        "comp_wire_points": comp_wire_points,
        "connections": connections,
        "binary_wires": binary_wires,
        "skeleton": skeleton,
        "mask": mask,
        "text_boxes": text_boxes
    }

# -----------------------
# Draw results and build netlist
# -----------------------
def draw_connections(image, results):
    out = image.copy()
    # draw skeleton overlay faintly
    skeleton = results["skeleton"]
    if skeleton is not None:
        sk_col = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
        overlay = out.copy()
        overlay[sk_col[:,:,0] > 0] = (200, 200, 200)  # light gray for skeleton
        out = cv2.addWeighted(overlay, 0.2, out, 0.8, 0)

    # draw component boxes and wire points
    for idx, comp in enumerate(results["components"]):
        x1, y1, x2, y2 = comp["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 128, 0), 1)
        pt = results["comp_wire_points"].get(idx)
        if pt is not None:
            cv2.circle(out, pt, 4, (0, 0, 255), -1)

    # draw paths
    for (i, j, path) in results["connections"]:
        pts = np.array(path, dtype=np.int32)
        if pts.shape[0] >= 2:
            cv2.polylines(out, [pts], False, (0, 255, 0), 2)
            cv2.circle(out, tuple(pts[0]), 4, (255, 0, 0), -1)
            cv2.circle(out, tuple(pts[-1]), 4, (255, 0, 0), -1)

    cv2.imwrite(os.path.join(OUT_DIR, "connections_result.jpg"), out)
    return out

def build_nets_from_connections(num_components, connections):
    # union-find
    parent = {i: i for i in range(num_components)}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    for a, b, _ in connections:
        union(a, b)

    nets = defaultdict(list)
    for i in range(num_components):
        nets[find(i)].append(i)
    return list(nets.values())

def export_netlist_readable(components, nets, connections, output_file="netlist.txt"):
    with open(output_file, "w") as f:
        f.write("=== CIRCUIT NETLIST (readable) ===\n\n")
        f.write(f"Total components: {len(components)}\n")
        f.write(f"Total nets: {len(nets)}\n\n")
        for n_idx, n_comps in enumerate(nets):
            f.write(f"Net {n_idx}:\n")
            for comp_idx in n_comps:
                f.write(f"  - {components[comp_idx].get('class_name','unknown')} (ID {comp_idx}) bbox={components[comp_idx]['bbox']}\n")
            f.write("\n")
        f.write("Connections (direct):\n")
        for a,b,_ in connections:
            f.write(f"  {a} <-> {b}\n")
    print(f"[INFO] written readable netlist to {output_file}")

def generate_simple_spice(components, nets, output_file="netlist.sp"):
    with open(output_file, "w") as f:
        f.write("* Auto-generated simple SPICE-like netlist\n\n")
        net_names = {i: f"n{i}" for i in range(len(nets))}
        comp_to_netnames = {}
        for net_idx, comp_idxs in enumerate(nets):
            for ci in comp_idxs:
                comp_to_netnames.setdefault(ci, []).append(net_names[net_idx])

        counter = defaultdict(int)
        for idx, comp in enumerate(components):
            ctype = comp.get("class_name", "X").lower()
            counter[ctype] += 1
            name = f"{ctype[0].upper()}{counter[ctype]}"
            nets_for_comp = comp_to_netnames.get(idx, ["0"])
            if ctype in ("resistor", "res"):
                a = nets_for_comp[0] if len(nets_for_comp)>=1 else "0"
                b = nets_for_comp[1] if len(nets_for_comp)>=2 else "0"
                f.write(f"{name} {a} {b} 1k\n")
            elif ctype in ("capacitor", "cap"):
                a = nets_for_comp[0] if len(nets_for_comp)>=1 else "0"
                b = nets_for_comp[1] if len(nets_for_comp)>=2 else "0"
                f.write(f"{name} {a} {b} 1u\n")
            elif ctype in ("inductor", "ind"):
                a = nets_for_comp[0] if len(nets_for_comp)>=1 else "0"
                b = nets_for_comp[1] if len(nets_for_comp)>=2 else "0"
                f.write(f"{name} {a} {b} 1m\n")
            elif "source" in ctype or "voltage" in ctype or "battery" in ctype:
                a = nets_for_comp[0] if len(nets_for_comp)>=1 else "0"
                b = nets_for_comp[1] if len(nets_for_comp)>=2 else "0"
                f.write(f"{name} {a} {b} DC 5V\n")
            else:
                # generic
                f.write(f"* {name} type={ctype} nets={' '.join(nets_for_comp)}\n")
        f.write("\n.end\n")
    print(f"[INFO] written SPICE-like netlist to {output_file}")

# -----------------------
# Run as script
# -----------------------
if __name__ == "__main__":
    # load image
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"Image not found at {IMG_PATH}")

    # load detections
    if not os.path.exists(DETECTIONS_JSON):
        raise FileNotFoundError(f"Detections JSON not found: {DETECTIONS_JSON}")
    with open(DETECTIONS_JSON, "r") as f:
        data = json.load(f)
    components_raw = data.get("predictions", data.get("components", []))
    if not components_raw:
        raise RuntimeError("No components found in detections json under 'predictions' or 'components'")

    print("[INFO] starting detection pipeline...")
    results = detect_wires_and_connections(img, components_raw)

    # draw & save
    out_img = draw_connections(img, results)
    cv2.imwrite(os.path.join(OUT_DIR, "final_overlay.jpg"), out_img)

    # build netlist
    nets = build_nets_from_connections(len(results["components"]), results["connections"])
    export_netlist_readable(results["components"], nets, results["connections"], os.path.join(OUT_DIR, "netlist.txt"))
    generate_simple_spice(results["components"], nets, os.path.join(OUT_DIR, "netlist.sp"))

    # Write some debug files
    cv2.imwrite(os.path.join(OUT_DIR, "binary_wires.jpg"), results["binary_wires"])
    if results["skeleton"] is not None:
        cv2.imwrite(os.path.join(OUT_DIR, "skeleton.jpg"), results["skeleton"])

    print("[DONE] outputs written to:", OUT_DIR)
