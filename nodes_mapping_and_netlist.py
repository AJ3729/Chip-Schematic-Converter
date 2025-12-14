import cv2
import json
import numpy as np
from collections import defaultdict

# =========================
# FILE PATHS
# =========================
IMG_PATH = "data/cleaned/circuit_1199.jpg"
DETECTIONS_PATH = "detections.json"

# =========================
# CLASSES THAT ARE NOT WIRES
# =========================
NON_WIRE_CLASSES = {
    "resistor",
    "capacitor",
    "inductor",
    "DC Supply",
    "Independent DC Current",
    "ground"
}

# =========================
# LOAD IMAGE + DETECTIONS
# =========================
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"Could not load image: {IMG_PATH}")

h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

with open(DETECTIONS_PATH, "r") as f:
    data = json.load(f)

predictions = data["predictions"]

# =========================
# TEXT MASK (CONNECTED COMPONENT HEURISTIC)
# =========================
def detect_text_mask(gray_img):
    """
    Binary mask (255=text) using connected-component heuristics.
    Designed to capture handwritten labels (e.g., 10Ω, 15mH, 40V).
    """
    bin_inv = cv2.adaptiveThreshold(
        gray_img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10
    )

    # connect characters slightly
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_inv = cv2.dilate(bin_inv, k, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_inv, connectivity=8)
    text_mask = np.zeros_like(gray_img, dtype=np.uint8)

    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        aspect = ww / hh if hh > 0 else 0

        if (20 < area < 2500 and
            0.2 < aspect < 6.0 and
            hh < 50 and ww < 120):
            text_mask[y:y+hh, x:x+ww] = 255

    return text_mask

# =========================
# BUILD NON-WIRE MASK (components + text)
# =========================
non_wire_mask = np.zeros((h, w), dtype=np.uint8)

text_mask = detect_text_mask(gray)
cv2.imwrite("01b_text_mask.png", text_mask)
non_wire_mask = cv2.bitwise_or(non_wire_mask, text_mask)

for det in predictions:
    if det["class_name"] not in NON_WIRE_CLASSES:
        continue

    cx, cy = det["x"], det["y"]
    bw, bh = det["width"], det["height"]

    x1 = int(cx - bw / 2)
    y1 = int(cy - bh / 2)
    x2 = int(cx + bw / 2)
    y2 = int(cy + bh / 2)

    pad = 8  # slightly larger than before to fully remove symbols
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    non_wire_mask[y1:y2, x1:x2] = 255

cv2.imwrite("01_non_wire_mask.png", non_wire_mask)

# =========================
# WIRE EXTRACTION
# =========================
wire_candidate = gray.copy()
wire_candidate[non_wire_mask > 0] = 255
cv2.imwrite("02_wire_candidates.png", wire_candidate)

edges = cv2.Canny(wire_candidate, 40, 120)

k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
wires = cv2.dilate(edges, k, iterations=1)
wires = cv2.morphologyEx(wires, cv2.MORPH_OPEN, k)

# remove tiny blobs
num, labels, stats, _ = cv2.connectedComponentsWithStats(wires, connectivity=8)
clean_wires = np.zeros_like(wires)

for i in range(1, num):
    area = stats[i, cv2.CC_STAT_AREA]
    if area >= 60:
        clean_wires[labels == i] = 255

cv2.imwrite("03_wire_binary.png", clean_wires)

overlay = img.copy()
overlay[clean_wires > 0] = (0, 255, 0)
result = cv2.addWeighted(img, 0.65, overlay, 0.35, 0)
cv2.imwrite("04_wire_overlay.png", result)

# =========================
# BUILD WIRE NODES
# =========================
def build_wire_nodes(clean_wires_bin):
    num_labels, labels = cv2.connectedComponents((clean_wires_bin > 0).astype(np.uint8), connectivity=8)
    node_map = labels.astype(np.int32) - 1  # background -> -1
    return node_map, num_labels - 1

node_map, num_nodes = build_wire_nodes(clean_wires)
print(f"[INFO] wire nodes detected: {num_nodes}")

# =========================
# COMPONENT -> NODE ASSOCIATION (WHITESPACE-AWARE)
# =========================
def bbox_xyxy(det):
    cx, cy = det["x"], det["y"]
    bw, bh = det["width"], det["height"]
    x1 = int(round(cx - bw/2))
    y1 = int(round(cy - bh/2))
    x2 = int(round(cx + bw/2))
    y2 = int(round(cy + bh/2))
    return x1, y1, x2, y2

def collect_nodes_in_rect(node_map, x1, y1, x2, y2):
    h, w = node_map.shape
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    region = node_map[y1:y2, x1:x2]
    ids = region[region != -1]
    if ids.size == 0:
        return {}
    # count pixels per node id
    uniq, counts = np.unique(ids, return_counts=True)
    return {int(u): int(c) for u, c in zip(uniq, counts)}

def find_component_nodes(det, node_map, max_expand=22):
    """
    Finds the most likely node(s) connected to a component.
    Strategy:
      - Expand bbox outward gradually (handles whitespace gaps)
      - As soon as any nodes are found, rank by pixel support and take top 2.
    """
    x1, y1, x2, y2 = bbox_xyxy(det)

    for expand in range(2, max_expand + 1, 2):
        hits = collect_nodes_in_rect(
            node_map,
            x1 - expand, y1 - expand,
            x2 + expand, y2 + expand
        )
        if hits:
            # pick nodes with strongest pixel evidence
            ranked = sorted(hits.items(), key=lambda kv: kv[1], reverse=True)
            nodes = [nid for nid, _ in ranked[:2]]
            # enforce exactly 2 terminals for 2-terminal parts
            if len(nodes) == 1:
                nodes = [nodes[0], nodes[0]]  # same node on both ends (rare but valid)
            elif len(nodes) == 0:
                nodes = [None, None]
            return nodes

    return [None, None]

def build_component_pin_nets(predictions, node_map):
    comps = []
    for i, det in enumerate(predictions):
        cls = det["class_name"].lower()

        # ground: we only need 1 node; still store as [n, n] for consistency
        if "ground" in cls:
            nodes = find_component_nodes(det, node_map, max_expand=26)
            n = nodes[0] if nodes[0] is not None else nodes[1]
            comps.append({"id": i, "class_name": det["class_name"], "nodes": [n, n]})
            continue

        nodes = find_component_nodes(det, node_map, max_expand=22)
        comps.append({"id": i, "class_name": det["class_name"], "nodes": nodes})

    return comps

components_with_nodes = build_component_pin_nets(predictions, node_map)

# =========================
# FORCE GROUND NODE = 0
# =========================
def build_node_name_map(components_with_nodes):
    ground_node = None
    for c in components_with_nodes:
        if "ground" in c["class_name"].lower():
            if c["nodes"][0] is not None:
                ground_node = c["nodes"][0]
                break

    if ground_node is None:
        raise RuntimeError("No ground-connected node found (ground symbol did not touch any wire node).")

    # assign names
    all_nodes = sorted({n for c in components_with_nodes for n in c["nodes"] if n is not None})
    node_name_map = {}
    next_idx = 1
    for n in all_nodes:
        if n == ground_node:
            node_name_map[n] = "0"
        else:
            node_name_map[n] = f"n{next_idx}"
            next_idx += 1
    return node_name_map

node_name_map = build_node_name_map(components_with_nodes)

for c in components_with_nodes:
    c["node_names"] = [
        node_name_map[n] if n is not None else None
        for n in c["nodes"]
    ]

# =========================
# EXPORT NETLISTS (NO OCR VALUES)
# =========================
def export_readable_netlist(components_with_nodes, out_path="netlist_readable.txt"):
    with open(out_path, "w") as f:
        f.write("=== NETLIST (NO OCR VALUES) ===\n")
        f.write("nodes are derived from wire connected-components\n\n")
        for c in components_with_nodes:
            f.write(f"ID {c['id']:02d}  {c['class_name']:<24}  nodes={c['nodes']}  names={c['node_names']}\n")
    print(f"[OK] wrote {out_path}")

def export_spice_netlist(components_with_nodes, out_path="netlist.sp"):
    counters = defaultdict(int)

    with open(out_path, "w") as f:
        f.write("* Auto-generated SPICE netlist (NO OCR values)\n")
        f.write("* Values are placeholders.\n\n")

        for c in components_with_nodes:
            cls = c["class_name"].lower()
            a, b = c["node_names"]

            if "ground" in cls:
                continue

            # If a node is missing, still write but comment it (so you can see it)
            if a is None or b is None:
                f.write(f"* UNSNAPPED {c['class_name']} nodes={c['nodes']}\n")
                continue

            if "resistor" in cls:
                counters["R"] += 1
                f.write(f"R{counters['R']} {a} {b} 1k\n")
            elif "capacitor" in cls:
                counters["C"] += 1
                f.write(f"C{counters['C']} {a} {b} 1u\n")
            elif "inductor" in cls:
                counters["L"] += 1
                f.write(f"L{counters['L']} {a} {b} 1m\n")
            elif "dc supply" in cls or "voltage" in cls or "supply" in cls:
                counters["V"] += 1
                f.write(f"V{counters['V']} {a} {b} DC 5V\n")
            elif "current" in cls:
                counters["I"] += 1
                f.write(f"I{counters['I']} {a} {b} DC 1mA\n")
            else:
                counters["X"] += 1
                f.write(f"* X{counters['X']} ({c['class_name']}) {a} {b}\n")

        f.write("\n.op\n.end\n")

    print(f"[OK] wrote {out_path}")

export_readable_netlist(components_with_nodes, "netlist_readable.txt")
export_spice_netlist(components_with_nodes, "netlist.sp")

# =========================
# DEBUG OVERLAY (wires + bboxes + node hits)
# =========================
debug = img.copy()

# wires
debug[clean_wires > 0] = (0, 255, 0)

# bboxes + node markers
for det in predictions:
    x1, y1, x2, y2 = bbox_xyxy(det)
    cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 0, 0), 2)

for c in components_with_nodes:
    x1, y1, x2, y2 = bbox_xyxy(predictions[c["id"]])
    # show node hits as circles near bbox center-left/right for visibility
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    a, b = c["nodes"]
    # cyan if snapped, red if missing
    if a is not None:
        cv2.circle(debug, (cx - 10, cy), 5, (255, 255, 0), -1)
    else:
        cv2.circle(debug, (cx - 10, cy), 5, (0, 0, 255), -1)

    if b is not None:
        cv2.circle(debug, (cx + 10, cy), 5, (255, 255, 0), -1)
    else:
        cv2.circle(debug, (cx + 10, cy), 5, (0, 0, 255), -1)

cv2.imwrite("06_netlist_debug_overlay.png", debug)
print("[OK] wrote 06_netlist_debug_overlay.png")
