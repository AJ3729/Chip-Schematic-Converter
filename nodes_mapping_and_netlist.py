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
    "Independent AC Current",
    "AC Supply",
    "diode",
    "Zener Diode",
    "MOSFET Transistor",
    "operational amplifier",
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

predictions = data["detections"]

# =========================
# TEXT MASK (CONNECTED COMPONENT HEURISTIC)
# =========================
def detect_text_mask(gray_img):
    bin_inv = cv2.adaptiveThreshold(
        gray_img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10
    )
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_inv = cv2.dilate(bin_inv, k, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_inv, connectivity=8)
    text_mask = np.zeros_like(gray_img, dtype=np.uint8)

    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        aspect = ww / hh if hh > 0 else 0
        if (
            20 < area < 2500 and
            0.2 < aspect < 6.0 and
            hh < 50 and ww < 120
        ):
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
    if det["class"] not in NON_WIRE_CLASSES:
        continue
    cx, cy = det["x"], det["y"]
    bw, bh = det["width"], det["height"]
    x1 = max(0, int(cx - bw / 2) - 8)
    y1 = max(0, int(cy - bh / 2) - 8)
    x2 = min(w, int(cx + bw / 2) + 8)
    y2 = min(h, int(cy + bh / 2) + 8)
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

num, labels, stats, _ = cv2.connectedComponentsWithStats(wires, connectivity=8)
clean_wires = np.zeros_like(wires)

for i in range(1, num):
    area = stats[i, cv2.CC_STAT_AREA]
    if area >= 20:
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
    num_labels, labels = cv2.connectedComponents(
        (clean_wires_bin > 0).astype(np.uint8),
        connectivity=8
    )
    node_map = labels.astype(np.int32) - 1  # background -> -1
    return node_map, num_labels - 1

node_map, num_nodes = build_wire_nodes(clean_wires)
print(f"[INFO] wire nodes detected: {num_nodes}")

# =========================
# BBOX HELPER
# =========================
def bbox_xyxy(det):
    cx, cy = det["x"], det["y"]
    bw, bh = det["width"], det["height"]
    x1 = int(round(cx - bw / 2))
    y1 = int(round(cy - bh / 2))
    x2 = int(round(cx + bw / 2))
    y2 = int(round(cy + bh / 2))
    return x1, y1, x2, y2

# =========================
# NODE COLLECTION IN RECT
# =========================
def collect_nodes_in_rect(node_map, x1, y1, x2, y2):
    h, w = node_map.shape
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return {}
    region = node_map[y1:y2, x1:x2]
    ids = region[region != -1]
    if ids.size == 0:
        return {}
    uniq, counts = np.unique(ids, return_counts=True)
    return {int(u): int(c) for u, c in zip(uniq, counts)}

# =========================
# DIRECTIONAL TERMINAL SNAPPING
# =========================
def find_component_nodes(det, node_map, max_expand=30):
    x1, y1, x2, y2 = bbox_xyxy(det)
    bw = x2 - x1
    bh = y2 - y1
    horizontal = bw >= bh

    if horizontal:
        left_hits  = collect_nodes_in_rect(node_map, x1 - max_expand, y1, x1 + 4, y2)
        right_hits = collect_nodes_in_rect(node_map, x2 - 4, y1, x2 + max_expand, y2)
    else:
        left_hits  = collect_nodes_in_rect(node_map, x1, y1 - max_expand, x2, y1 + 4)
        right_hits = collect_nodes_in_rect(node_map, x1, y2 - 4, x2, y2 + max_expand)

    def best_node(hits):
        if not hits:
            return None
        return max(hits, key=hits.get)

    return [best_node(left_hits), best_node(right_hits)]

# =========================
# GROUND NODE SNAPPING
# =========================
def find_ground_node(det, node_map, max_expand=40):
    x1, y1, x2, y2 = bbox_xyxy(det)
    for expand in range(2, max_expand + 1, 2):
        hits = collect_nodes_in_rect(
            node_map,
            x1 - expand, y1 - expand,
            x2 + expand, y2 + expand
        )
        if hits:
            return max(hits, key=hits.get)
    return None

# =========================
# BUILD COMPONENT PIN NETS
# =========================
def build_component_pin_nets(predictions, node_map):
    comps = []
    for i, det in enumerate(predictions):
        cls = det["class"].lower()
        if "ground" in cls:
            g = find_ground_node(det, node_map)
            comps.append({
                "id": i,
                "class": det["class"],
                "nodes": [g, g],
                "kind": "ground"
            })
        else:
            nodes = find_component_nodes(det, node_map)
            comps.append({
                "id": i,
                "class": det["class"],
                "nodes": nodes,
                "kind": "two_terminal"
            })
    return comps

components_with_nodes = build_component_pin_nets(predictions, node_map)

# =========================
# DEBUG: PRINT GROUND STATUS
# =========================
for c in components_with_nodes:
    if "ground" in c["class"].lower():
        print(f"[DEBUG] ground component: raw node id = {c['nodes'][0]}")

# =========================
# BUILD NODE NAME MAP
# Guarantees ground node -> "0"
# All other nodes -> "n1", "n2", ...
# Skips "n0" entirely to avoid any collision with "0"
# =========================
def build_node_name_map(components_with_nodes):
    # Step 1: find the raw integer id of the ground node
    ground_raw_id = None
    for c in components_with_nodes:
        if "ground" in c["class"].lower():
            if c["nodes"][0] is not None:
                ground_raw_id = c["nodes"][0]
                break

    # Step 2: fallback — use most-connected node if no ground detected
    if ground_raw_id is None:
        print("[WARN] no ground symbol detected — using most-connected node as ground")
        node_counts = defaultdict(int)
        for c in components_with_nodes:
            for n in c["nodes"]:
                if n is not None:
                    node_counts[n] += 1
        if node_counts:
            ground_raw_id = max(node_counts, key=node_counts.get)
        else:
            ground_raw_id = 0

    print(f"[INFO] ground raw node id: {ground_raw_id}")

    # Step 3: collect all unique raw node ids
    all_raw_nodes = sorted({
        n
        for c in components_with_nodes
        for n in c["nodes"]
        if n is not None
    })

    print(f"[INFO] all raw node ids: {all_raw_nodes}")

    # Step 4: build the name map
    # ground -> "0", everything else -> "n1", "n2", ... (never "n0")
    node_name_map = {}
    counter = 1
    for raw_id in all_raw_nodes:
        if raw_id == ground_raw_id:
            node_name_map[raw_id] = "0"
        else:
            node_name_map[raw_id] = f"n{counter}"
            counter += 1

    print(f"[INFO] node name map: {node_name_map}")
    return node_name_map

node_name_map = build_node_name_map(components_with_nodes)

# Attach human-readable node names to every component
for c in components_with_nodes:
    c["node_names"] = [
        node_name_map[n] if n is not None else None
        for n in c["nodes"]
    ]

# =========================
# EXPORT READABLE NETLIST
# =========================
def export_readable_netlist(components_with_nodes, out_path="netlist_readable.txt"):
    with open(out_path, "w") as f:
        f.write("=== NETLIST (NO OCR VALUES) ===\n")
        f.write("nodes derived from wire connected-components\n\n")
        for c in components_with_nodes:
            f.write(
                f"ID {c['id']:02d}  "
                f"{c['class']:<28}  "
                f"raw={c['nodes']}  "
                f"names={c['node_names']}\n"
            )
    print(f"[OK] wrote {out_path}")

# =========================
# EXPORT SPICE NETLIST
# =========================
def export_spice_netlist(components_with_nodes, out_path="netlist.sp"):
    counters = defaultdict(int)
    skipped = []
    wrote_any = False

    def sanitize(node):
        if node is None:
            return None
        s = str(node).strip().replace(" ", "_")
        return s if s else None

    with open(out_path, "w") as f:
        f.write("* Auto-generated SPICE netlist (NO TEXT OCR USED)\n\n")

        for c in components_with_nodes:
            cls = c["class"].lower()

            # ground symbols are reference markers only — not SPICE elements
            if "ground" in cls:
                continue

            a = sanitize(c["node_names"][0])
            b = sanitize(c["node_names"][1])

            # skip unsnapped terminals
            if a is None or b is None:
                msg = f"* UNSNAPPED {c['class']} raw_nodes={c['nodes']}"
                f.write(msg + "\n")
                skipped.append(msg)
                continue

            # skip same-node (snapping failure)
            if a == b:
                msg = f"* SAME_NODE_SKIPPED {c['class']} both_on={a}"
                f.write(msg + "\n")
                skipped.append(msg)
                continue

            if "resistor" in cls:
                counters["R"] += 1
                f.write(f"R{counters['R']} {a} {b} 1k\n")
                wrote_any = True

            elif "capacitor" in cls:
                counters["C"] += 1
                f.write(f"C{counters['C']} {a} {b} 1u\n")
                wrote_any = True

            elif "inductor" in cls:
                counters["L"] += 1
                f.write(f"L{counters['L']} {a} {b} 1m\n")
                wrote_any = True

            elif "dc supply" in cls or "supply" in cls:
                counters["V"] += 1
                f.write(f"V{counters['V']} {a} {b} DC 5\n")
                wrote_any = True

            elif "ac supply" in cls:
                counters["V"] += 1
                f.write(f"V{counters['V']} {a} {b} AC 1\n")
                wrote_any = True

            elif "dc current" in cls or "independent dc current" in cls:
                counters["I"] += 1
                f.write(f"I{counters['I']} {a} {b} DC 1m\n")
                wrote_any = True

            elif "ac current" in cls or "independent ac current" in cls:
                counters["I"] += 1
                f.write(f"I{counters['I']} {a} {b} AC 1m\n")
                wrote_any = True

            elif "diode" in cls and "zener" not in cls:
                counters["D"] += 1
                f.write(f"D{counters['D']} {a} {b} Ddefault\n")
                wrote_any = True

            elif "zener" in cls:
                counters["Z"] += 1
                f.write(f"D{counters['Z']} {a} {b} Zdefault\n")
                wrote_any = True

            elif "mosfet" in cls:
                counters["M"] += 1
                # MOSFET needs drain gate source body — approximated here
                f.write(f"M{counters['M']} {a} {b} 0 0 NMOS\n")
                wrote_any = True

            else:
                counters["X"] += 1
                f.write(f"* UNKNOWN {c['class']} {a} {b}\n")

        if not wrote_any:
            f.write("* WARNING: no valid components written\n")

        f.write("\n.op\n.end\n")

    print(f"[OK] wrote {out_path}")

    if skipped:
        print(f"[INFO] {len(skipped)} component(s) skipped:")
        for s in skipped:
            print(f"       {s}")

# =========================
# RUN EXPORTS
# =========================
export_readable_netlist(components_with_nodes, "netlist_readable.txt")
export_spice_netlist(components_with_nodes, "netlist.sp")

# =========================
# DIAGNOSTIC SUMMARY
# =========================
total = sum(1 for c in components_with_nodes if "ground" not in c["class"].lower())
snapped_both = sum(
    1 for c in components_with_nodes
    if "ground" not in c["class"].lower()
    and c["node_names"][0] is not None
    and c["node_names"][1] is not None
    and c["node_names"][0] != c["node_names"][1]
)
snapped_one = sum(
    1 for c in components_with_nodes
    if "ground" not in c["class"].lower()
    and None in c["node_names"]
)
same_node = sum(
    1 for c in components_with_nodes
    if "ground" not in c["class"].lower()
    and c["node_names"][0] is not None
    and c["node_names"][0] == c["node_names"][1]
)

print(f"\n[DIAGNOSTIC SUMMARY]")
print(f"  total non-ground components : {total}")
print(f"  fully snapped (both nodes)  : {snapped_both}  ({100*snapped_both/total:.1f}%)" if total else "")
print(f"  partially snapped (one None): {snapped_one}")
print(f"  same-node failures          : {same_node}")
ground_name = next(
    (c["node_names"][0] for c in components_with_nodes if "ground" in c["class"].lower()),
    "NOT FOUND"
)
print(f"  ground node assigned as     : {ground_name}")

# =========================
# DEBUG OVERLAY
# =========================
debug = img.copy()
debug[clean_wires > 0] = (0, 255, 0)

for det in predictions:
    x1, y1, x2, y2 = bbox_xyxy(det)
    cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 0, 0), 2)

for c in components_with_nodes:
    x1, y1, x2, y2 = bbox_xyxy(predictions[c["id"]])
    cx_c = int((x1 + x2) / 2)
    cy_c = int((y1 + y2) / 2)

    col_a = (0, 255, 255) if c["nodes"][0] is not None else (0, 0, 255)
    col_b = (0, 255, 255) if c["nodes"][1] is not None else (0, 0, 255)

    cv2.circle(debug, (cx_c - 10, cy_c), 5, col_a, -1)
    cv2.circle(debug, (cx_c + 10, cy_c), 5, col_b, -1)

cv2.imwrite("06_netlist_debug_overlay.png", debug)
print("\n[OK] wrote 06_netlist_debug_overlay.png")