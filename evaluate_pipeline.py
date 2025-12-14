import os
import csv
import subprocess
import cv2
import json
import numpy as np

# =========================
# CONFIG
# =========================
CLEAN_DIR = "data/cleaned"
DETECTIONS_PATH = "detections.json"   # shared detections file
OUT_CSV = "pipeline_evaluation.csv"
MAX_IMAGES = 100                      # use first 50–100 images
NGSPICE_TIMEOUT = 5                   # seconds

# =========================
# LOAD DETECTIONS
# =========================
with open(DETECTIONS_PATH, "r") as f:
    predictions = json.load(f)["predictions"]

def extract_clean_wires(gray, predictions):
    """
    Evaluation-only reproduction of the pipeline's wire extraction stage.
    Does NOT modify the pipeline code.
    """
    h, w = gray.shape

    NON_WIRE_CLASSES = {
        "resistor",
        "capacitor",
        "inductor",
        "DC Supply",
        "Independent DC Current",
        "ground"
    }

    non_wire_mask = np.zeros((h, w), dtype=np.uint8)

    # remove component regions
    for det in predictions:
        if det["class_name"] not in NON_WIRE_CLASSES:
            continue

        cx, cy = det["x"], det["y"]
        bw, bh = det["width"], det["height"]

        x1 = int(cx - bw / 2) - 8
        y1 = int(cy - bh / 2) - 8
        x2 = int(cx + bw / 2) + 8
        y2 = int(cy + bh / 2) + 8

        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)

        non_wire_mask[y1:y2, x1:x2] = 255

    # suppress non-wire pixels
    wire_candidate = gray.copy()
    wire_candidate[non_wire_mask > 0] = 255

    # edge detection
    edges = cv2.Canny(wire_candidate, 40, 120)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    wires = cv2.dilate(edges, k, iterations=1)
    wires = cv2.morphologyEx(wires, cv2.MORPH_OPEN, k)

    # remove small blobs
    num, labels, stats, _ = cv2.connectedComponentsWithStats(wires, connectivity=8)
    clean_wires = np.zeros_like(wires)

    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= 60:
            clean_wires[labels == i] = 255

    return clean_wires


# =========================
# IMPORT YOUR PIPELINE
# =========================
# These must already exist in your project
from nodes_mapping_and_netlist import (
    build_wire_nodes,
    build_component_pin_nets,
    build_node_name_map,
    export_spice_netlist
)

# =========================
# HELPER: RUN NGSPICE
# =========================
def run_ngspice(netlist_path):
    try:
        proc = subprocess.run(
            ["ngspice", "-b", netlist_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=NGSPICE_TIMEOUT
        )
        out = proc.stdout.decode().lower()
        err = proc.stderr.decode().lower()

        if "singular matrix" in out or "floating" in out:
            return False, "singular_or_floating"
        if proc.returncode != 0:
            return False, "parse_error"
        return True, "ok"

    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception:
        return False, "ngspice_error"

# =========================
# MAIN LOOP
# =========================
rows = []

images = sorted([
    f for f in os.listdir(CLEAN_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])[:MAX_IMAGES]

for idx, img_name in enumerate(images):
    print(f"[{idx+1}/{len(images)}] {img_name}")

    img_path = os.path.join(CLEAN_DIR, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    row = {
        "image": img_name,
        "num_components": len(predictions),
        "num_wire_nodes": 0,
        "terminal_snap_rate": 0.0,
        "fully_connected_rate": 0.0,
        "netlist_generated": 0,
        "ngspice_ok": 0,
        "failure_reason": ""
    }

    try:
        # --- wire extraction ---
        clean_wires = extract_clean_wires(img, predictions)

        # --- node detection ---
        node_map, num_nodes = build_wire_nodes(clean_wires)
        row["num_wire_nodes"] = num_nodes

        # --- terminal snapping ---
        comps = build_component_pin_nets(predictions, node_map)

        total_terms = 0
        snapped_terms = 0
        fully_connected = 0

        for c in comps:
            nodes = c["nodes"]
            total_terms += 2
            snapped_terms += sum(n is not None for n in nodes)
            if all(n is not None for n in nodes):
                fully_connected += 1

        row["terminal_snap_rate"] = snapped_terms / max(1, total_terms)
        row["fully_connected_rate"] = fully_connected / max(1, len(comps))

        # --- netlist ---
        node_name_map = build_node_name_map(comps)
        for c in comps:
            c["node_names"] = [
                node_name_map[n] if n is not None else None
                for n in c["nodes"]
            ]

        netlist_path = f"tmp_{idx}.sp"
        export_spice_netlist(comps, netlist_path)
        row["netlist_generated"] = 1

        # --- ngspice ---
        ok, reason = run_ngspice(netlist_path)
        row["ngspice_ok"] = int(ok)
        row["failure_reason"] = reason if not ok else ""

        os.remove(netlist_path)

    except Exception as e:
        row["failure_reason"] = type(e).__name__

    rows.append(row)

# =========================
# WRITE CSV
# =========================
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\n[OK] Wrote evaluation results to {OUT_CSV}")
