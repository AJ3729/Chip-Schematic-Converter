#!/usr/bin/env python3
"""Localhost demo UI: drop a hand-drawn schematic, walk through every
pipeline stage with a render per stage.

Runs the REAL pipeline (local YOLO weights, current config) — nothing
is mocked. Outputs land in demo/runs/<id>/ and are served back to the
single-page UI in demo/static/index.html.

    ./venv/bin/python demo/app.py     # http://localhost:5001
"""

from __future__ import annotations

import colorsys
import json
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from flask import Flask, jsonify, request, send_from_directory  # noqa: E402

from schematic2netlist.classes import class_terminals  # noqa: E402
from schematic2netlist.config import load_config  # noqa: E402
from schematic2netlist.detect import detect_ultralytics  # noqa: E402
from schematic2netlist.netlist import (  # noqa: E402
    assign_node_names,
    build_node_name_map,
    export_spice_netlist,
)
from schematic2netlist.nodes import (  # noqa: E402
    bbox_xyxy,
    build_wire_nodes,
    build_wire_nodes_crossover_aware,
)
from schematic2netlist.preprocess import preprocess_image_meta  # noqa: E402
from schematic2netlist.repair import build_ledger, repair_circuit  # noqa: E402
from schematic2netlist.simulate import run_ngspice_diag  # noqa: E402
from schematic2netlist.snapping import build_component_pin_nets  # noqa: E402
from schematic2netlist.textmask import detect_text_mask  # noqa: E402
from schematic2netlist.wires import (  # noqa: E402
    build_non_wire_mask,
    extract_wires,
    stitch_wire_islands,
    stitchable_mask,
)

app = Flask(__name__, static_folder="static", static_url_path="")
RUNS = REPO / "demo" / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
CFG = load_config()

EXAMPLES = ["circuit_1.jpg", "circuit_251.jpg", "circuit_619.jpg", "circuit_995.jpg"]

CLASS_COLORS: dict[str, tuple] = {}


def _color_for_class(name: str):
    if name not in CLASS_COLORS:
        h = (hash(name) % 360) / 360.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
        CLASS_COLORS[name] = (int(b * 255), int(g * 255), int(r * 255))
    return CLASS_COLORS[name]


def _net_palette(n: int):
    cols = []
    for i in range(max(n, 1)):
        r, g, b = colorsys.hsv_to_rgb((i * 0.61803) % 1.0, 0.85, 0.95)
        cols.append((int(b * 255), int(g * 255), int(r * 255)))
    return cols


def _save(run: Path, name: str, img) -> str:
    cv2.imwrite(str(run / name), img)
    return name


def _bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def process(image_bytes: bytes, run: Path) -> dict:
    stages = []

    def stage(sid, title, caption, image_name, extra=None):
        stages.append({"id": sid, "title": title, "caption": caption,
                       "image": image_name, "extra": extra or {}})

    # ---- 0. original ----
    arr = np.frombuffer(image_bytes, np.uint8)
    orig = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if orig is None:
        raise ValueError("could not decode image")
    _save(run, "00_original.png", orig)
    stage("original", "Original drawing",
          "The photo as uploaded — skewed, shadowed, hand-drawn.",
          "00_original.png")

    # ---- 1. preprocess ----
    tmp = run / "_upload.png"
    cv2.imwrite(str(tmp), orig)
    canvas, meta = preprocess_image_meta(str(tmp), CFG)
    tmp.unlink()
    clean_path = run / "01_cleaned.png"
    cv2.imwrite(str(clean_path), canvas)
    stage("preprocess", "Preprocessing",
          f"Deskewed {meta['angle_deg']:+.1f}°, shadow-normalized, binarized, "
          f"cropped and scaled onto a {meta['target_size']} px canvas.",
          "01_cleaned.png")

    img = _bgr(canvas)
    gray = canvas

    # ---- 2. detection ----
    dets = detect_ultralytics([clean_path], CFG)[0]
    vis = img.copy()
    for d in dets:
        x1, y1, x2, y2 = bbox_xyxy(d)
        col = _color_for_class(d["class"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        cv2.putText(vis, f"{d['class']} {d['confidence']:.2f}",
                    (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)
    stage("detect", "Component detection",
          f"Local YOLOv8s finds {len(dets)} symbols across 17 classes — "
          "including wire crossovers.",
          _save(run, "02_detect.png", vis))

    # ---- 3. masking ----
    text_mask = detect_text_mask(gray, CFG) if CFG["textmask"]["enabled"] else None
    non_wire = build_non_wire_mask(gray, dets, CFG, text_mask)
    vis = img.copy()
    overlay = img.copy()
    overlay[non_wire > 0] = (60, 60, 230)
    if text_mask is not None:
        overlay[text_mask > 0] = (230, 140, 40)
    vis = cv2.addWeighted(img, 0.55, overlay, 0.45, 0)
    stage("mask", "Masking non-wire ink",
          "Component bodies (red) and handwriting (blue) are removed so "
          "only conductors remain.",
          _save(run, "03_mask.png", vis))

    # ---- 4. wire extraction ----
    _, wires0 = extract_wires(gray, non_wire, CFG)
    vis = img.copy()
    vis[wires0 > 0] = (80, 200, 80)
    stage("wires", "Wire extraction",
          "Remaining ink is binarized into candidate conductors "
          f"(method: {CFG['wires'].get('method', 'canny')}).",
          _save(run, "04_wires.png", vis))

    # ---- 5. stitching ----
    wires = wires0
    if CFG["wires"].get("stitch_masked_gaps"):
        stitchable = stitchable_mask(gray.shape, dets, CFG, text_mask)
        wires = stitch_wire_islands(wires0, stitchable, CFG)
        bridges = cv2.subtract(wires, wires0)
        vis = img.copy()
        vis[wires0 > 0] = (80, 200, 80)
        vis[bridges > 0] = (230, 60, 200)
        n_br = int(cv2.connectedComponents((bridges > 0).astype(np.uint8))[0]) - 1
        stage("stitch", "Gap stitching",
              f"{max(n_br, 0)} bridge(s) (magenta) reconnect wires split by our "
              "own masking — collinearity-checked, never through a component.",
              _save(run, "05_stitch.png", vis))

    # ---- 6. nets ----
    if CFG["nodes"].get("handle_crossovers"):
        from schematic2netlist.classes import canonical_class
        xb = [d for d in dets if canonical_class(d["class"]) == "Wire Crossover"]
        node_map, n_nodes = build_wire_nodes_crossover_aware(
            wires, xb, connectivity=CFG["nodes"]["connectivity"])
    else:
        node_map, n_nodes = build_wire_nodes(
            wires, connectivity=CFG["nodes"]["connectivity"])
    pal = _net_palette(n_nodes)
    vis = np.full_like(img, 255)
    for nid in range(n_nodes):
        vis[node_map == nid] = pal[nid]
    vis = cv2.addWeighted(img, 0.25, vis, 0.75, 0)
    stage("nets", "Electrical nets",
          f"{n_nodes} connected regions — one colour per electrical net. "
          "Crossing wires stay separate at detected crossovers.",
          _save(run, "06_nets.png", vis))

    # ---- 7. snapping ----
    comps = build_component_pin_nets(dets, node_map, CFG)
    name_map = build_node_name_map(comps, ground_fallback=CFG["netlist"]["ground_fallback"])
    assign_node_names(comps, name_map)
    vis = img.copy()
    for nid in range(n_nodes):
        vis[node_map == nid] = pal[nid]
    vis = cv2.addWeighted(img, 0.45, vis, 0.55, 0)
    n_snap, n_tot = 0, 0
    for c in comps:
        x1, y1, x2, y2 = bbox_xyxy(dets[c["id"]])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (40, 40, 40), 1)
        k = len(c["nodes"])
        for t, node in enumerate(c["nodes"]):
            n_tot += 1
            off = int((t - (k - 1) / 2) * 12)
            cx, cy = (x1 + x2) // 2 + off, (y1 + y2) // 2
            if node is not None:
                n_snap += 1
                cv2.circle(vis, (cx, cy), 5, pal[node], -1)
                cv2.circle(vis, (cx, cy), 5, (30, 30, 30), 1)
            else:
                cv2.drawMarker(vis, (cx, cy), (0, 0, 220),
                               cv2.MARKER_TILTED_CROSS, 10, 2)
        exp = class_terminals(c["class"])
        if exp != k:
            pass
    stage("snap", "Terminal snapping",
          f"{n_snap}/{n_tot} terminals attached — each dot takes the colour "
          "of the net its pin touches on the component boundary.",
          _save(run, "07_snap.png", vis))

    # ---- 8. netlist + repair + simulate ----
    net_path = run / "netlist.sp"
    export_spice_netlist(comps, str(net_path),
                         placeholders=CFG["netlist"]["placeholders"])
    rep = repair_circuit(comps, name_map, CFG) if CFG.get("repair", {}).get("enabled") else None
    rep_path = run / "netlist_repaired.sp"
    export_spice_netlist(comps, str(rep_path),
                         placeholders=CFG["netlist"]["placeholders"],
                         extra_lines=rep.extra_lines if rep else None)
    ok0, cat0, _ = run_ngspice_diag(str(net_path), CFG)
    ok1, cat1, _ = run_ngspice_diag(str(rep_path), CFG)
    ledger = build_ledger("upload", ok0, ok1, rep) if rep else None
    stage("netlist", "SPICE netlist + transparent repair",
          "The recovered topology as a simulatable netlist. The repair "
          "layer adds the minimal, logged assumptions needed for DC "
          "solvability — topology untouched.",
          "07_snap.png",
          extra={
              "netlist": net_path.read_text(),
              "netlist_repaired": rep_path.read_text(),
              "ngspice_before": {"ok": bool(ok0), "category": cat0},
              "ngspice_after": {"ok": bool(ok1), "category": cat1},
              "ledger": ledger,
          })

    return {"stages": stages}


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/examples")
def examples():
    out = [e for e in EXAMPLES if (REPO / "data" / "raw" / e).exists()]
    return jsonify(out)


@app.get("/api/example-thumb/<name>")
def example_thumb(name):
    safe = Path(name).name
    p = REPO / "data" / "raw" / safe
    if not p.exists():
        return ("not found", 404)
    img = cv2.imread(str(p))
    h, w = img.shape[:2]
    s = 160 / max(h, w)
    img = cv2.resize(img, (int(w * s), int(h * s)))
    ok, buf = cv2.imencode(".jpg", img)
    return app.response_class(buf.tobytes(), mimetype="image/jpeg")


@app.post("/api/process")
def api_process():
    run_id = uuid.uuid4().hex[:12]
    run = RUNS / run_id
    run.mkdir(parents=True)
    try:
        if "file" in request.files:
            data = request.files["file"].read()
        else:
            name = Path(request.json.get("example", "")).name
            p = REPO / "data" / "raw" / name
            if not p.exists():
                return jsonify({"error": "unknown example"}), 400
            data = p.read_bytes()
        result = process(data, run)
    except Exception as e:  # noqa: BLE001 — surface to the UI
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
    result["run_id"] = run_id
    (run / "result.json").write_text(json.dumps(
        {k: v for k, v in result.items() if k != "stages"} |
        {"stages": [{k2: v2 for k2, v2 in s.items() if k2 != "extra"}
                    for s in result["stages"]]}))
    return jsonify(result)


@app.get("/runs/<run_id>/<name>")
def run_file(run_id, name):
    return send_from_directory(RUNS / Path(run_id).name, Path(name).name)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
