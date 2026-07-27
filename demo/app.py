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
from schematic2netlist.benchmark import align_components  # noqa: E402
from schematic2netlist.classes import class_role  # noqa: E402
from schematic2netlist.gt import gt_to_components, load_gt  # noqa: E402

app = Flask(__name__, static_folder="static", static_url_path="")
RUNS = REPO / "demo" / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
CFG = load_config()

EXAMPLES = ["circuit_1.jpg", "circuit_995.jpg", "circuit_1016.jpg", "circuit_619.jpg"]
GT_DIR = REPO / "data" / "gt_netlists_verified_v2"

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


def _san_net(n: str) -> str:
    return str(n).strip().replace(" ", "_")


def _run_transient(comps: list[dict], nets: list[str], run: Path, tag: str,
                   extra_lines=None):
    """Simulate a transient (5 ms, SIN sources for AC) and return
    (ok, category, time_array, {net: waveform}). AC placeholders are
    swapped for time-domain sine sources so waveforms aren't flat."""
    import subprocess

    ph = dict(CFG["netlist"]["placeholders"])
    ph.update({"ac_supply": "SIN(0 1 1k)", "ac_current": "SIN(0 1m 1k)"})
    base = run / f"sim_{tag}_base.sp"
    export_spice_netlist(comps, str(base), placeholders=ph,
                         extra_lines=extra_lines)
    probes = [f"v({_san_net(n)})" for n in nets if _san_net(n) != "0"]
    data_file = run / f"sim_{tag}.dat"
    control = (
        "\n.options gmin=1e-9 reltol=0.01 abstol=1e-9 itl4=100 method=gear\n"
        ".control\n"
        "set filetype=ascii\n"
        "tran 5u 5m uic\n"
        + (f"wrdata {data_file} " + " ".join(probes) + "\n" if probes else "")
        + ".endc\n.end\n"
    )
    text = base.read_text().replace("\n.op\n.end\n", control)
    # ideal diode/zener models abort transients ("timestep too small");
    # series resistance is the standard numerical softening
    text = text.replace(".model Ddefault D\n",
                        ".model Ddefault D(rs=5)\n")
    text = text.replace(".model Zdefault D(bv=5.1)\n",
                        ".model Zdefault D(bv=5.1 rs=5)\n")
    sim_path = run / f"sim_{tag}.sp"
    sim_path.write_text(text)

    try:
        proc = subprocess.run(
            [CFG["simulate"]["ngspice_binary"], "-b", str(sim_path)],
            capture_output=True, timeout=15)
    except Exception as e:  # noqa: BLE001
        return False, type(e).__name__, None, {}

    from schematic2netlist.simulate import parse_ngspice_output
    out_text = proc.stdout.decode(errors="replace") + \
        proc.stderr.decode(errors="replace")
    ok, cat, _ = parse_ngspice_output(
        proc.stdout.decode(errors="replace"),
        proc.stderr.decode(errors="replace"), proc.returncode)
    if "timestep too small" in out_text.lower():
        cat = "tran_aborted"
    waves: dict[str, np.ndarray] = {}
    t = None
    if data_file.exists() and probes:
        try:
            data = np.loadtxt(str(data_file))
            if data.ndim == 1:
                data = data.reshape(-1, 2)
            t = data[:, 0]
            for i, n in enumerate([n for n in nets if _san_net(n) != "0"]):
                col = 2 * i + 1
                if col < data.shape[1]:
                    waves[n] = data[:, col]
            ok = True
            cat = "ok"
        except Exception:  # noqa: BLE001 — fall through with parse verdict
            pass
    for n in nets:
        if _san_net(n) == "0":
            waves[n] = np.zeros_like(t) if t is not None else None
    return ok, cat, t, waves


def _simulation_stage(comps, dets, run: Path, rep, gt_stem: str | None):
    """Build the waveform-comparison payload (stage 10)."""
    pred_bench = [{
        "id": c["id"], "class": c["class"],
        "nets": list(c.get("node_names", [])),
        "bbox": [dets[c["id"]]["x"], dets[c["id"]]["y"],
                 dets[c["id"]]["width"], dets[c["id"]]["height"]],
    } for c in comps]

    gt_bench = None
    gt_comps_sim = None
    if gt_stem and (GT_DIR / f"{gt_stem}.json").exists():
        gt = load_gt(GT_DIR / f"{gt_stem}.json")
        gt_bench = gt_to_components(gt)
        by_id = {c["id"]: c for c in gt["components"]}
        for c in gt_bench:
            c["bbox"] = by_id[c["id"]]["bbox"]
        gt_comps_sim = [{"id": c["id"], "class": c["class"],
                         "nodes": list(c["nets"]),
                         "node_names": list(c["nets"])} for c in gt_bench]

    # choose probes: matched 2-terminal passives/sources, both sides wired
    PROBE_ROLES = {"resistor", "capacitor", "inductor", "vdc", "vac",
                   "idc", "iac", "diode", "zener"}
    probes = []
    if gt_bench:
        pred_al, gt_al, _ = align_components(pred_bench, gt_bench)
        gt_by_id = {c["id"]: c for c in gt_al}
        for pc in pred_al:
            gc = gt_by_id.get(pc["id"])
            if not gc or class_role(pc["class"]) not in PROBE_ROLES:
                continue
            if (len([n for n in pc["nets"][:2] if n]) == 2
                    and len([n for n in gc["nets"][:2] if n]) == 2):
                probes.append({"class": pc["class"],
                               "pred_nets": pc["nets"][:2],
                               "gt_nets": gc["nets"][:2]})
            if len(probes) >= 4:
                break
    else:
        for pc in pred_bench:
            if class_role(pc["class"]) in PROBE_ROLES \
                    and len([n for n in pc["nets"][:2] if n]) == 2:
                probes.append({"class": pc["class"],
                               "pred_nets": pc["nets"][:2], "gt_nets": None})
            if len(probes) >= 4:
                break

    pred_nets = sorted({n for p in probes for n in p["pred_nets"]})
    ok_p, cat_p, t_p, w_p = _run_transient(
        comps, pred_nets, run, "pred",
        extra_lines=rep.extra_lines if rep else None)

    gt_sim = None
    t_g, w_g = None, {}
    if gt_comps_sim is not None:
        gt_rep = repair_circuit(gt_comps_sim, {}, CFG) \
            if CFG.get("repair", {}).get("enabled") else None
        gt_nets = sorted({n for p in probes if p["gt_nets"]
                          for n in p["gt_nets"]})
        ok_g, cat_g, t_g, w_g = _run_transient(
            gt_comps_sim, gt_nets, run, "gt",
            extra_lines=gt_rep.extra_lines if gt_rep else None)
        gt_sim = {"ok": bool(ok_g), "category": cat_g}

    # ngspice's adaptive time-stepping gives each sim a different number
    # of points — resample both onto one shared uniform grid so traces
    # are comparable point-for-point. NEVER extrapolate beyond what a
    # sim actually computed (an aborted transient would otherwise be
    # silently flat-extended): the grid ends at the earliest sim end.
    ends = [t[-1] for t in (t_p, t_g) if t is not None and len(t)]
    t_max = min(ends) if ends else 0.0
    grid = np.linspace(0.0, max(t_max, 1e-12), 250)
    partial = t_max < 4.5e-3   # aborted before ~90% of the 5 ms window

    def series(t, waves, nets):
        if t is None or nets is None:
            return None
        a, b = nets
        va = waves.get(a)
        vb = waves.get(b)
        if va is None or vb is None:
            return None
        return np.interp(grid, t, va - vb)

    out_probes = []
    for i, p in enumerate(probes):
        vp = series(t_p, w_p, p["pred_nets"])
        vg = series(t_g, w_g, p["gt_nets"]) if p["gt_nets"] else None
        match = None
        if vp is not None and vg is not None:
            # terminal order on 2-terminal parts is arbitrary on BOTH
            # sides, so the probe polarity is unordered — score both
            # orientations and display the better one
            rng = float(vg.max() - vg.min()) or 1e-9
            def m(v):
                return max(0.0, 1.0 - float(np.sqrt(((v - vg) ** 2).mean())) / rng)
            if m(-vp) > m(vp):
                vp = -vp
            match = m(vp)
        out_probes.append({
            "label": f"{p['class']} #{i + 1} — voltage across",
            "pred_nets": p["pred_nets"], "gt_nets": p["gt_nets"],
            "t": (grid * 1000).round(4).tolist(),
            "pred": None if vp is None else vp.round(5).tolist(),
            "gt": None if vg is None else vg.round(5).tolist(),
            "match": None if match is None else round(match * 100, 1),
        })

    return {
        "probes": out_probes,
        "pred_sim": {"ok": bool(ok_p), "category": cat_p},
        "gt_sim": gt_sim,
        "has_gt": gt_bench is not None,
        "t_end_ms": round(t_max * 1000, 4),
        "partial": bool(partial),
    }


def process(image_bytes: bytes, run: Path, gt_stem: str | None = None) -> dict:
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

    # ---- 9. transient simulation + waveform comparison ----
    sim = _simulation_stage(comps, dets, run, rep, gt_stem)
    cap = ("Both netlists get identical placeholder values, so any "
           "difference between the traces is topology-recovery error — "
           "matching waveforms mean the drawing's behaviour was captured."
           if sim["has_gt"] else
           "Transient response of the recovered netlist (no verified "
           "reference exists for an uploaded image).")
    stage("simulate", "ngspice waveforms — recovered vs verified",
          cap, "07_snap.png", extra=sim)

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
        gt_stem = None
        if "file" in request.files:
            data = request.files["file"].read()
        else:
            name = Path(request.json.get("example", "")).name
            p = REPO / "data" / "raw" / name
            if not p.exists():
                return jsonify({"error": "unknown example"}), 400
            data = p.read_bytes()
            gt_stem = Path(name).stem
        result = process(data, run, gt_stem=gt_stem)
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
