"""End-to-end per-image orchestration:

image -> detections -> text mask -> non-wire mask -> wire extraction ->
node inference -> terminal snapping -> node naming -> SPICE netlist.

Debug artifact names match the legacy scripts so runs remain visually
comparable across the restructure.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from schematic2netlist import detect as detect_mod
from schematic2netlist import metrics as metrics_mod
from schematic2netlist.netlist import (
    assign_node_names,
    build_node_name_map,
    export_readable_netlist,
    export_spice_netlist,
)
from schematic2netlist.nodes import bbox_xyxy, build_wire_nodes
from schematic2netlist.snapping import build_component_pin_nets
from schematic2netlist.textmask import detect_text_mask
from schematic2netlist.wires import build_non_wire_mask, extract_wires


def _write_debug_overlay(img, clean_wires, detections, comps, out_path):
    debug = img.copy()
    debug[clean_wires > 0] = (0, 255, 0)
    for det in detections:
        x1, y1, x2, y2 = bbox_xyxy(det)
        cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 0, 0), 2)
    for c in comps:
        x1, y1, x2, y2 = bbox_xyxy(detections[c["id"]])
        cx_c = int((x1 + x2) / 2)
        cy_c = int((y1 + y2) / 2)
        col_a = (0, 255, 255) if c["nodes"][0] is not None else (0, 0, 255)
        col_b = (0, 255, 255) if c["nodes"][1] is not None else (0, 0, 255)
        cv2.circle(debug, (cx_c - 10, cy_c), 5, col_a, -1)
        cv2.circle(debug, (cx_c + 10, cy_c), 5, col_b, -1)
    cv2.imwrite(str(out_path), debug)


def run_pipeline(
    image_path: str | Path,
    cfg: dict,
    detections: list[dict] | None = None,
    out_dir: str | Path | None = None,
) -> dict:
    """Run the full pipeline on one image.

    If ``detections`` is None they are resolved via the configured
    detection backend (per-image cache first). When ``out_dir`` is given,
    debug artifacts and netlists are written there.
    """
    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if detections is None:
        detections = detect_mod.detect(image_path, cfg)

    save = out_dir is not None
    if save:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # --- text mask + non-wire mask ---
    text_mask = None
    if cfg["textmask"]["enabled"]:
        text_mask = detect_text_mask(gray, cfg)
        if save:
            cv2.imwrite(str(out_dir / "01b_text_mask.png"), text_mask)

    non_wire_mask = build_non_wire_mask(gray, detections, cfg, text_mask)
    if save:
        cv2.imwrite(str(out_dir / "01_non_wire_mask.png"), non_wire_mask)

    # --- wire extraction ---
    wire_candidate, clean_wires = extract_wires(gray, non_wire_mask, cfg)
    if save:
        cv2.imwrite(str(out_dir / "02_wire_candidates.png"), wire_candidate)
        cv2.imwrite(str(out_dir / "03_wire_binary.png"), clean_wires)
        overlay = img.copy()
        overlay[clean_wires > 0] = (0, 255, 0)
        blended = cv2.addWeighted(img, 0.65, overlay, 0.35, 0)
        cv2.imwrite(str(out_dir / "04_wire_overlay.png"), blended)

    # --- nodes + snapping ---
    node_map, num_nodes = build_wire_nodes(
        clean_wires, connectivity=cfg["nodes"]["connectivity"]
    )
    comps = build_component_pin_nets(detections, node_map, cfg)

    # --- node naming + netlist export ---
    node_name_map = build_node_name_map(
        comps, ground_fallback=cfg["netlist"]["ground_fallback"]
    )
    assign_node_names(comps, node_name_map)

    netlist_info = None
    if save:
        export_readable_netlist(comps, str(out_dir / "netlist_readable.txt"))
        netlist_info = export_spice_netlist(
            comps,
            str(out_dir / "netlist.sp"),
            placeholders=cfg["netlist"]["placeholders"],
        )
        _write_debug_overlay(
            img, clean_wires, detections, comps,
            out_dir / "06_netlist_debug_overlay.png",
        )

    coverage = metrics_mod.coverage_stats(comps)

    return {
        "image": str(image_path),
        "detections": detections,
        "components": comps,
        "num_wire_nodes": num_nodes,
        "node_name_map": node_name_map,
        "coverage": coverage,
        "netlist": netlist_info,
        "out_dir": str(out_dir) if save else None,
    }
