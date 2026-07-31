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
from schematic2netlist.classes import canonical_class
from schematic2netlist.nodes import (
    build_wire_nodes_learned,
    bbox_xyxy,
    build_wire_nodes,
    build_wire_nodes_crossover_aware,
)
from schematic2netlist.repair import build_ledger, export_ledger, repair_circuit
from schematic2netlist.connectivity_repair import repair_connectivity
from schematic2netlist.snapping import build_component_pin_nets
from schematic2netlist.textmask import detect_text_mask
from schematic2netlist.wires import (
    build_non_wire_mask,
    extract_wires,
    stitch_wire_islands,
    stitchable_mask,
)


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
        # terminal count is class-dependent (1 for GND, 3 for transistors)
        n = len(c["nodes"])
        for t, node in enumerate(c["nodes"]):
            col = (0, 255, 255) if node is not None else (0, 0, 255)
            off = int((t - (n - 1) / 2) * 12)
            cv2.circle(debug, (cx_c + off, cy_c), 5, col, -1)
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

    # "Text" detections (from the 18-class detector) are mask evidence,
    # never components: they are removed here and rasterized into the
    # text mask below, so they can neither reach component building nor
    # inflate benchmark alignment. Motivation is measured, not assumed —
    # the heuristic mask fully misses 10.5% of text boxes (48% of test
    # images affected), and unmasked text enters the wire mask as phony
    # wire. With a 17-class cache this partition is a no-op.
    text_dets = [d for d in detections
                 if canonical_class(d["class"]) == "Text"]
    detections = [d for d in detections
                  if canonical_class(d["class"]) != "Text"]

    save = out_dir is not None
    if save:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # --- text mask + non-wire mask ---
    text_mask = None
    if cfg["textmask"]["enabled"]:
        # Oracle/override hook: a directory of per-stem mask PNGs
        # replaces the heuristic. Used by the GT-text oracle (bounds how
        # much perfect text masking is worth) and later by
        # detector-based masking. Default off; missing file falls back
        # to the heuristic so partial mask sets cannot silently zero
        # out masking.
        mask_dir = cfg["textmask"].get("mask_dir")
        if mask_dir:
            p = Path(mask_dir) / (Path(str(image_path)).stem + ".png")
            if p.exists():
                text_mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if text_mask is None:
            text_mask = detect_text_mask(gray, cfg)
        # union in detector-found text boxes (padded like components).
        # Detection and the heuristic are complementary: the detector
        # catches labels the CC heuristic loses to wire contact; the
        # heuristic still covers strokes outside any detected box.
        if text_dets:
            pad = int(cfg["textmask"].get("det_pad", 2))
            Hh, Ww = text_mask.shape
            for d in text_dets:
                x1 = max(0, int(d["x"] - d["width"] / 2) - pad)
                y1 = max(0, int(d["y"] - d["height"] / 2) - pad)
                x2 = min(Ww, int(d["x"] + d["width"] / 2) + pad)
                y2 = min(Hh, int(d["y"] + d["height"] / 2) + pad)
                text_mask[y1:y2, x1:x2] = 255
        if save:
            cv2.imwrite(str(out_dir / "01b_text_mask.png"), text_mask)

    non_wire_mask = build_non_wire_mask(gray, detections, cfg, text_mask)
    if save:
        cv2.imwrite(str(out_dir / "01_non_wire_mask.png"), non_wire_mask)

    # --- wire extraction ---
    wire_candidate, clean_wires = extract_wires(gray, non_wire_mask, cfg)

    # tier-1 fix: reconnect islands split by our own masking (text boxes,
    # component pad rings) — the measured dominant cause of net shattering
    if cfg["wires"].get("stitch_masked_gaps"):
        stitchable = stitchable_mask(gray.shape, detections, cfg, text_mask)
        clean_wires = stitch_wire_islands(clean_wires, stitchable, cfg)

    if save:
        cv2.imwrite(str(out_dir / "02_wire_candidates.png"), wire_candidate)
        cv2.imwrite(str(out_dir / "03_wire_binary.png"), clean_wires)
        overlay = img.copy()
        overlay[clean_wires > 0] = (0, 255, 0)
        blended = cv2.addWeighted(img, 0.65, overlay, 0.35, 0)
        cv2.imwrite(str(out_dir / "04_wire_overlay.png"), blended)

    # --- nodes + snapping ---
    ncfg = cfg["nodes"]
    # `method` supersedes the older boolean; the boolean is still read so
    # existing configs and committed run_meta records keep working.
    method = ncfg.get("method")
    if method is None:
        method = "crossover" if ncfg.get("handle_crossovers") else "cc"
    crossover_boxes = [
        d for d in detections
        if canonical_class(d["class"]) == "Wire Crossover"
    ]
    def _build_nodes(wires):
        """Nodes + component pin nets from a wire mask.

        Closed over ``detections``/``cfg`` and used both for the first pass and
        for every rebuild inside connectivity repair, so there is exactly ONE
        node-construction path. A diagnostic that re-implemented this dispatch
        instead silently diverged from the pipeline and produced a wrong
        conclusion, which is why the repair stage is handed this function rather
        than the ingredients to rebuild it.
        """
        info = None
        if method == "learned":
            nm, nn, info = build_wire_nodes_learned(
                wires, crossover_boxes,
                weights=ncfg["junction_weights"],
                threshold=ncfg.get("junction_threshold", 0.4),
                site_box=ncfg.get("junction_site_box", 15),
                context=ncfg.get("junction_context", 3.0),
                thin_input=ncfg.get("junction_thin_input", False),
                relink=ncfg.get("relink", "band"),
                connectivity=ncfg["connectivity"],
            )
        elif method == "vector":
            from .vector_nodes import build_wire_nodes_vector
            vcfg = ncfg.get("vector", {}) or {}
            nm, nn, info = build_wire_nodes_vector(
                wires, crossover_boxes,
                connectivity=ncfg["connectivity"],
                **{k: v for k, v in vcfg.items()},
            )
        elif method == "crossover":
            nm, nn = build_wire_nodes_crossover_aware(
                wires, crossover_boxes,
                connectivity=ncfg["connectivity"],
                relink=ncfg.get("relink", "band"),
            )
        elif method == "cc":
            nm, nn = build_wire_nodes(
                wires, connectivity=ncfg["connectivity"]
            )
        else:
            raise ValueError(f"Unknown nodes.method: {method!r}")
        cs = build_component_pin_nets(detections, nm, cfg)
        # the repair stage inspects node_names, so they must exist on rebuild
        for c in cs:
            c["node_names"] = [None if x is None else f"n{x}"
                               for x in c["nodes"]]
        return nm, nn, info, cs

    node_map, num_nodes, junction_info, comps = _build_nodes(clean_wires)

    # --- constraint-triggered connectivity repair.
    # Unlike C5 below, this DOES change topology -- it acts only where an
    # electrical fact makes the current answer impossible (a component with
    # every pin on one net, a net with a single terminal). See
    # connectivity_repair.py. ---
    conn_repair = None
    if cfg.get("connectivity_repair", {}).get("enabled"):
        (clean_wires, node_map, rebuilt_n, rebuilt_info, comps,
         conn_repair) = repair_connectivity(
            clean_wires, node_map, comps, detections, cfg, _build_nodes)
        if conn_repair["applied"]:
            num_nodes = rebuilt_n if rebuilt_n is not None else num_nodes
            junction_info = (rebuilt_info if rebuilt_info is not None
                             else junction_info)

    # --- node naming + netlist export ---
    node_name_map = build_node_name_map(
        comps, ground_fallback=cfg["netlist"]["ground_fallback"]
    )
    assign_node_names(comps, node_name_map)

    # --- design-intent repair (M4 / C5): diagnose + minimal logged fixes.
    # Topology is untouched; repairs are extra SPICE lines only. ---
    repair_result = None
    if cfg.get("repair", {}).get("enabled"):
        repair_result = repair_circuit(comps, node_name_map, cfg)

    netlist_info = None
    if save:
        export_readable_netlist(comps, str(out_dir / "netlist_readable.txt"))
        netlist_info = export_spice_netlist(
            comps,
            str(out_dir / "netlist.sp"),
            placeholders=cfg["netlist"]["placeholders"],
        )
        if repair_result is not None:
            export_spice_netlist(
                comps,
                str(out_dir / "netlist_repaired.sp"),
                placeholders=cfg["netlist"]["placeholders"],
                extra_lines=repair_result.extra_lines,
            )
            ledger = build_ledger(
                Path(image_path).name, None, None, repair_result
            )
            export_ledger(ledger, str(out_dir / "ledger.json"))
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
        # exposed for diagnostics (weld localisation, oracle checks); it is
        # the label image net assembly produced, not a copy
        "node_map": node_map,
        "clean_wires": clean_wires,
        "node_name_map": node_name_map,
        "coverage": coverage,
        "netlist": netlist_info,
        "repair": repair_result,
        "nodes_method": method,
        "connectivity_repair": conn_repair,
        "junction_info": junction_info,
        "out_dir": str(out_dir) if save else None,
    }
