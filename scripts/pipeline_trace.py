#!/usr/bin/env python3
"""Stage-by-stage trace of the pipeline on ONE image, as a troubleshooting report.

Answers, for every stage the pipeline performs: what went in, what came out, WHICH
FILE AND LINE did it, which config keys steered it, and what it means when the
picture looks wrong.

Design decisions that matter for trusting the output:

  the real functions      Every stage calls the same function the pipeline calls.
                          Nothing is re-implemented. demo/app.py re-implements the
                          node dispatch and had already drifted -- it never passes
                          ``relink``, so it can render connectivity the pipeline
                          would not produce. A troubleshooting tool that can show
                          something the pipeline does not do is worse than none.

  paths via inspect       Source locations come from ``inspect.getsourcelines``,
                          not from strings, so a path or line number cannot go
                          stale as code moves. If a symbol is renamed this file
                          fails loudly instead of pointing somewhere wrong.

  deltas, not just states A stage that adds or removes ink is shown as a DIFF --
                          bridged pixels in magenta, deleted blobs in red, split
                          regions outlined -- because "what changed" is the
                          question being asked when something is wrong, and two
                          near-identical masks side by side do not answer it.

Output is one self-contained HTML file (images embedded) plus the loose PNGs.

Usage:
    python scripts/pipeline_trace.py --image data/cleaned_1024/circuit_1.jpg
    python scripts/pipeline_trace.py --stem circuit_113 --out-dir /tmp/trace
    python scripts/pipeline_trace.py --stem circuit_1 --raw   # from the photo
"""

from __future__ import annotations

import argparse
import base64
import html
import inspect
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from schematic2netlist import preprocess as pp
from schematic2netlist import wires as wr
from schematic2netlist import skeleton as sk
from schematic2netlist import ports as pt
from schematic2netlist import snapping as sn
from schematic2netlist import nodes as nd
from schematic2netlist import netlist as nl
from schematic2netlist import connectivity_repair as cr
from schematic2netlist import repair as rp
from schematic2netlist import simulate as sim
from schematic2netlist import metrics as mt
from schematic2netlist.classes import canonical_class
from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.determinism import set_global_seed
from schematic2netlist.textmask import detect_text_mask

GREEN, MAGENTA, RED, BLUE, ORANGE = ((80, 200, 80), (230, 60, 200),
                                     (60, 60, 235), (235, 160, 60),
                                     (40, 170, 245))


def where(fn) -> str:
    """'src/schematic2netlist/wires.py:208  _bridge_collinear()' from the live
    object, so a path or line number cannot drift as code moves.

    C extensions (cv2, numpy) have no Python source; they are named by module so
    the reader still knows what ran, rather than being silently dropped.
    """
    try:
        f = Path(inspect.getsourcefile(fn)).resolve()
        _, line = inspect.getsourcelines(fn)
    except (TypeError, OSError):
        mod = getattr(fn, "__module__", None) or "?"
        return f"[native] {mod}.{getattr(fn, '__name__', fn)}()"
    try:
        rel = f.relative_to(ROOT)
    except ValueError:
        rel = f
    return f"{rel}:{line}  {fn.__qualname__}()"


class Trace:
    def __init__(self, out: Path, cfg: dict):
        self.out = out
        self.cfg = cfg
        self.stages: list[dict] = []
        out.mkdir(parents=True, exist_ok=True)

    def cfgval(self, dotted: str):
        node = self.cfg
        for k in dotted.split("."):
            if not isinstance(node, dict) or k not in node:
                return "<absent>"
            node = node[k]
        return node

    def add(self, sid, title, what, img=None, sources=(), keys=(), stats=None,
            note="", text=None):
        name = None
        if img is not None:
            name = f"{len(self.stages):02d}_{sid}.png"
            cv2.imwrite(str(self.out / name), img)
        self.stages.append({
            "sid": sid, "title": title, "what": what, "image": name,
            "sources": [where(s) if callable(s) else str(s) for s in sources],
            "keys": [(k, self.cfgval(k)) for k in keys],
            "stats": stats or {}, "note": note, "text": text,
        })
        print(f"  [{len(self.stages):02d}] {title}")


def bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray


def tint(base, mask, colour, alpha=1.0):
    out = base.copy()
    if alpha >= 1.0:
        out[mask > 0] = colour
        return out
    lay = base.copy()
    lay[mask > 0] = colour
    return cv2.addWeighted(base, 1 - alpha, lay, alpha, 0)


def diff_view(base, before, after):
    """added in magenta, removed in red, kept in green."""
    v = base.copy()
    kept = cv2.bitwise_and(before, after)
    v = tint(v, kept, GREEN)
    v = tint(v, cv2.subtract(after, before), MAGENTA)
    v = tint(v, cv2.subtract(before, after), RED)
    return v


def npx(m):
    return int((m > 0).sum())


def palette(n, seed=0):
    rng = np.random.default_rng(seed)
    return [tuple(int(c) for c in rng.integers(40, 235, 3)) for _ in range(max(n, 1))]


class _Args:
    def __init__(self, raw=False):
        self.raw = raw


def build_trace(src_img, stem: str, cfg: dict, out: Path,
                raw: bool = False) -> Trace:
    """Trace every stage and return the populated Trace.

    Exposed so demo/app.py can render the SAME stages instead of
    re-implementing the pipeline -- which it used to do, and had already
    drifted (it never passed nodes.relink).
    """
    args = _Args(raw)
    return _trace(src_img, stem, cfg, out, args)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image")
    ap.add_argument("--stem", help="a test-split stem, e.g. circuit_113")
    ap.add_argument("--raw", action="store_true",
                    help="start from the ORIGINAL photo so preprocessing stages "
                         "are real work rather than a no-op on a cleaned frame")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default="results/trace")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["seed"])
    RAWDIR = ROOT / ("data/digitize_hcd/extracted/Digitize-HCD Dataset/"
                     "Component Symbol and Text Label Data/Circuit Diagram Images")

    if args.stem:
        stem = args.stem
        src_img = (RAWDIR / f"{stem}.jpg") if args.raw else \
            Path(cfg["preprocess"]["images_dir"]) / f"{stem}.jpg"
    elif args.image:
        src_img = Path(args.image)
        stem = src_img.stem
    else:
        raise SystemExit("give --image or --stem")
    if not Path(src_img).exists():
        raise SystemExit(f"no such image: {src_img}")

    out = Path(args.out_dir) / stem
    T = _trace(src_img, stem, cfg, out, args)
    write_html(T, stem, src_img)


def _trace(src_img, stem: str, cfg: dict, out: Path, args) -> Trace:
    T = Trace(out, cfg)
    print(f"tracing {stem} -> {out}")

    # ================= 00 input =================
    orig = cv2.imread(str(src_img), cv2.IMREAD_COLOR)
    T.add("input", "Input image", f"Read from <code>{src_img}</code>.", orig,
          sources=[cv2.imread], stats={"shape": f"{orig.shape[1]}x{orig.shape[0]}"},
          note="If this is already a cleaned 1024 px frame, the preprocessing "
               "stages below are near no-ops. Pass --raw to trace a real photo.")

    # ================= preprocessing =================
    g0 = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    ink0 = cv2.threshold(cv2.GaussianBlur(
        g0, (cfg["preprocess"]["blur_kernel"],) * 2, 0), 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    angle = pp._estimate_skew(ink0, cfg["preprocess"])
    T.add("skew", "1. Skew estimation",
          f"Length-weighted median of Hough segments folded mod 90 gives "
          f"<b>{angle:+.3f}&deg;</b>. Rejected outright if |angle| exceeds "
          f"<code>max_skew_deg</code>.",
          tint(bgr(g0), ink0, BLUE, 0.5), sources=[pp._estimate_skew],
          keys=["preprocess.hough_min_line_frac", "preprocess.hough_threshold",
                "preprocess.hough_max_gap", "preprocess.max_skew_deg",
                "preprocess.blur_kernel"],
          stats={"angle_deg": round(angle, 4), "ink_px": npx(ink0)},
          note="A wrong angle here rotates every downstream coordinate. If the "
               "estimate is 0.000 on a visibly skewed page, Hough found too few "
               "long segments -- lower hough_min_line_frac.")

    speck_in = ink0.copy()
    speck_out = pp._remove_specks(255 - speck_in, cfg["preprocess"])
    removed = cv2.subtract(speck_in, 255 - speck_out)
    T.add("specks", "2. Speck removal",
          "A blob survives on area <b>OR</b> extent, so hairline strokes are "
          "kept where a pure area rule would delete them. Red = removed.",
          diff_view(bgr(g0), speck_in, 255 - speck_out),
          sources=[pp._remove_specks],
          keys=["preprocess.remove_specks", "preprocess.speck_min_area",
                "preprocess.speck_min_extent"],
          stats={"removed_px": npx(removed)},
          note="Red on an actual wire means speck_min_area/extent are too "
               "aggressive; that ink is gone for the rest of the pipeline.")

    # CRITICAL. Preprocessing must run exactly once. Feeding an
    # already-cleaned frame back through preprocess_image_meta re-crops and
    # re-scales it, so the canvas no longer matches the coordinate frame the
    # cached detections and GT boxes live in -- and every stage after it traces a
    # pipeline nobody runs. That mistake scored circuit_1077 at 0.52
    # terminal-pair F1 when the pipeline gets 1.00 on it.
    if args.raw:
        canvas, meta = pp.preprocess_image_meta(str(src_img), cfg)
        cached = Path(cfg["preprocess"]["images_dir"]) / f"{stem}.jpg"
        agree = "not checked"
        if cached.exists():
            ref = cv2.imread(str(cached), cv2.IMREAD_GRAYSCALE)
            agree = ("identical to the committed frame" if ref is not None
                     and ref.shape == canvas.shape
                     and int(np.abs(ref.astype(int)
                                    - canvas.astype(int)).max()) <= 1
                     else "DIFFERS from the committed frame")
        what = (f"Rotated {meta['angle_deg']:+.2f}&deg;, cropped to "
                f"<code>{meta['crop']}</code>, scaled "
                f"&times;{meta['scale']:.4f} onto a {meta['target_size']} px "
                f"canvas. Result is {agree}.")
        extra_note = ""
    else:
        canvas = cv2.imread(str(src_img), cv2.IMREAD_GRAYSCALE)
        meta = {"angle_deg": 0.0, "scale": 1.0, "rotated90": False,
                "target_size": cfg["preprocess"]["target_size"],
                "crop": "n/a (already applied)"}
        what = ("This input is <b>already a preprocessed frame</b>, so the "
                "pipeline consumes it directly and this stage is a pass-through. "
                "The two stages above were recomputed on it for illustration.")
        extra_note = (" Re-running preprocessing on an already-cleaned frame "
                      "would re-crop it and break alignment with the cached "
                      "detections and GT boxes; use --raw to trace it for real.")
    T.add("preprocess", "3. Deskew, shadow-normalise, crop, scale", what,
          bgr(canvas),
          sources=[pp.preprocess_image_meta, pp.project_point, pp.project_bbox],
          keys=["preprocess.target_size", "preprocess.crop_pad",
                "preprocess.crop_pad_frac", "preprocess.shadow_std_threshold",
                "preprocess.shadow_dilate_kernel", "preprocess.landscape_ratio"],
          stats={k: meta[k] for k in ("angle_deg", "scale", "rotated90",
                                      "target_size") if k in meta},
          note="meta{} here is what project_point/unproject_point use to move GT "
               "boxes and text annotations between the photo and this canvas. If "
               "GT boxes look offset downstream, suspect this transform." + extra_note)

    gray = canvas
    img = bgr(canvas)

    # ================= detection =================
    dpath = Path(cfg["detect"]["cache_dir"]) / f"{stem}.json"
    if dpath.exists():
        dets_all = load_cached_detections(
            str(dpath), min_confidence=cfg["detect"].get("confidence"))
        det_src = f"cache <code>{dpath}</code>"
    else:
        from schematic2netlist.detect import detect_ultralytics
        cv2.imwrite(str(out / "_frame.png"), canvas)
        dets_all = detect_ultralytics([out / "_frame.png"], cfg)[0]
        det_src = "live YOLO inference"

    vis = img.copy()
    from collections import Counter
    per_class = Counter(canonical_class(d["class"]) for d in dets_all)
    for d in dets_all:
        x1, y1, x2, y2 = nd.bbox_xyxy(d)
        cv2.rectangle(vis, (x1, y1), (x2, y2), ORANGE, 2)
        cv2.putText(vis, f"{canonical_class(d['class'])} {d['confidence']:.2f}",
                    (x1, max(11, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, ORANGE, 1)
    T.add("detect", "4. Component detection",
          f"{len(dets_all)} boxes from {det_src}, filtered at "
          f"<code>confidence &ge; {cfg['detect'].get('confidence')}</code>.",
          vis, sources=[load_cached_detections, nd.bbox_xyxy],
          keys=["detect.backend", "detect.weights", "detect.confidence",
                "detect.image_size", "detect.cache_dir"],
          stats=dict(sorted(per_class.items())),
          note="A WRONG CLASS here is the single largest remaining detection "
               "error source: the GT-class oracle is +0.0211 strict success. A "
               "spurious box is worse than a miss -- it blanks its own ink in "
               "the next stage and can sever a net.")

    text_dets = [d for d in dets_all if canonical_class(d["class"]) == "Text"]
    dets = [d for d in dets_all if canonical_class(d["class"]) != "Text"]
    T.add("partition", "5. Text/component partition",
          f"{len(text_dets)} 'Text' detections are mask evidence, never "
          f"components; {len(dets)} remain as components. With a 17-class cache "
          f"this is a no-op.",
          None, sources=["src/schematic2netlist/pipeline.py  run_pipeline() "
                         "text_dets partition"],
          stats={"text_dets": len(text_dets), "component_dets": len(dets)},
          note="If Text boxes reached component building they would inflate "
               "benchmark alignment and be scored as circuit elements.")

    # ================= masks =================
    tmask = detect_text_mask(gray, cfg) if cfg["textmask"]["enabled"] else None
    if tmask is not None:
        T.add("textmask", "6. Handwriting mask",
              "Adaptive-threshold connected components filtered by area, aspect "
              "and size — labels like &quot;10k&quot; are not conductors.",
              tint(img, tmask, BLUE, 0.55), sources=[detect_text_mask],
              keys=["textmask.enabled", "textmask.adaptive_block_size",
                    "textmask.adaptive_c", "textmask.dilate_kernel",
                    "textmask.min_area", "textmask.max_area",
                    "textmask.min_aspect", "textmask.max_aspect"],
              stats={"masked_px": npx(tmask),
                     "frac_of_frame": round(npx(tmask) / tmask.size, 5)},
              note="dilate_kernel is 6, an EVEN size, so OpenCV anchors it "
                   "off-centre and the mask is offset by half a pixel on both "
                   "axes. Blue covering a wire means that wire is about to be "
                   "deleted.")

    non_wire = wr.build_non_wire_mask(gray, dets, cfg, tmask)
    T.add("nonwire", "7. Non-wire mask",
          "Component boxes (padded by <code>component_mask_pad</code>) unioned "
          "with the handwriting mask. Everything red is removed from the ink.",
          tint(img, non_wire, RED, 0.5), sources=[wr.build_non_wire_mask],
          keys=["wires.component_mask_pad", "wires.non_wire_classes"],
          stats={"masked_px": npx(non_wire)},
          note="component_mask_pad is 0 deliberately: at 8 it erased the wire "
               "pixels immediately next to every component -- exactly the "
               "terminals snapping then has to find -- and cost 0.163 strict "
               "success on its own. Wire Crossover is absent from "
               "non_wire_classes because masking it would sever the crossing.")

    # ================= wire extraction, in its three real steps =================
    cand = gray.copy()
    cand[non_wire > 0] = 255
    ink = cv2.threshold(cand, 0, 255,
                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ink[non_wire > 0] = 0
    T.add("ink", "8. Ink binarisation",
          "Otsu on the masked frame. C2: the ink itself, not Canny edges — an "
          "edge detector turns one pen stroke into TWO parallel lines with a "
          "hollow gap that morphology then has to glue back together.",
          tint(img, ink, GREEN), sources=[wr.extract_wires_ink, wr.extract_wires],
          keys=["wires.method"], stats={"ink_px": npx(ink)},
          note="Speckle here means the frame has low contrast; Otsu is global.")

    _cand_ref, _clean_ref = wr.extract_wires(gray, non_wire, cfg)
    bridged = wr._bridge_collinear(ink, cfg)
    n_before = cv2.connectedComponents(ink)[0] - 1
    n_after = cv2.connectedComponents(bridged)[0] - 1
    T.add("bridge", "9. Gap bridging",
          f"Line-kernel closings join stroke fragments across pen lifts. "
          f"Components {n_before} &rarr; {n_after}. "
          f"<b>Magenta = pixels this stage invented.</b>",
          diff_view(img, ink, bridged),
          sources=[wr._bridge_collinear, wr._line_kernel, wr._oriented_ink,
                   wr._bridge_guarded],
          keys=["wires.bridge_span", "wires.bridge_mode",
                "wires.bridge_odd_kernel", "wires.bridge_run",
                "wires.bridge_thick"],
          stats={"added_px": npx(cv2.subtract(bridged, ink)),
                 "deleted_px": npx(cv2.subtract(ink, bridged)),
                 "cc_before": n_before, "cc_after": n_after},
          note="THE STAGE TO SUSPECT FIRST FOR WELDED NETS. A horizontal closing "
               "bridges horizontal gaps between ANY ink, so two side-by-side "
               "VERTICAL rails closer than bridge_span fuse into one conductor. "
               "That is why the span is 7 and not the 18 it shipped with. If "
               "deleted_px is non-zero the kernel is even-length and the closing "
               "is destroying ink (bridge_odd_kernel).")

    clean0 = wr._filter_blobs(bridged, cfg)
    T.add("filter", "10. Noise-blob filter",
          "A blob survives on area <b>OR</b> extent — a hairline wire has small "
          "area but large extent, and an area-only rule shatters nets. "
          "<b>Red = dropped.</b>",
          diff_view(img, bridged, clean0), sources=[wr._filter_blobs],
          keys=["wires.min_blob_area", "wires.min_blob_extent"],
          stats={"dropped_px": npx(cv2.subtract(bridged, clean0)),
                 "kept_px": npx(clean0)},
          note="Measured harmless: across 40 frames this drops 20 blobs / 623 px "
               "and NONE of them bridged two surviving components.")

    # the manual three-step reconstruction above must equal what the pipeline's
    # own extract_wires returns, or this trace is describing a different program
    if int(np.abs(clean0.astype(int) - _clean_ref.astype(int)).max()) != 0:
        raise SystemExit(
            "trace divergence: the ink -> bridge -> filter reconstruction does "
            "not match wires.extract_wires(). Fix the trace, do not ship it.")

    wires_mask = clean0
    if cfg["wires"].get("stitch_masked_gaps"):
        stitchable = wr.stitchable_mask(gray.shape, dets, cfg, tmask)
        wires_mask = wr.stitch_wire_islands(clean0, stitchable, cfg)
        added = cv2.subtract(wires_mask, clean0)
        T.add("stitch", "11. Stitching across self-inflicted holes",
              "Reconnects islands separated by regions OUR OWN masking deleted — "
              "text boxes and component pad rings, never a component body. "
              "Collinearity is required on both endpoints.",
              diff_view(tint(img, stitchable, BLUE, 0.25), clean0, wires_mask),
              sources=[wr.stitch_wire_islands, wr.stitchable_mask,
                       wr._local_direction],
              keys=["wires.stitch_masked_gaps", "wires.stitch_max_gap",
                    "wires.stitch_angle_tol_deg", "wires.stitch_min_inside_frac",
                    "wires.stitch_dir_radius", "wires.stitch_passes"],
              stats={"added_px": npx(added),
                     "stitchable_px": npx(stitchable)},
              note="CURRENTLY INERT: with component_mask_pad=0 the pad ring is "
                   "empty, so only text boxes are stitchable. Turning this off "
                   "changes every topology metric by exactly 0.0000. It solved a "
                   "self-inflicted problem that was later fixed at the source.")

    # ================= skeleton / sites =================
    thin = sk.thin(wires_mask)
    sites = sk.intersection_sites_with_degree(
        thin, min_sep=cfg["nodes"].get("min_sep", 9)) \
        if "intersection_sites_with_degree" in dir(sk) else []
    vis = tint(img, wires_mask, GREEN, 0.35)
    vis = tint(vis, thin, (30, 30, 30))
    for s in sites:
        x, y = (s[0], s[1]) if isinstance(s, (tuple, list)) else (0, 0)
        deg = s[2] if isinstance(s, (tuple, list)) and len(s) > 2 else None
        cv2.circle(vis, (int(x), int(y)), 7, MAGENTA, 2)
        if deg is not None:
            cv2.putText(vis, str(deg), (int(x) + 8, int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, MAGENTA, 1)
    T.add("skeleton", "12. Skeleton and intersection sites",
          f"Thinned centreline (black) with {len(sites)} branch sites (magenta, "
          "labelled by degree). This is where a crossing decision would be made.",
          vis, sources=[sk.thin, sk.intersection_sites_with_degree, sk.crop_site],
          stats={"skeleton_px": npx(thin), "sites": len(sites)},
          note="A degree-4 site is a crossing OR a junction and the image cannot "
               "reliably tell which: six approaches on 4822 causally-labelled "
               "sites top out at 0.659 AUC, and giving the pipeline PERFECT "
               "crossover boxes makes strict success WORSE. Do not spend effort "
               "here.")

    # ================= nodes: plain CC vs crossover-aware =================
    xb = [d for d in dets if canonical_class(d["class"]) == "Wire Crossover"]
    ncfg = cfg["nodes"]
    cc_map, cc_n = nd.build_wire_nodes(
        wires_mask, connectivity=ncfg["connectivity"])
    pal = palette(cc_n + 2, 1)
    vis = np.full_like(img, 255)
    for i in range(cc_n):
        vis[cc_map == i] = pal[i]
    T.add("cc", "13. Connected components (the baseline)",
          f"{cc_n} regions by plain 8-connectivity. Every crossing is welded "
          "here — this is the arrangement crossover handling has to improve on.",
          cv2.addWeighted(img, 0.25, vis, 0.75, 0),
          sources=[nd.build_wire_nodes], keys=["nodes.connectivity"],
          stats={"nodes": cc_n},
          note="72.6% of wire nodes carrying component terminals fuse two or "
               "more GT nets at this point.")

    method = ncfg.get("method") or ("crossover" if ncfg.get("handle_crossovers")
                                    else "cc")
    if method == "crossover":
        node_map, n_nodes = nd.build_wire_nodes_crossover_aware(
            wires_mask, xb, connectivity=ncfg["connectivity"],
            relink=ncfg.get("relink", "band"))
        node_src = [nd.build_wire_nodes_crossover_aware, nd._edge_label]
    elif method == "cc":
        node_map, n_nodes = cc_map, cc_n
        node_src = [nd.build_wire_nodes]
    else:
        node_map, n_nodes = cc_map, cc_n
        node_src = [nd.build_wire_nodes]

    pal = palette(n_nodes + 2, 2)
    vis = np.full_like(img, 255)
    for i in range(n_nodes):
        vis[node_map == i] = pal[i]
    vis = cv2.addWeighted(img, 0.25, vis, 0.75, 0)
    for d in xb:
        x1, y1, x2, y2 = nd.bbox_xyxy(d)
        cv2.rectangle(vis, (x1, y1), (x2, y2), MAGENTA, 2)
    T.add("nodes", f"14. Net assembly ({method})",
          f"{n_nodes} electrical nets, one colour each. At each detected "
          f"<b>Wire Crossover</b> box (magenta, {len(xb)} here) the wires are "
          f"notched apart and opposite arms reconnected, so a crossing does not "
          f"weld two nets.",
          vis, sources=node_src,
          keys=["nodes.method", "nodes.handle_crossovers", "nodes.relink",
                "nodes.connectivity", "nodes.junction_site_box"],
          stats={"nodes": n_nodes, "vs_plain_cc": f"{cc_n} -> {n_nodes}",
                 "crossover_boxes": len(xb)},
          note="relink='angle' samples a ring and pairs the most-opposite arms "
               "by measured direction; 'band' votes in four fixed strips at the "
               "box edges and is placement-dependent -- a 2 px box shift once "
               "took terminal-pair F1 from 1.000 to 0.523.")

    # ================= snapping =================
    templates = pt.load_templates()
    vis = tint(img, wires_mask, GREEN, 0.3)
    n_sites = 0
    for d in dets:
        cls = canonical_class(d["class"])
        for pose in ("0", "90", "180", "270"):
            try:
                s = pt.predicted_sites(cls, d, pose, templates)
            except Exception:
                s = None
            if s:
                n_sites += len(s)
                break
        x1, y1, x2, y2 = nd.bbox_xyxy(d)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (170, 170, 170), 1)
    comps = sn.build_component_pin_nets(dets, node_map, cfg)
    for c in comps:
        x1, y1, x2, y2 = nd.bbox_xyxy(dets[c["id"]])
        cxm, cym = (x1 + x2) // 2, (y1 + y2) // 2
        n = len(c["nodes"])
        for t, node in enumerate(c["nodes"]):
            col = (0, 200, 255) if node is not None else RED
            cv2.circle(vis, (cxm + int((t - (n - 1) / 2) * 13), cym), 6, col, -1)
    n_unsnapped = sum(1 for c in comps for x in c["nodes"] if x is None)
    T.add("snap", "15. Terminal snapping (port identity)",
          f"Each component's pins are matched to nets. C3 assigns over DISTINCT "
          f"NODES by Hungarian, not over boundary runs — the bug that fix "
          f"removed was worth +0.133 per-component accuracy on its own. "
          f"Yellow = snapped, red = unsnapped ({n_unsnapped}).",
          vis, sources=[sn.build_component_pin_nets, sn.snap_ports,
                        pt.match_ports, pt.predicted_sites, pt.load_templates,
                        sn.find_ground_node, sn._boundary_run_sites],
          keys=["snapping.strategy", "snapping.max_expand",
                "snapping.window_depth", "snapping.expand_step",
                "snapping.ground_max_expand"],
          stats={"components": len(comps), "port_sites": n_sites,
                 "unsnapped_terminals": n_unsnapped},
          note="A red dot means no net was found within max_expand of that pin. "
               "Either the wire ink next to it was masked away (see stage 7) or "
               "the drawing genuinely leaves it floating.")

    # ================= connectivity repair =================
    for c in comps:
        c["node_names"] = [None if x is None else f"n{x}" for x in c["nodes"]]
    shorts, ones = cr.find_violations(comps, dets)
    rep_stats = {"self_shorted_components": len(shorts),
                 "one_terminal_nets": len(ones)}
    if cfg.get("connectivity_repair", {}).get("enabled"):
        def rebuild(w):
            nm, nn = nd.build_wire_nodes_crossover_aware(
                w, xb, connectivity=ncfg["connectivity"],
                relink=ncfg.get("relink", "band")) if method == "crossover" \
                else nd.build_wire_nodes(w, connectivity=ncfg["connectivity"])
            cs = sn.build_component_pin_nets(dets, nm, cfg)
            for c in cs:
                c["node_names"] = [None if x is None else f"n{x}"
                                   for x in c["nodes"]]
            return nm, nn, None, cs
        w2, node_map2, nn2, _i, comps2, report = cr.repair_connectivity(
            wires_mask, node_map, comps, dets, cfg, rebuild)
        rep_stats |= report["actions"]
        vis = diff_view(img, wires_mask, w2)
        for cid in shorts:
            x1, y1, x2, y2 = nd.bbox_xyxy(dets[cid])
            cv2.rectangle(vis, (x1, y1), (x2, y2), RED, 2)
        T.add("connrepair", "16. Constraint-triggered connectivity repair",
              "Acts only where an ELECTRICAL fact makes the answer impossible: a "
              "component with every pin on one net (0.60% in GT), or a net with a "
              "single terminal (0.00% — 0 of 1509). Magenta = ink added, red = "
              "erased body band.",
              vis, sources=[cr.repair_connectivity, cr.find_violations,
                            cr._erase_body, cr._bridge_fragment],
              keys=["connectivity_repair.enabled", "connectivity_repair.passes",
                    "connectivity_repair.body_frac",
                    "connectivity_repair.max_gap",
                    "connectivity_repair.dir_tol_deg"],
              stats=rep_stats,
              note="Worth +0.0211 strict success (4 gained, 0 lost) and it "
                   "COMPOUNDS with connectivity work: the identical repair on "
                   "the older wire settings gained only half as much, because "
                   "strict success is a product over components.")
        wires_mask, node_map, comps = w2, node_map2, comps2
        if nn2:
            n_nodes = nn2
    else:
        T.add("connrepair", "16. Connectivity repair (disabled)",
              "Violations detected but not acted on.", None,
              sources=[cr.find_violations], stats=rep_stats,
              keys=["connectivity_repair.enabled"])

    # ================= naming, netlist, repair, simulate =================
    name_map = nl.build_node_name_map(
        comps, ground_fallback=cfg["netlist"]["ground_fallback"])
    nl.assign_node_names(comps, name_map)
    T.add("naming", "17. Node naming and ground selection",
          "Nets are named, and one is chosen as ground (node 0) — SPICE cannot "
          "solve without a reference.", None,
          sources=[nl.build_node_name_map, nl.assign_node_names],
          keys=["netlist.ground_fallback"],
          stats={"named_nets": len(set(name_map.values())),
                 "ground_node": next((k for k, v in name_map.items()
                                      if str(v) == "0"), "<none>")},
          note="ground_fallback='most_connected' picks the busiest net when no "
               "GND symbol was detected. 'fail' refuses instead, which is the v2 "
               "behaviour and an ablation axis.")

    repair_result = None
    if cfg.get("repair", {}).get("enabled"):
        repair_result = rp.repair_circuit(comps, name_map, cfg)
    sp_path = out / "netlist.sp"
    nl.export_readable_netlist(comps, str(out / "netlist_readable.txt"))
    info = nl.export_spice_netlist(
        comps, str(sp_path), placeholders=cfg["netlist"]["placeholders"])
    if repair_result is not None:
        nl.export_spice_netlist(
            comps, str(out / "netlist_repaired.sp"),
            placeholders=cfg["netlist"]["placeholders"],
            extra_lines=repair_result.extra_lines)
    spice = sp_path.read_text()
    T.add("spice", "18. SPICE netlist",
          f"Written to <code>{sp_path}</code>; the repaired variant, with the "
          f"repair layer's extra lines appended verbatim under a banner, is "
          f"<code>{out / 'netlist_repaired.sp'}</code>.", None,
          sources=[nl.export_spice_netlist, nl.export_readable_netlist],
          keys=["netlist.placeholders", "netlist.ground_fallback"],
          stats={"lines": len(spice.splitlines()),
                 **{k: v for k, v in (info or {}).items()
                    if isinstance(v, (int, float, str, bool))}},
          text=spice,
          note="Values are PLACEHOLDERS. The dataset ships real transcribed "
               "values, but they cannot be attached: value labels are not "
               "reliably associable with their own component (the capacitor's "
               "nearest label is often the inductor's), which was measured at "
               "29.4% precision and a NET LOSS of 14 labels.")

    if repair_result is not None:
        led = rp.build_ledger(f"{stem}.jpg", None, None, repair_result)
        T.add("repair", "19. Design-intent repair (C5)",
              "Electrical rule checks diagnose why the circuit cannot simulate, "
              "then a minimal, fully-logged set of repairs is applied. Topology "
              "is NEVER changed here — that is stage 16's job.", None,
              sources=[rp.repair_circuit, rp.build_ledger, rp.export_ledger],
              keys=["repair.enabled", "repair.max_assumptions",
                    "repair.strategies", "repair.shunt_r"],
              stats={"entries": len(led.get("entries", [])),
                     "solvable_before": led.get("solvable_before"),
                     "solvable_after": led.get("solvable_after"),
                     "assumptions": led.get("num_assumptions"),
                     "gauge": led.get("num_gauge")},
              text=json.dumps(led, indent=1)[:4000],
              note="Every entry is classified as a behaviour-invariant GAUGE "
                   "choice or a flagged ASSUMPTION, so a reader can see exactly "
                   "what was assumed on their behalf.")

    target = (out / "netlist_repaired.sp") if (out / "netlist_repaired.sp").exists() \
        else sp_path
    ok, reason, diag = sim.run_ngspice_diag(str(target), cfg)
    T.add("simulate", "20. ngspice",
          f"Ran <code>{target.name}</code>. Result: "
          f"<b>{'solvable' if ok else reason}</b>.",
          None, sources=[sim.run_ngspice_diag, sim.run_ngspice,
                         sim.parse_ngspice_output],
          keys=["simulate.ngspice_binary", "simulate.timeout_s",
                "simulate.diagnostics"],
          stats={"solvable": ok, "reason": reason,
                 "failing_nodes": (diag or {}).get("nodes", [])[:8]},
          note="reason='timeout' would make this metric wall-clock dependent, "
               "the same defect that made nGED non-reproducible. If you see it, "
               "raise simulate.timeout_s and re-check.")

    # ================= metrics =================
    gp = Path(cfg["benchmark"]["gt_dir"]) / f"{stem}.json"
    if gp.exists():
        from schematic2netlist.benchmark import (align_components,
                                                 canonicalize_terminals)
        from schematic2netlist.gt import gt_to_components, load_gt
        gt = load_gt(str(gp))
        gc = gt_to_components(gt)
        by = {c["id"]: c for c in gt["components"]}
        for c in gc:
            c["bbox"] = by[c["id"]]["bbox"]
        pred = [{"id": c["id"], "class": c["class"],
                 "nets": list(c.get("node_names", [])),
                 "bbox": [dets[c["id"]]["x"], dets[c["id"]]["y"],
                          dets[c["id"]]["width"], dets[c["id"]]["height"]]}
                for c in comps]
        p, g, st = align_components(pred, gc)
        pc, gcn = canonicalize_terminals(p), canonicalize_terminals(g)
        tp = mt.terminal_pair_metrics(pc, gcn)
        nf = mt.net_level_metrics(pc, gcn)
        strict = int(st["unmatched_gt"] == 0 and tp["f1"] == 1.0
                     and nf["f1"] == 1.0)

        # ---- 22. WHERE the topology is wrong, drawn on the frame ----
        # Every other view shows what the pipeline produced. This one shows
        # what is WRONG with it, which is the question being asked when an
        # image scores badly, and it cannot be answered without ground truth
        # overlaid: circuit_415 has 55.6% of its nets welded and looks
        # perfectly well separated to the eye.
        from collections import defaultdict as _dd
        pred_of, gt_of = {}, {}
        for c in pc:
            for k_, n_ in enumerate(c["nets"]):
                pred_of[(c["id"], k_)] = n_
        for c in gcn:
            for k_, n_ in enumerate(c["nets"]):
                gt_of[(c["id"], k_)] = n_
        nets_on_node = _dd(set)      # predicted node -> distinct GT nets on it
        nodes_of_net = _dd(set)      # GT net -> distinct predicted nodes
        for term, pn in pred_of.items():
            gn = gt_of.get(term)
            if pn is not None and gn is not None:
                nets_on_node[pn].add(gn)
                nodes_of_net[gn].add(pn)

        by_id = {c["id"]: c for c in comps}
        id_of_pred = {}
        for c in pc:
            id_of_pred[c["id"]] = c
        vis = tint(img, wires_mask, (200, 200, 200), 0.55)
        n_weld = n_split = n_ok = 0
        for c in pc:
            src_c = by_id.get(c["id"])
            det = dets[src_c["id"]] if src_c is not None else None
            if det is None:
                continue
            x1, y1, x2, y2 = nd.bbox_xyxy(det)
            cxm, cym = (x1 + x2) // 2, (y1 + y2) // 2
            nt = len(c["nets"])
            for k_, pn in enumerate(c["nets"]):
                gn = gt_of.get((c["id"], k_))
                ox = cxm + int((k_ - (nt - 1) / 2) * 15)
                if pn is None or gn is None:
                    col = (110, 110, 110)
                elif len(nets_on_node[pn]) > 1:
                    col = RED; n_weld += 1           # welded to another net
                elif len(nodes_of_net[gn]) > 1:
                    col = (40, 170, 245); n_split += 1   # its net is split
                else:
                    col = (60, 190, 60); n_ok += 1
                cv2.circle(vis, (ox, cym), 7, col, -1)
                cv2.circle(vis, (ox, cym), 7, (30, 30, 30), 1)
        # outline the welded predicted nodes so the offending conductor is
        # visible, not just the terminals attached to it
        name_to_id = {}
        for c in comps:
            for n_, nn_ in zip(c.get("nodes", []), c.get("node_names", [])):
                if n_ is not None and nn_ is not None:
                    name_to_id[nn_] = int(n_)
        for pn, nets in nets_on_node.items():
            if len(nets) < 2:
                continue
            nid = name_to_id.get(pn)
            if nid is None:
                continue
            m = (node_map == nid).astype(np.uint8) * 255
            vis[m > 0] = RED
        T.add("errors", "22. WHERE the topology is wrong",
              f"<b>Red</b> = terminal on a node that carries {len(gcn) and ''}two or "
              f"more DIFFERENT ground-truth nets (welded, and the offending "
              f"conductor is painted red too). <b>Blue</b> = its net is split "
              f"across several nodes. <b>Green</b> = correct. "
              f"{n_weld} welded, {n_split} split, {n_ok} correct.",
              vis,
              sources=[mt._terminal_pairs, align_components,
                       canonicalize_terminals],
              stats={"terminals_welded": n_weld, "terminals_split": n_split,
                     "terminals_correct": n_ok,
                     "welded_nodes": sum(1 for v in nets_on_node.values()
                                         if len(v) > 1),
                     "split_gt_nets": sum(1 for v in nodes_of_net.values()
                                          if len(v) > 1)},
              note="Red conductors are where to look. Trace one back through "
                   "stage 9 (bridging) and stage 14 (net assembly): if the red "
                   "region is a long rail touching several components, the weld "
                   "is upstream in the ink; if it is a compact blob at a "
                   "crossing, the notch/relink decision is at fault.")
        T.add("metrics", "21. Scoring against verified ground truth",
              f"Strict end-to-end success requires EVERY component correct: "
              f"<b>{'PASS' if strict else 'FAIL'}</b>.", None,
              sources=[align_components, canonicalize_terminals,
                       mt.terminal_pair_metrics, mt.net_level_metrics,
                       mt.per_component_connected_accuracy,
                       mt.per_component_recall_accuracy, mt.normalized_ged],
              keys=["benchmark.gt_dir"],
              stats={"terminal_pair_f1": round(tp["f1"], 4),
                     "terminal_pair_precision": round(tp["precision"], 4),
                     "terminal_pair_recall": round(tp["recall"], 4),
                     "net_f1": round(nf["f1"], 4),
                     "per_component_exact": round(
                         mt.per_component_connected_accuracy(pc, gcn), 4),
                     "per_component_recall": round(
                         mt.per_component_recall_accuracy(pc, gcn), 4),
                     "nged": round(mt.normalized_ged(pc, gcn), 4),
                     "unmatched_gt": st["unmatched_gt"],
                     "unmatched_pred": st["unmatched_pred"],
                     "strict_success": strict},
              note="unmatched_gt > 0 makes strict success IMPOSSIBLE for this "
                   "image regardless of connectivity. per_component_exact "
                   "penalises welds; per_component_recall does not and is the "
                   "variant every pre-2026-07-30 result used.")

    return T


def write_html(T: Trace, stem: str, src_img) -> None:
    def b64(name):
        if not name:
            return None
        return base64.b64encode((T.out / name).read_bytes()).decode()

    css = """
:root{--bg:#0d1117;--fg:#e6edf3;--mut:#8b949e;--card:#161b22;--bd:#30363d;
--acc:#58a6ff;--warn:#d29922;--pane:#0b0f14}
@media(prefers-color-scheme:light){:root{--bg:#fff;--fg:#1f2328;--mut:#59636e;
--card:#f6f8fa;--bd:#d1d9e0;--acc:#0969da;--warn:#9a6700;--pane:#fff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
.wrap{max-width:1400px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:26px;margin:0 0 6px}.sub{color:var(--mut);margin-bottom:26px}
.toc{background:var(--card);border:1px solid var(--bd);border-radius:10px;
padding:14px 18px;margin-bottom:30px}
.toc a{color:var(--acc);text-decoration:none;display:inline-block;margin:3px 14px 3px 0;
font-size:13.5px}
.st{background:var(--card);border:1px solid var(--bd);border-radius:12px;
margin-bottom:26px;overflow:hidden}
.st>h2{margin:0;padding:14px 18px;font-size:17px;border-bottom:1px solid var(--bd);
display:flex;align-items:center;gap:10px}
.body{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);gap:0}
@media(max-width:900px){.body{grid-template-columns:1fr}}
.imgcell{padding:16px;border-right:1px solid var(--bd);background:var(--pane)}
@media(max-width:900px){.imgcell{border-right:0;border-bottom:1px solid var(--bd)}}
.imgcell img{width:100%;height:auto;border-radius:8px;display:block}
.noimg{color:var(--mut);font-size:13px;padding:26px 0;text-align:center;
border:1px dashed var(--bd);border-radius:8px}
.pane{padding:16px 18px;min-width:0}
.what{margin:0 0 14px}
h4{margin:16px 0 6px;font-size:11.5px;letter-spacing:.09em;text-transform:uppercase;
color:var(--mut)}
h4:first-child{margin-top:0}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.src{font-size:12.5px;background:var(--pane);border:1px solid var(--bd);
border-radius:6px;padding:7px 9px;margin-bottom:5px;overflow-x:auto;
white-space:pre}
.kv{width:100%;border-collapse:collapse;font-size:13px}
.kv td{padding:3px 8px 3px 0;vertical-align:top;border-bottom:1px solid var(--bd)}
.kv td:first-child{color:var(--mut);white-space:nowrap;width:1%}
.kv td:last-child{font-family:ui-monospace,monospace;word-break:break-word}
.note{margin-top:14px;padding:11px 13px;border-left:3px solid var(--warn);
background:rgba(210,153,34,.09);border-radius:0 6px 6px 0;font-size:13.5px}
pre.txt{background:var(--pane);border:1px solid var(--bd);border-radius:8px;
padding:11px;font-size:12px;max-height:340px;overflow:auto;margin:8px 0 0}
.n{background:var(--acc);color:#fff;border-radius:6px;padding:1px 8px;font-size:13px;
font-weight:600;flex:none}
"""
    parts = [f"<title>Pipeline trace — {html.escape(stem)}</title>",
             f"<style>{css}</style>", "<div class=wrap>",
             f"<h1>Pipeline trace — {html.escape(stem)}</h1>",
             f"<div class=sub>Every stage the pipeline performs, in order, with "
             f"the file and line that implements it. Source: "
             f"<code>{html.escape(str(src_img))}</code> &middot; "
             f"{len(T.stages)} stages &middot; artifacts in "
             f"<code>{html.escape(str(T.out))}</code></div>"]
    parts.append("<div class=toc><b>Stages</b><br>" + "".join(
        f"<a href='#{s['sid']}'>{i:02d} {html.escape(s['title'])}</a>"
        for i, s in enumerate(T.stages)) + "</div>")

    for i, s in enumerate(T.stages):
        img = b64(s["image"])
        parts.append(f"<div class=st id='{s['sid']}'>"
                     f"<h2><span class=n>{i:02d}</span>"
                     f"{html.escape(s['title'])}</h2><div class=body>")
        parts.append("<div class=imgcell>" + (
            f"<img src='data:image/png;base64,{img}' alt='{s['sid']}'>" if img
            else "<div class=noimg>no image for this stage — it transforms "
                 "structured data, not pixels</div>") + "</div>")
        parts.append("<div class=pane>")
        parts.append(f"<p class=what>{s['what']}</p>")
        if s["sources"]:
            parts.append("<h4>Implemented in</h4>")
            for src in s["sources"]:
                parts.append(f"<div class=src>{html.escape(src)}</div>")
        if s["keys"]:
            parts.append("<h4>Config</h4><table class=kv>")
            for k, v in s["keys"]:
                sv = json.dumps(v) if not isinstance(v, str) else v
                if len(str(sv)) > 150:
                    sv = str(sv)[:150] + " …"
                parts.append(f"<tr><td>{html.escape(k)}</td>"
                             f"<td>{html.escape(str(sv))}</td></tr>")
            parts.append("</table>")
        if s["stats"]:
            parts.append("<h4>Output</h4><table class=kv>")
            for k, v in s["stats"].items():
                parts.append(f"<tr><td>{html.escape(str(k))}</td>"
                             f"<td>{html.escape(str(v))}</td></tr>")
            parts.append("</table>")
        if s["note"]:
            parts.append(f"<div class=note>{s['note']}</div>")
        if s["text"]:
            parts.append(f"<pre class=txt>{html.escape(s['text'])}</pre>")
        parts.append("</div></div></div>")

    parts.append("</div>")
    p = T.out / "trace.html"
    p.write_text("\n".join(parts))
    (T.out / "trace.json").write_text(json.dumps(
        [{k: v for k, v in s.items() if k != "text"} for s in T.stages],
        indent=2, default=str) + "\n")
    print(f"\nwrote {p}\n      {T.out}/trace.json")


if __name__ == "__main__":
    main()
