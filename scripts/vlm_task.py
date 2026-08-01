#!/usr/bin/env python3
"""The VLM anchor task, with no vendor in it.

Both provider runners (``vlm_baseline.py`` for Anthropic, ``vlm_openai.py`` for
OpenAI) import this module and nothing else in common. That is deliberate: if
each runner built its own prompt, image, or schema, a difference between the
two models would be confounded by a difference in what they were asked. Here
the task is constructed once and the runners only carry it over the wire.

Nothing in this file imports a provider SDK.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Both runners resolve provider credentials from the environment. The repo
# already keeps secrets in a gitignored .env (see .env.example), so load it
# here rather than making every invocation export them by hand — which is how
# keys end up in shell history.
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from schematic2netlist.class_head import reclassify as class_head_reclassify
from schematic2netlist.classes import canonical_class, canonical_classes, class_terminals
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.nodes import bbox_xyxy
from schematic2netlist.ports import port_names

PROMPTS = json.loads((ROOT / "configs/vlm_prompts.json").read_text())

SCHEMA_B = {
    "type": "object",
    "properties": {
        "components": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "terminals": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "terminals"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["components"],
    "additionalProperties": False,
}

SCHEMA_A = {
    "type": "object",
    "properties": {
        "components": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "class": {"type": "string"},
                    "bbox": {"type": "array", "items": {"type": "number"}},
                    "terminals": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["class", "bbox", "terminals"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["components"],
    "additionalProperties": False,
}


def load_detections(stem: str, cfg) -> list[dict]:
    """Detections exactly as the pipeline sees them, class head included.

    ``class_head_reclassify`` runs INSIDE ``run_pipeline`` and rewrites labels
    in place, so the on-disk cache still holds the pre-correction classes.
    Reading the cache directly would hand the models different labels than the
    pipeline uses on the same boxes, and would make the scorer align on a
    different class than ``benchmark.py`` does — the comparison would silently
    stop being like for like. Verified: without this, 6 of 40 self-test images
    score differently from the benchmark on identical predictions.
    """
    frame = Path(cfg["preprocess"]["images_dir"]) / f"{stem}.jpg"
    dets = load_cached_detections(
        f"{cfg['detect']['cache_dir']}/{stem}.json",
        min_confidence=cfg["detect"].get("confidence"))
    gray = cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE)
    if gray is not None:
        class_head_reclassify(dets, gray, cfg)
    return dets


def terminal_labels(cls: str) -> list[str]:
    """Named pins where the dataset defines them, else positional."""
    names = port_names(cls)
    n = class_terminals(cls)
    if names and len(names) == n:
        return list(names)
    return [f"terminal {i}" for i in range(n)]


def is_component(det) -> bool:
    """Wire Crossover is a drawing annotation, not a component — it carries no
    terminals and never appears in GT. It is still drawn (the pipeline uses
    those boxes to notch, so hiding them would understate what the model is
    given), but it is never assigned nets."""
    return class_terminals(canonical_class(det["class"])) > 0


def annotate(frame, dets) -> np.ndarray:
    """Draw each detection's box and id. Visual grounding beats coordinates
    alone — the model has to know which symbol id 7 refers to."""
    vis = frame.copy()
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy(d)]
        comp = is_component(d)
        col = (0, 90, 220) if comp else (150, 150, 150)
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        tag = str(i) if comp else "x"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        ly = y1 - 4 if y1 - th - 6 >= 0 else y2 + th + 6
        cv2.rectangle(vis, (x1, ly - th - 4), (x1 + tw + 6, ly + 3), col, -1)
        cv2.putText(vis, tag, (x1 + 3, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def build_task(stem: str, variant: str, cfg) -> dict:
    """The whole request, minus transport.

    Returns {system, text, image_bytes, media_type, schema, n_components}.
    Variant A sends the frame's own JPEG bytes unmodified — re-encoding to PNG
    would inflate the payload roughly 4x for pixels that are already lossy.
    Variant B must draw on the frame, so it re-encodes; PNG there because the
    box outlines and id glyphs are exactly the kind of hard edge JPEG smears.
    """
    frame_path = Path(cfg["preprocess"]["images_dir"]) / f"{stem}.jpg"
    frame = cv2.imread(str(frame_path))
    dets = load_detections(stem, cfg)

    if variant == "b":
        lines = []
        for i, d in enumerate(dets):
            if not is_component(d):
                continue
            cls = canonical_class(d["class"])
            labs = terminal_labels(cls)
            lines.append(f"  id {i}: {cls} — {len(labs)} terminal(s), "
                         f"in order: {', '.join(labs)}")
        text = PROMPTS["variant_b_user"].format(component_list="\n".join(lines))
        ok, buf = cv2.imencode(".png", annotate(frame, dets))
        image_bytes, media_type = buf.tobytes(), "image/png"
        schema = SCHEMA_B
        n_comp = len(lines)
    else:
        comp_classes = [c for c in canonical_classes()
                        if class_terminals(c) > 0 and c != "Wire Crossover"]
        counts = "\n".join(
            f"  {c}: {class_terminals(c)} — {', '.join(terminal_labels(c))}"
            for c in comp_classes)
        text = PROMPTS["variant_a_user"].format(
            class_list=", ".join(comp_classes), terminal_counts=counts,
            width=frame.shape[1], height=frame.shape[0])
        image_bytes, media_type = frame_path.read_bytes(), "image/jpeg"
        schema = SCHEMA_A
        n_comp = sum(1 for d in dets if is_component(d))

    return {"system": PROMPTS["system"], "text": text,
            "image_bytes": image_bytes, "media_type": media_type,
            "schema": schema, "n_components": n_comp, "n_detections": len(dets)}


def split_names(cfg, split: str, limit: int = 0,
                splits_dir: str | None = None) -> list[str]:
    d = Path(splits_dir) if splits_dir else ROOT / "data/splits"
    names = [l.strip() for l in open(d / f"{split}.txt") if l.strip()]
    return names[:limit] if limit else names
