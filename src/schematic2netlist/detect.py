"""Component detection.

Backends:
- "cached": load a per-image JSON produced by an earlier detection run.
- "ultralytics": local YOLO inference (batch-capable), the primary path
  once local weights exist (Phase C).
- "roboflow": hosted Roboflow API, kept as a legacy fallback.

Detections are normalized to::

    {"class": str, "confidence": float,
     "x": cx, "y": cy, "width": w, "height": h}   # center-based bbox

This module fixes the two legacy evaluation bugs:
(a) the legacy evaluator read det["class_name"] while the JSON stored
    "class" — `normalize_detection` accepts either key;
(b) one shared detections.json was applied to every image — detections
    are now cached per image under detect.cache_dir keyed by image stem.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from schematic2netlist.classes import canonical_class


def normalize_detection(det: dict) -> dict:
    """Return a detection dict with the canonical "class" key and the
    class name canonicalized to the published Digitize-HCD vocabulary
    (legacy Roboflow names are mapped via aliases)."""
    if "class" in det:
        cls = det["class"]
    elif "class_name" in det:
        cls = det["class_name"]
    else:
        raise KeyError(
            "Detection has neither 'class' nor 'class_name': "
            f"{sorted(det.keys())}"
        )
    return {
        "class": canonical_class(cls),
        "confidence": float(det.get("confidence", 1.0)),
        "x": float(det["x"]),
        "y": float(det["y"]),
        "width": float(det["width"]),
        "height": float(det["height"]),
    }


def load_cached_detections(path: str | Path) -> list[dict]:
    """Load and normalize detections from a JSON cache file.

    Accepts both legacy shapes: {"detections": [...]} and
    {"predictions": [...]}.
    """
    with open(path) as f:
        data = json.load(f)
    dets = data.get("detections", data.get("predictions"))
    if dets is None:
        raise ValueError(f"{path}: no 'detections' or 'predictions' key")
    return [normalize_detection(d) for d in dets]


def cache_path_for_image(image_path: str | Path, cfg: dict) -> Path:
    """Per-image detection cache location: <cache_dir>/<image stem>.json."""
    return Path(cfg["detect"]["cache_dir"]) / (Path(image_path).stem + ".json")


def save_detections(image_path: str | Path, detections: list[dict], cfg: dict) -> Path:
    out = cache_path_for_image(image_path, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {"image": os.path.basename(str(image_path)), "detections": detections},
            f,
            indent=4,
        )
    return out


def detect_roboflow(image_path: str | Path, cfg: dict) -> list[dict]:
    """Hosted Roboflow inference (legacy fallback). Needs ROBOFLOW_API_KEY."""
    import cv2  # deferred: keep import cost out of cached path

    try:
        from dotenv import load_dotenv
        from inference_sdk import InferenceHTTPClient
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Roboflow backend requires the 'roboflow' extra: "
            "pip install -e '.[roboflow]'"
        ) from e

    load_dotenv()
    client = InferenceHTTPClient(
        api_url=cfg["detect"]["api_url"],
        api_key=os.getenv("ROBOFLOW_API_KEY"),
    )
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    results = client.infer(image, model_id=cfg["detect"]["model_id"])
    return [normalize_detection(p) for p in results["predictions"]]


def detect_ultralytics(image_paths: list[str | Path], cfg: dict) -> list[list[dict]]:
    """Local batch YOLO inference. Requires weights + the 'train' extra."""
    try:
        from ultralytics import YOLO
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Ultralytics backend requires the 'train' extra: "
            "pip install -e '.[train]'"
        ) from e

    weights = cfg["detect"]["weights"]
    if not weights:
        raise ValueError("detect.weights must point to local YOLO weights")
    model = YOLO(weights)
    results = model.predict(
        [str(p) for p in image_paths],
        conf=cfg["detect"]["confidence"],
        imgsz=cfg["detect"]["image_size"],
        verbose=False,
    )
    all_dets: list[list[dict]] = []
    for res in results:
        dets = []
        names = res.names
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            dets.append(
                {
                    "class": names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "x": (x1 + x2) / 2,
                    "y": (y1 + y2) / 2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                }
            )
        all_dets.append(dets)
    return all_dets


def detect(image_path: str | Path, cfg: dict) -> list[dict]:
    """Detect components in one image using the configured backend.

    Every backend writes/reads the per-image cache so downstream stages
    never share detections across images.
    """
    backend = cfg["detect"]["backend"]
    cache = cache_path_for_image(image_path, cfg)

    if cache.exists():
        return load_cached_detections(cache)
    if backend == "cached":
        raise FileNotFoundError(
            f"No cached detections at {cache}. Run detection with the "
            "'ultralytics' or 'roboflow' backend, or pass an explicit "
            "detections file."
        )
    if backend == "roboflow":
        dets = detect_roboflow(image_path, cfg)
    elif backend == "ultralytics":
        dets = detect_ultralytics([image_path], cfg)[0]
    else:
        raise ValueError(f"Unknown detect.backend: {backend!r}")
    save_detections(image_path, dets, cfg)
    return dets
