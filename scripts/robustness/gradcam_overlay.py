#!/usr/bin/env python3
"""Grad-CAM over the YOLOv8 detector, drawn on the cleaned frame AND the raw scan.

Two views, because they answer different questions. The CLEANED overlay shows
where the detector looked in the space it actually operates in -- the 1024 frame
at imgsz 640 -- and is the honest picture of the model's attention. The RAW
overlay puts that same attention back on the photograph a person recognises,
which is the only view in which "it ignored the transistor in the corner" is a
sentence someone can check against the drawing.

Method is EigenCAM by default. Grad-CAM proper needs a scalar to differentiate,
and a detector's output is a set of boxes, so a target has to be invented --
usually "sum the objectness of the top-k boxes", which then makes the map a
function of that arbitrary choice. EigenCAM takes the first principal component
of the activations and needs no target at all, so the map is a property of the
features rather than of a scoring convention. --method gradcam is available for
comparison and uses summed objectness.

The raw overlay is warped through the SAME recorded transform the pipeline used
(rotation, optional rot90, crop, scale, canvas offset), vectorised with
cv2.remap rather than looped per pixel.

Usage:
    python scripts/robustness/gradcam_overlay.py --condition clean --limit 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pytorch_grad_cam import EigenCAM, GradCAM              # noqa: E402
from pytorch_grad_cam.utils.image import show_cam_on_image  # noqa: E402


class YoloTensorOut(torch.nn.Module):
    """YOLOv8 returns (predictions, features); pytorch-grad-cam wants a tensor.

    Unwrapped, base_cam calls .cpu() on the tuple and dies. Returning just the
    prediction tensor [B, 4+nc, anchors] makes the library's own target
    handling work unchanged for both methods.
    """

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        o = self.m(x)
        t = o[0] if isinstance(o, (list, tuple)) else o
        while isinstance(t, (list, tuple)):
            t = t[0]
        return t


def load_model(weights: str):
    from ultralytics import YOLO
    y = YOLO(weights)
    m = y.model.float().eval()
    for p in m.parameters():
        p.requires_grad_(True)
    return YoloTensorOut(m)


def target_layers(w):
    """The two FINEST feature maps feeding the detect head.

    The head itself is a poor choice: its outputs are already decoded boxes, so
    a CAM there shows the decoding grid rather than the evidence. Detect draws
    from three levels -- P3 80x80, P4 40x40, P5 20x20 -- and the obvious pick,
    the last block before the head, is P5.

    P5 is the wrong one, and silently so. Measured over ten images, the twenty
    most confident anchors land 189 times on P3, 11 on P4 and NEVER on P5:
    components in this corpus are small enough relative to the 640 px input
    that the stride-32 branch contributes nothing. Its gradient is therefore
    exactly 0.0, and a Grad-CAM aimed there returns a uniform map that looks
    like a working heatmap and carries no information. Taking the two finest
    levels instead is what makes the gradient method produce anything at all.
    """
    seq = w.m.model
    feeds = list(seq[-1].f)          # [15, 18, 21] for yolov8s
    return [seq[i] for i in feeds[:2]]


class TopKConfidence:
    """Scalar target for gradcam: the summed confidence of the strongest boxes.

    Summing ALL 8400 anchors would be dominated by background and the map goes
    flat; the top 20 are the boxes the detector would actually emit.
    """

    def __call__(self, out):
        cls = out[4:, :] if out.dim() == 2 else out[0, 4:, :]
        best = cls.sigmoid().max(dim=0)[0]
        return best.topk(min(20, best.numel()))[0].sum()


def cam_for(cam, img_1024: np.ndarray, targets=None, size: int = 640) -> np.ndarray:
    """CAM at 1024x1024, computed at the detector's own input size."""
    x = cv2.resize(img_1024, (size, size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
    t = (t.float() / 255.0).unsqueeze(0)
    g = cam(input_tensor=t, targets=targets)[0]
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    return cv2.resize(g, (1024, 1024), interpolation=cv2.INTER_LINEAR)


def overlay(bgr: np.ndarray, heat: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return cv2.cvtColor(
        (show_cam_on_image(rgb, heat, use_rgb=True, image_weight=1 - alpha)
         ).astype(np.uint8), cv2.COLOR_RGB2BGR)


def warp_to_raw(heat: np.ndarray, meta: dict, raw_hw: tuple[int, int]) -> np.ndarray:
    """Resample a 1024-frame heatmap onto raw-image geometry.

    Builds the forward map raw->cleaned for every raw pixel and hands it to
    cv2.remap. project_point is applied as array algebra; calling it per pixel
    would be ~5M Python calls per image.
    """
    H, W = raw_hw
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    m = np.asarray(meta["rotation_matrix"], dtype=np.float32)
    xr = m[0, 0] * xs + m[0, 1] * ys + m[0, 2]
    yr = m[1, 0] * xs + m[1, 1] * ys + m[1, 2]
    if meta["rotated90"]:
        w_before = meta["size_before_rot90"][0]
        xr, yr = yr, (w_before - 1) - xr
    cx, cy = meta["crop"][0], meta["crop"][1]
    s = meta["scale"]
    ox, oy = meta["canvas_offset"]
    mx = ((xr - cx) * s + ox).astype(np.float32)
    my = ((yr - cy) * s + oy).astype(np.float32)
    return cv2.remap(heat, mx, my, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", default="clean")
    ap.add_argument("--method", choices=("gradcam", "eigencam"), default="gradcam")
    ap.add_argument("--split", default="test")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-root", default="results/robustness/gradcam")
    a = ap.parse_args()

    import yaml
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    model = load_model(cfg["detect"]["weights"])

    cond = a.condition
    clean_dir = ROOT / "data/robustness/cleaned" / cond
    raw_dir = ROOT / "data/robustness/raw" / cond
    tf_path = ROOT / "data/robustness/transforms" / f"{cond}.json"
    if not tf_path.exists():
        raise SystemExit(f"no transforms for {cond}: run run_condition.py first")
    tf = json.loads(tf_path.read_text())

    out_c = ROOT / a.out_root / "cleaned" / cond
    out_r = ROOT / a.out_root / "raw" / cond
    out_c.mkdir(parents=True, exist_ok=True)
    out_r.mkdir(parents=True, exist_ok=True)

    stems = [Path(l.strip()).stem for l in
             (ROOT / a.splits_dir / f"{a.split}.txt").read_text().split() if l.strip()]
    if a.limit:
        stems = stems[:a.limit]

    CamCls = EigenCAM if a.method == "eigencam" else GradCAM
    cam = CamCls(model=model, target_layers=target_layers(model))
    targets = None if a.method == "eigencam" else [TopKConfidence()]

    n = 0
    for stem in stems:
        # lossless arms store both the corrupted scan and the cleaned frame
        # as PNG, so neither extension can be assumed
        cp = next((clean_dir / f"{stem}{e}" for e in (".jpg", ".png")
                   if (clean_dir / f"{stem}{e}").exists()), clean_dir / f"{stem}.jpg")
        rp = next((raw_dir / f"{stem}{e}" for e in (".jpg", ".png")
                   if (raw_dir / f"{stem}{e}").exists()), raw_dir / f"{stem}.jpg")
        if not cp.exists() or stem not in tf:
            continue
        img_c = cv2.imread(str(cp))
        heat = cam_for(cam, img_c, targets)
        cv2.imwrite(str(out_c / f"{stem}_cam.jpg"), overlay(img_c, heat),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        img_r = cv2.imread(str(rp))
        if img_r is not None:
            hr = warp_to_raw(heat, tf[stem], img_r.shape[:2])
            cv2.imwrite(str(out_r / f"{stem}_cam.jpg"), overlay(img_r, hr),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        n += 1
    print(f"[{cond}] {a.method}: {n} images")
    print(f"  cleaned overlays -> {out_c.relative_to(ROOT)}")
    print(f"  raw overlays     -> {out_r.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
