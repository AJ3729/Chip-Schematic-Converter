#!/usr/bin/env python3
"""Capture-degradation models applied to a COPY of the raw scans.

Two families, kept apart because they are not equally safe to score.

PHOTOMETRIC corruptions change pixel values and leave geometry alone, so the
ground truth -- which lives in the 1024 frame produced by preprocessing -- stays
valid by construction. Any accuracy drop is the pipeline's.

GEOMETRIC corruptions move the ink. Preprocessing is supposed to rectify them,
and whether it does is exactly the interesting question, but if it does NOT the
recovered 1024 frame no longer coincides with the one the GT was traced in, and
a scoring drop then mixes "the pipeline failed" with "the boxes stopped lining
up". These are still run, and the frame drift is measured separately per
condition so the two causes can be told apart rather than silently blended.

Severity 0 is the identity. It exists so the harness can be shown to reproduce
the published numbers before any conclusion is drawn from the corrupted ones --
a robustness sweep whose clean arm does not match is measuring its own plumbing.
"""

from __future__ import annotations

import hashlib

import cv2
import numpy as np

# Deterministic per (condition, image): the same corrupted pixel is produced on
# every run, so a rerun is a rerun and not a new sample.
#
# blake2b, NOT hash(). Python salts string hashing per interpreter unless
# PYTHONHASHSEED is set, so hash(("circuit_1013", "gauss3")) differs between
# processes and every subprocess drew a FRESH noise field. The corruptions were
# still valid draws at the right parameters, but they were not reproducible, and
# two arms meant to receive identical pixels did not. A project whose central
# claim is determinism cannot ship a stochastic harness that only looks seeded.
def _rng(stem: str, cond: str) -> np.random.Generator:
    digest = hashlib.blake2b(f"{stem}\x00{cond}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big") % (2**32))


def clean(img, sev, stem):
    return img.copy()


def gauss_noise(img, sev, stem):
    sigma = {1: 8, 2: 16, 3: 32}[sev]
    r = _rng(stem, f"gauss{sev}")
    out = img.astype(np.float32) + r.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def speckle(img, sev, stem):
    """Salt and pepper: scanner dust and dead pixels, not sensor noise."""
    p = {1: 0.01, 2: 0.03, 3: 0.06}[sev]
    r = _rng(stem, f"speckle{sev}")
    out = img.copy()
    m = r.random(img.shape[:2])
    out[m < p / 2] = 0
    out[(m >= p / 2) & (m < p)] = 255
    return out


def blur(img, sev, stem):
    k = {1: 3, 2: 7, 3: 13}[sev]
    return cv2.GaussianBlur(img, (k, k), 0)


def jpeg(img, sev, stem):
    q = {1: 40, 2: 20, 3: 10}[sev]
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img.copy()


def brightness(img, sev, stem):
    """Under-exposure: the failure mode of a phone photo in a dim room."""
    g = {1: 0.75, 2: 0.55, 3: 0.40}[sev]
    return np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)


def contrast(img, sev, stem):
    """Washed-out ink -- pencil on white under flat light."""
    a = {1: 0.75, 2: 0.55, 3: 0.40}[sev]
    m = img.astype(np.float32).mean()
    return np.clip((img.astype(np.float32) - m) * a + m, 0, 255).astype(np.uint8)


def downscale(img, sev, stem):
    """Resolution loss, then resampled back: a photo taken too far away."""
    f = {1: 0.75, 2: 0.50, 3: 0.35}[sev]
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def rotate(img, sev, stem):
    deg = {1: 2.0, 2: 5.0, 3: 10.0}[sev]
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def perspective(img, sev, stem):
    """A page photographed off-axis; what the rectifier is meant to undo."""
    f = {1: 0.02, 2: 0.05, 3: 0.09}[sev]
    h, w = img.shape[:2]
    d = f * min(h, w)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[d, d * 0.6], [w - d * 0.6, 0],
                      [w - d, h - d * 0.6], [d * 0.6, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


PHOTOMETRIC = {
    "gauss_noise": gauss_noise, "speckle": speckle, "blur": blur,
    "jpeg": jpeg, "brightness": brightness, "contrast": contrast,
    "downscale": downscale,
}
GEOMETRIC = {"rotate": rotate, "perspective": perspective}
ALL = {"clean": clean, **PHOTOMETRIC, **GEOMETRIC}


def conditions(severities=(1, 2, 3)) -> list[tuple[str, str, int]]:
    """(condition_name, family, severity), control first."""
    out = [("clean", "control", 0)]
    for fam in PHOTOMETRIC:
        out += [(f"{fam}_s{s}", "photometric", s) for s in severities]
    for fam in GEOMETRIC:
        out += [(f"{fam}_s{s}", "geometric", s) for s in severities]
    return out


def apply(name: str, img, stem: str):
    """Corrupt `img` for condition `name`.

    A trailing "_fix" or "_lossless" is stripped first. Those arms re-run the SAME corruption
    through the patched preprocessing copy, so they must receive byte-identical
    pixels -- otherwise the comparison measures new noise as well as the fix.
    The per-image seed keys on the family and severity rather than the full
    condition name, so stripping the suffix reproduces the pixels exactly.
    """
    for suffix in ("_fix", "_lossless"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    if name == "clean":
        return clean(img, 0, stem)
    fam, sev = name.rsplit("_s", 1)
    return ALL[fam](img, int(sev), stem)
