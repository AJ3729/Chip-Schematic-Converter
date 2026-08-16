#!/usr/bin/env python3
"""Imaging characterization of both corpora (task B4).

Answers the photographic-robustness question with measurement rather than
assertion: how do Digitize-HCD and CGHD actually differ as photographs?

Measured per image, on the ORIGINAL photographs (not the rectified frames,
which have already had the differences normalised out of them):

  rectification magnitude   how far the recorded transform is from the
                            identity -- i.e. how skewed the shot was
  shadow field strength     the std of the illumination field the shadow
                            normalisation stage estimates
  native resolution         pixels, and aspect ratio
  JPEG quality              estimated from the quantisation tables
  EXIF                      camera make/model where present
  background variation      pixel std outside the page region

Usage:
    python scripts/corpus_characterization.py
    python scripts/corpus_characterization.py --limit 150
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "reports/corpus_characterization.md"
OUTJ = ROOT / "results/corpus_characterization.json"


def jpeg_quality(path: Path) -> float | None:
    """Estimate JPEG quality from the luminance quantisation table.

    The standard tables scale linearly with quality, so the mean of the table
    inverts to an approximate setting. Returns None for PNG or on any parse
    failure -- an absent number is better than an invented one.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return None
    i = 0
    while i < len(data) - 1:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xDB:                       # DQT
            ln = int.from_bytes(data[i + 2:i + 4], "big")
            tbl = data[i + 5:i + 5 + 64]
            if len(tbl) == 64:
                m = sum(tbl) / 64.0
                q = (100 - m * 100 / 255) if m > 0 else None
                return round(float(np.clip(q, 1, 100)), 1) if q else None
            i += 2 + ln
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xDA:
            break
        ln = int.from_bytes(data[i + 2:i + 4], "big")
        i += 2 + ln
    return None


def exif_camera(path: Path) -> str | None:
    try:
        from PIL import Image, ExifTags
        with Image.open(path) as im:
            ex = im.getexif()
            if not ex:
                return None
            tags = {ExifTags.TAGS.get(k, k): v for k, v in ex.items()}
            make = str(tags.get("Make", "")).strip()
            model = str(tags.get("Model", "")).strip()
            s = f"{make} {model}".strip()
            return s or None
    except Exception:                                        # noqa: BLE001
        return None


def measure(path: Path, meta: dict | None) -> dict | None:
    im = cv2.imread(str(path))
    if im is None:
        return None
    H, W = im.shape[:2]
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # illumination field: heavy blur approximates the shadow the normaliser
    # removes; its spread is the shadow strength.
    field = cv2.GaussianBlur(g, (0, 0), sigmaX=max(W, H) / 25.0)
    shadow = float(np.std(field))

    # background variation: the outer 8% border, which is page/table rather
    # than drawing on almost every shot.
    b = max(4, int(0.08 * min(W, H)))
    border = np.concatenate([g[:b].ravel(), g[-b:].ravel(),
                             g[:, :b].ravel(), g[:, -b:].ravel()])
    bg_std = float(np.std(border))

    rect = None
    if meta:
        # deviation of the recorded rectification from the identity
        ang = meta.get("rotation_deg") or meta.get("skew_deg") or meta.get("angle")
        if ang is not None:
            rect = abs(float(ang))
    return {
        "width": W, "height": H, "megapixels": round(W * H / 1e6, 3),
        "aspect": round(W / H, 4),
        "shadow_field_std": round(shadow, 3),
        "background_std": round(bg_std, 3),
        "jpeg_quality_est": jpeg_quality(path),
        "camera": exif_camera(path),
        "rectification_deg": rect,
    }


def summarize(rows: list[dict], key: str) -> dict:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "mean": round(statistics.mean(vals), 3),
            "median": round(statistics.median(vals), 3),
            "p10": round(float(np.percentile(vals, 10)), 3),
            "p90": round(float(np.percentile(vals, 90)), 3)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=400)
    a = ap.parse_args()

    tf = {}
    p = ROOT / "data/transforms_1024.json"
    if p.exists():
        tf = json.loads(p.read_text())

    corpora: dict[str, list[dict]] = {}

    import random as _r0
    hcd_all = sorted((ROOT / "data/raw").glob("*.jpg"))
    _r0.Random(0).shuffle(hcd_all)
    hcd = sorted(hcd_all[: a.limit])
    rows = []
    for f in hcd:
        m = measure(f, tf.get(f.stem))
        if m:
            m["stem"] = f.stem
            rows.append(m)
    corpora["Digitize-HCD"] = rows
    print(f"Digitize-HCD: {len(rows)} images measured")

    # drafter_0 is EXCLUDED everywhere else in this work (917 of its 1,038
    # images carry no annotation), so it must be excluded here too -- a
    # characterization of images that appear in no result is worthless. And
    # sample randomly across drafters: taking the first N by path order gave a
    # sample that was 100% drafter_0.
    import random as _rnd
    cg_all = [f for f in (ROOT / "data/cghd/extracted").glob("drafter_*/images/*")
              if f.suffix.lower() in (".jpg", ".jpeg", ".png")
              and f.parent.parent.name != "drafter_0"]
    _rnd.Random(0).shuffle(cg_all)
    cg = sorted(cg_all[: a.limit])
    rows = []
    for f in cg:
        m = measure(f, None)
        if m:
            m["stem"] = f.stem
            m["drafter"] = f.parent.parent.name
            rows.append(m)
    corpora["CGHD"] = rows
    print(f"CGHD: {len(rows)} images measured")

    KEYS = ["megapixels", "aspect", "shadow_field_std", "background_std",
            "jpeg_quality_est"]
    summary = {c: {k: summarize(r, k) for k in KEYS} for c, r in corpora.items()}
    for c, r in corpora.items():
        cams = collections.Counter(x["camera"] for x in r if x.get("camera"))
        summary[c]["cameras"] = dict(cams.most_common(8))
        summary[c]["n_with_exif"] = sum(1 for x in r if x.get("camera"))
        summary[c]["portrait_fraction"] = round(
            sum(1 for x in r if x["aspect"] < 1) / len(r), 4) if r else None

    OUTJ.write_text(json.dumps(
        {"_what": "Imaging characterization of both corpora, measured on the "
                  "ORIGINAL photographs.",
         "summary": summary,
         "per_image": {c: r for c, r in corpora.items()}}, indent=1) + "\n")

    lines = [
        "# Corpus characterization (task B4)",
        "",
        "Measured on the **original photographs**, not the rectified frames —",
        "rectification exists precisely to remove these differences, so",
        "measuring the frames would hide what the corpora actually differ in.",
        "",
        f"Digitize-HCD: {len(corpora['Digitize-HCD'])} images. "
        f"CGHD: {len(corpora['CGHD'])} images (v12).",
        "",
        "## Side by side",
        "",
        "| property | Digitize-HCD | CGHD |",
        "| --- | --- | --- |",
    ]
    labels = {"megapixels": "resolution (MP)", "aspect": "aspect ratio w/h",
              "shadow_field_std": "shadow field strength",
              "background_std": "background variation",
              "jpeg_quality_est": "JPEG quality (est.)"}
    for k in KEYS:
        h, c = summary["Digitize-HCD"][k], summary["CGHD"][k]
        f = lambda d: (f"{d['median']} [{d['p10']}–{d['p90']}]"  # noqa: E731
                       if d.get("n") else "—")
        lines.append(f"| {labels[k]} (median [p10–p90]) | {f(h)} | {f(c)} |")
    lines.append(f"| portrait fraction | "
                 f"{summary['Digitize-HCD']['portrait_fraction']} | "
                 f"{summary['CGHD']['portrait_fraction']} |")
    lines.append(f"| images with EXIF camera | "
                 f"{summary['Digitize-HCD']['n_with_exif']} | "
                 f"{summary['CGHD']['n_with_exif']} |")
    def med(c, k):
        return summary[c][k].get("median")

    ratios = {k: (med("CGHD", k) / med("Digitize-HCD", k))
              for k in KEYS
              if med("CGHD", k) and med("Digitize-HCD", k)}
    biggest = sorted(ratios.items(),
                     key=lambda kv: abs(np.log(kv[1])), reverse=True)

    lines += [
        "",
        "## Conclusion",
        "",
        "Derived from the table above, ranked by how far each property differs",
        "(|log ratio|), rather than asserted:",
        "",
    ]
    for k, r in biggest:
        direction = "higher" if r > 1 else "lower"
        lines.append(f"- **{labels[k]}**: CGHD is {r:.2f}x — "
                     f"{abs(1-r)*100:.0f}% {direction} than Digitize-HCD.")
    lines += [
        "",
        "**Sampling.** Both corpora are sampled uniformly at random with a",
        "fixed seed. `drafter_0` is excluded from CGHD, as it is everywhere",
        "else in this work: 917 of its 1,038 images carry no annotation, so it",
        "appears in no result and characterising it would describe images the",
        "evaluation never sees.",
        "",
        "**Reading this against the transfer result.** The cross-corpus",
        "detection drop tracks component scale (recall 0.183 in the smallest",
        "size quintile against ~0.55 elsewhere). Whichever imaging properties",
        "differ most above are candidates for *why* small components are hard —",
        "compression and background clutter both destroy the fine strokes that",
        "distinguish a small symbol — but this table establishes association,",
        "not cause. No experiment here isolates one property.",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT} and {OUTJ}")
    for k in KEYS:
        print(f"  {labels[k]:26s} HCD {summary['Digitize-HCD'][k].get('median')}"
              f"   CGHD {summary['CGHD'][k].get('median')}")


if __name__ == "__main__":
    main()
