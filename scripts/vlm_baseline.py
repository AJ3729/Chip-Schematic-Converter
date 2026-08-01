#!/usr/bin/env python3
"""The external anchor: a VLM on the same images, scored by the same metrics.

0.4421 strict success has no comparison point — there is no prior work on this
dataset and task, so a reader cannot tell whether it is good. This runs a
frontier vision-language model over the SAME test images and (via
``score_vlm.py``) through the SAME metric cascade, so the number acquires a
meaning.

Two variants, and the second is the scientifically interesting one:

  A  free-form      the model gets the image and must produce components,
                    boxes and nets. This is the baseline a reviewer asks for:
                    can a general model just do this?

  B  connectivity   the model gets the image WITH our detected component boxes
                    drawn and listed, and returns only the net of each
                    terminal. Two things follow. Alignment becomes free — it
                    returns our detection ids, so component matching is the
                    identity map and cannot confound the comparison. And it
                    isolates the one stage that owns the error: the GT-injection
                    oracle attributes terminal-pair F1 as detection 0.065 /
                    wires 0.181 / snapping 0.003, so variant B hands the model
                    detection and snapping for free and tests wire tracing
                    alone.

Variant B is also a second, independent test of the ceiling claim. The finding
that the residual connectivity error is information-limited currently rests
entirely on OUR methods failing, which a reviewer may discount. If a frontier
model with perfect component detection also cannot recover the topology — and
fails on the same images — that is corroboration from a completely different
method class. If it succeeds where we fail, that is real headroom and the
per-image diff says where. Both outcomes are worth having.

Responses are cached per image, so an interrupted sweep resumes and a rerun
costs nothing.

Usage:
    python scripts/vlm_baseline.py --variant b --dry-run     # no API calls
    python scripts/vlm_baseline.py --variant b --repeat 3
    python scripts/score_vlm.py --run-dir results/vlm/b
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from schematic2netlist.class_head import reclassify as class_head_reclassify
from schematic2netlist.classes import canonical_class, canonical_classes, class_terminals
from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.nodes import bbox_xyxy
from schematic2netlist.ports import port_names

MODEL = "claude-opus-5"
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
    Reading the cache directly would hand the VLM different labels than the
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
    terminals and never appears in GT. It still gets drawn (the pipeline uses
    those boxes to notch, so hiding them would understate what the VLM is
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


def build_request(stem: str, variant: str, cfg) -> tuple[dict, np.ndarray, list]:
    idir = Path(cfg["preprocess"]["images_dir"])
    frame = cv2.imread(str(idir / f"{stem}.jpg"))
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
        image = annotate(frame, dets)
        schema = SCHEMA_B
    else:
        comp_classes = [c for c in canonical_classes()
                        if class_terminals(c) > 0 and c != "Wire Crossover"]
        counts = "\n".join(
            f"  {c}: {class_terminals(c)} — {', '.join(terminal_labels(c))}"
            for c in comp_classes)
        text = PROMPTS["variant_a_user"].format(
            class_list=", ".join(comp_classes), terminal_counts=counts,
            width=frame.shape[1], height=frame.shape[0])
        image = frame
        schema = SCHEMA_A

    ok, buf = cv2.imencode(".png", image)
    payload = {
        "model": MODEL,
        "max_tokens": 32000,
        "system": PROMPTS["system"],
        "output_config": {"format": {"type": "json_schema", "schema": schema},
                          "effort": "high"},
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
                                         "media_type": "image/png",
                                         "data": base64.b64encode(buf).decode()}},
            {"type": "text", "text": text},
        ]}],
    }
    return payload, image, dets


def call(client, payload: dict, use_fallback: bool) -> dict:
    """One request. Streamed because thinking is on by default on Opus 5 and
    counts against max_tokens, so responses can be long."""
    kwargs = dict(payload)
    if use_fallback:
        kwargs["betas"] = ["server-side-fallback-2026-07-01"]
        kwargs["fallbacks"] = "default"
        stream_fn = client.beta.messages.stream
    else:
        stream_fn = client.messages.stream
    with stream_fn(**kwargs) as s:
        msg = s.get_final_message()
    if msg.stop_reason == "refusal":
        cat = getattr(getattr(msg, "stop_details", None), "category", None)
        return {"error": "refusal", "category": cat}
    text = next((b.text for b in msg.content if b.type == "text"), None)
    if text is None:
        return {"error": "no_text", "stop_reason": msg.stop_reason}
    out = json.loads(text)
    out["_usage"] = {"input": msg.usage.input_tokens,
                     "output": msg.usage.output_tokens}
    out["_model"] = msg.model
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["a", "b"], default="b")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--repeat", type=int, default=3,
                    help="independent passes; a single pass is not a measurement")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true",
                    help="build and save one request; make no API call")
    ap.add_argument("--no-fallback", action="store_true",
                    help="disable the server-side refusal fallback")
    args = ap.parse_args()

    cfg = load_config(args.config)
    names = [l.strip() for l in open(ROOT / f"data/splits/{args.split}.txt")
             if l.strip()]
    if args.limit:
        names = names[: args.limit]
    out = Path(args.out_dir or f"results/vlm/{args.variant}")
    out.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        stem = Path(names[0]).stem
        payload, image, dets = build_request(stem, args.variant, cfg)
        cv2.imwrite(str(out / "dryrun_image.png"), image)
        txt = payload["messages"][0]["content"][1]["text"]
        (out / "dryrun_prompt.txt").write_text(
            f"SYSTEM:\n{payload['system']}\n\n{'='*60}\nUSER:\n{txt}")
        b64 = payload["messages"][0]["content"][0]["source"]["data"]
        print(f"variant {args.variant}, {len(names)} images x {args.repeat} "
              f"repeats = {len(names)*args.repeat} calls")
        n_c = sum(1 for d in dets if is_component(d))
        print(f"  first image  : {stem}, {len(dets)} detections "
              f"({n_c} components, {len(dets)-n_c} crossings)")
        print(f"  image bytes  : {len(b64)*3//4/1024:.0f} KiB PNG")
        print(f"  prompt chars : {len(txt)}")
        print(f"  wrote {out}/dryrun_image.png and dryrun_prompt.txt")
        print("\nNo API call made. Inspect those two files, then drop --dry-run.")
        return

    try:
        import anthropic
    except ImportError:
        sys.exit("pip install anthropic")
    client = anthropic.Anthropic()
    use_fallback = not args.no_fallback
    lock = threading.Lock()
    done = [0]
    total = len(names) * args.repeat

    def work(job):
        nm, rep = job
        stem = Path(nm).stem
        dst = out / f"rep{rep}" / f"{stem}.json"
        # A cached ERROR must not count as done, or a transient failure (or a
        # missing key) is baked in and the rerun silently skips it.
        if dst.exists() and "error" not in json.loads(dst.read_text()):
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        payload, _, _ = build_request(stem, args.variant, cfg)
        try:
            res = call(client, payload, use_fallback)
        except Exception as e:  # noqa: BLE001 — recorded, not swallowed
            msg = str(e)
            if use_fallback and "fallback" in msg.lower():
                res = call(client, payload, False)   # SDK/API too old for it
            else:
                res = {"error": type(e).__name__, "message": msg[:400]}
        dst.write_text(json.dumps(res, indent=1))
        with lock:
            done[0] += 1
            if done[0] % 10 == 0 or done[0] == total:
                print(f"  [{done[0]}/{total}]", flush=True)

    jobs = [(nm, r) for r in range(args.repeat) for nm in names]

    # Preflight on one job before fanning out: a bad credential or an
    # unsupported parameter should cost one request, not the whole sweep.
    work(jobs[0])
    probe = out / f"rep{jobs[0][1]}" / f"{Path(jobs[0][0]).stem}.json"
    if probe.exists():
        first = json.loads(probe.read_text())
        if "error" in first:
            sys.exit(f"first request failed, aborting before the remaining "
                     f"{len(jobs)-1} jobs:\n  {first.get('error')}: "
                     f"{first.get('message', '')}\n\n"
                     f"If this is authentication: export ANTHROPIC_API_KEY, or "
                     f"run `ant auth login`.")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, jobs[1:]))

    errs = [p for p in out.rglob("*.json")
            if "error" in json.loads(p.read_text())]
    print(f"\nwrote {out}  ({len(jobs)} jobs, {len(errs)} errored)")
    for p in errs[:5]:
        print(f"  ! {p.relative_to(out)}: "
              f"{json.loads(p.read_text()).get('error')}")
    print(f"\nscore with: python scripts/score_vlm.py --run-dir {out} "
          f"--variant {args.variant}")


if __name__ == "__main__":
    main()
