#!/usr/bin/env python3
"""The external anchor, Anthropic side: Claude on the same images, same metrics.

0.4421 strict success has no comparison point — there is no prior work on this
dataset and task, so a reader cannot tell whether it is good. This runs a
frontier model over the SAME test images and (via ``score_vlm.py``) through the
SAME metric cascade, so the number acquires a meaning.

Two variants, and the second is the scientifically interesting one:

  A  free-form      the model gets the image and must produce components,
                    boxes and nets. The baseline a reviewer asks for: can a
                    general model just do this?

  B  connectivity   the model gets the image WITH our detected component boxes
                    drawn and listed, and returns only the net of each
                    terminal. Alignment becomes free — it returns our detection
                    ids, so component matching is the identity map and cannot
                    confound the comparison. And it isolates the stage that
                    owns the error: the GT-injection oracle attributes
                    terminal-pair F1 as detection 0.065 / wires 0.181 /
                    snapping 0.003, so variant B hands the model detection and
                    snapping for free and tests wire tracing alone.

Variant B is also a second, independent test of the ceiling claim, which
currently rests entirely on OUR methods failing. Run alongside ``vlm_openai.py``
it becomes a third: two frontier models from different labs failing on the SAME
images is much harder to attribute to our implementation.

Batch mode is the default and costs 50% less. Submission is one batch per
repeat (190 requests, ~60 MB — the API cap is 256 MB), batch ids are checkpointed
so an interrupted poll resumes, and results are keyed by custom_id because the
Batches API returns them in arbitrary order.

Usage:
    python scripts/vlm_baseline.py --variant b --dry-run      # no API calls
    python scripts/vlm_baseline.py --variant b --repeat 3     # batched
    python scripts/vlm_baseline.py --variant b --sync         # live, 50% dearer
    python scripts/score_vlm.py --run-dir results/vlm/claude_b --variant b
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from vlm_task import annotate, build_task, load_detections, split_names

MODEL = "claude-opus-5"


def params_for(task: dict) -> dict:
    """Messages-API parameters for one image. Shared by sync and batch so the
    two paths cannot drift apart."""
    return {
        "model": MODEL,
        "max_tokens": 32000,
        "system": task["system"],
        "output_config": {
            "format": {"type": "json_schema", "schema": task["schema"]},
            "effort": "high",
        },
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {
                "type": "base64", "media_type": task["media_type"],
                "data": base64.b64encode(task["image_bytes"]).decode()}},
            {"type": "text", "text": task["text"]},
        ]}],
    }


def parse_message(msg) -> dict:
    """One API message -> the cached response shape score_vlm.py reads."""
    if msg.stop_reason == "refusal":
        cat = getattr(getattr(msg, "stop_details", None), "category", None)
        return {"error": "refusal", "category": cat}
    text = next((b.text for b in msg.content if b.type == "text"), None)
    if text is None:
        return {"error": "no_text", "stop_reason": msg.stop_reason}
    try:
        out = json.loads(text)
    except json.JSONDecodeError as e:
        return {"error": "bad_json", "message": str(e)[:200]}
    out["_usage"] = {"input": msg.usage.input_tokens,
                     "output": msg.usage.output_tokens}
    out["_model"] = msg.model
    return out


def is_done(path: Path) -> bool:
    """A cached ERROR does not count as done, or a transient failure (or a bad
    key) is baked in and the rerun silently skips it."""
    return path.exists() and "error" not in json.loads(path.read_text())


# --------------------------------------------------------------------- batch

def run_batch(client, names, args, cfg, out: Path) -> None:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    state_path = out / "batches.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    for rep in range(args.repeat):
        rep_dir = out / f"rep{rep}"
        rep_dir.mkdir(parents=True, exist_ok=True)
        todo = [nm for nm in names if not is_done(rep_dir / f"{Path(nm).stem}.json")]
        if not todo:
            print(f"rep{rep}: already complete")
            continue

        bid = state.get(str(rep))
        if not bid:
            reqs = [Request(custom_id=Path(nm).stem,
                            params=MessageCreateParamsNonStreaming(
                                **params_for(build_task(Path(nm).stem, args.variant, cfg))))
                    for nm in todo]
            batch = client.messages.batches.create(requests=reqs)
            bid = batch.id
            state[str(rep)] = bid
            state_path.write_text(json.dumps(state, indent=1))
            print(f"rep{rep}: submitted {len(reqs)} requests as {bid}")
        else:
            print(f"rep{rep}: resuming {bid}")

        while True:
            b = client.messages.batches.retrieve(bid)
            if b.processing_status == "ended":
                break
            c = b.request_counts
            print(f"  rep{rep} {b.processing_status}: "
                  f"processing={c.processing} succeeded={c.succeeded} "
                  f"errored={c.errored}", flush=True)
            time.sleep(args.poll_seconds)

        n_ok = n_err = 0
        for r in client.messages.batches.results(bid):
            # Results come back in ARBITRARY order — key on custom_id, never
            # on position.
            dst = rep_dir / f"{r.custom_id}.json"
            if r.result.type == "succeeded":
                res = parse_message(r.result.message)
            else:
                res = {"error": r.result.type,
                       "message": str(getattr(r.result, "error", ""))[:300]}
            n_ok += "error" not in res
            n_err += "error" in res
            dst.write_text(json.dumps(res, indent=1))
        print(f"rep{rep}: wrote {n_ok} ok, {n_err} errored")


# ---------------------------------------------------------------------- sync

def run_sync(client, names, args, cfg, out: Path) -> None:
    lock, done = threading.Lock(), [0]
    jobs = [(nm, r) for r in range(args.repeat) for nm in names]

    def work(job):
        nm, rep = job
        stem = Path(nm).stem
        dst = out / f"rep{rep}" / f"{stem}.json"
        if is_done(dst):
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            with client.messages.stream(
                    **params_for(build_task(stem, args.variant, cfg))) as s:
                res = parse_message(s.get_final_message())
        except Exception as e:  # noqa: BLE001 — recorded, not swallowed
            res = {"error": type(e).__name__, "message": str(e)[:400]}
        dst.write_text(json.dumps(res, indent=1))
        with lock:
            done[0] += 1
            if done[0] % 10 == 0:
                print(f"  [{done[0]}/{len(jobs)}]", flush=True)

    # Preflight on one job: a bad credential should cost one request, not the
    # whole sweep.
    work(jobs[0])
    probe = out / f"rep{jobs[0][1]}" / f"{Path(jobs[0][0]).stem}.json"
    if probe.exists():
        first = json.loads(probe.read_text())
        if "error" in first:
            sys.exit(f"first request failed, aborting before the remaining "
                     f"{len(jobs)-1}:\n  {first.get('error')}: "
                     f"{first.get('message','')}\n\n"
                     f"If this is authentication: export ANTHROPIC_API_KEY.")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, jobs[1:]))


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
    ap.add_argument("--sync", action="store_true",
                    help="live requests instead of the Batches API (2x the cost)")
    ap.add_argument("--workers", type=int, default=4, help="--sync only")
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--dry-run", action="store_true",
                    help="build one request, save it, make no API call")
    args = ap.parse_args()

    from schematic2netlist.config import load_config
    cfg = load_config(args.config)
    names = split_names(cfg, args.split, args.limit)
    out = Path(args.out_dir or f"results/vlm/claude_{args.variant}")
    out.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        stem = Path(names[0]).stem
        task = build_task(stem, args.variant, cfg)
        ext = "png" if task["media_type"].endswith("png") else "jpg"
        (out / f"dryrun_image.{ext}").write_bytes(task["image_bytes"])
        (out / "dryrun_prompt.txt").write_text(
            f"MODEL: {MODEL}\n\nSYSTEM:\n{task['system']}\n\n{'='*60}\n"
            f"USER:\n{task['text']}")
        mb = len(base64.b64encode(task["image_bytes"])) / 1e6
        n = len(names) * args.repeat
        print(f"variant {args.variant}: {len(names)} images x {args.repeat} "
              f"repeats = {n} requests, {MODEL}")
        print(f"  mode          : {'sync' if args.sync else 'batch (50% cheaper)'}")
        print(f"  first image   : {stem}, {task['n_detections']} detections "
              f"({task['n_components']} components)")
        print(f"  image payload : {mb:.2f} MB base64 -> "
              f"{mb*len(names):.0f} MB per batch (cap 256 MB)")
        print(f"  prompt chars  : {len(task['text'])}")
        print(f"  wrote {out}/dryrun_image.{ext} and dryrun_prompt.txt")
        print("\nNo API call made.")
        return

    try:
        import anthropic
    except ImportError:
        sys.exit("pip install anthropic")
    client = anthropic.Anthropic()

    if args.sync:
        run_sync(client, names, args, cfg, out)
    else:
        run_batch(client, names, args, cfg, out)

    errs = [p for p in out.rglob("rep*/*.json")
            if "error" in json.loads(p.read_text())]
    print(f"\nwrote {out}  ({len(errs)} errored)")
    print(f"score with: python scripts/score_vlm.py --run-dir {out} "
          f"--variant {args.variant}")


if __name__ == "__main__":
    main()
