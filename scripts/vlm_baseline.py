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
PRICE_IN, PRICE_OUT = 5.0, 25.0     # USD per Mtok, claude-opus-5
BATCH_DISCOUNT = 0.5


def params_for(task: dict, effort: str = "low", thinking: bool = False) -> dict:
    """Messages-API parameters for one image. Shared by sync and batch so the
    two paths cannot drift apart.

    Thinking is OFF by default and that is the whole cost story. Structured
    outputs already force JSON-only, so the visible answer is ~151 tokens; at
    effort=high the model additionally spends ~21,000 INVISIBLE thinking
    tokens per image, billed at the output rate. That is 99.3% of the bill.
    Opus 5 accepts thinking:disabled only at effort high or lower, hence the
    low default here.
    """
    cfg = {"format": {"type": "json_schema", "schema": task["schema"]},
           "effort": effort}
    out = {
        "model": MODEL,
        # Generous: thinking tokens count against max_tokens, and 4096 clipped
        # 19 of 190 images mid-answer (stop_reason=max_tokens, unparseable
        # JSON). A cap costs nothing when the model finishes early.
        "max_tokens": 16000,
        "system": task["system"],
        "output_config": cfg,
        "thinking": {"type": "adaptive"} if thinking else {"type": "disabled"},
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {
                "type": "base64", "media_type": task["media_type"],
                "data": base64.b64encode(task["image_bytes"]).decode()}},
            {"type": "text", "text": task["text"]},
        ]}],
    }
    return out


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

# Measured on this task, claude-opus-5, variant B (mean output tokens/image).
# Thinking tokens are invisible but billed as output and dominate the bill.
MEASURED_OUT = {("adaptive", "low"): 2368, ("adaptive", "medium"): 8000,
                ("adaptive", "high"): 21239, ("disabled", "low"): 317}
MEASURED_IN = 2687


def project_cost(n_requests: int, thinking: bool, effort: str) -> tuple[float, int]:
    key = ("adaptive" if thinking else "disabled", effort)
    out_tok = MEASURED_OUT.get(key, MEASURED_OUT[("adaptive", "high")])
    per = (MEASURED_IN * PRICE_IN + out_tok * PRICE_OUT) / 1e6
    return per * n_requests * BATCH_DISCOUNT, out_tok


def record_provenance(out: Path, args, cfg, names: list) -> Path:
    """Everything needed to judge or repeat this run, written BEFORE submission.

    The previous run's reasoning settings were not recoverable from its
    artifacts -- `results/vlm/PROVENANCE.md` section 4 records them as MISSING,
    because nothing persisted the resolved request parameters and the observed
    token counts sat between two of the measured constants. One JSON write at
    submission time closes that permanently, so it happens here and it happens
    before a single request leaves the machine.
    """
    import hashlib
    import platform
    import subprocess

    prompts_path = ROOT / "configs/vlm_prompts.json"
    probe = build_task(Path(names[0]).stem, args.variant, cfg)
    params = params_for(probe, args.effort, args.thinking)

    def sha(b: bytes) -> str:
        return hashlib.sha256(b).hexdigest()

    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             timeout=5, cwd=ROOT).stdout.decode().strip()
    except Exception:
        git = "unknown"
    try:
        import anthropic
        sdk = anthropic.__version__
    except Exception:
        sdk = "unknown"

    rec = {
        "_what": "resolved request provenance for this run, written before "
                 "submission. Section 4 of PROVENANCE.md was MISSING for the "
                 "earlier run; this file is the fix.",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": git,
        "model_requested": MODEL,
        "sdk": {"anthropic": sdk, "python": platform.python_version(),
                "platform": platform.platform()},
        "reasoning_settings": {
            "thinking": params["thinking"],
            "effort": args.effort,
            "max_tokens": params["max_tokens"],
            "temperature": "not set (provider default)",
            "seed": "not supported / not set -- runs are NOT deterministic",
        },
        "output_schema": params["output_config"]["format"]["schema"],
        "structured_output": params["output_config"]["format"]["type"],
        "prompts": {
            "file": str(prompts_path.relative_to(ROOT)),
            "sha256": sha(prompts_path.read_bytes()),
            "system": probe["system"],
            "user_template_rendered_for_first_image": probe["text"],
        },
        "run_shape": {
            "variant": args.variant,
            "split": args.split,
            "n_images": len(names),
            "n_repeats": args.repeat,
            "n_requests": len(names) * args.repeat,
            "mode": "sync" if args.sync else "batch",
            "batch_discount": BATCH_DISCOUNT if not args.sync else 1.0,
        },
        "inputs": {
            "images_dir": cfg["preprocess"]["images_dir"],
            "detections_dir": cfg["detect"]["cache_dir"],
            "_variant_b_note": (
                "variant B renders OUR detected boxes onto the frame, so the "
                "bytes sent depend on the detector. request_manifest_rep*.json "
                "hashes the exact bytes, which pins the detector state."
            ) if args.variant == "b" else "variant A sends the frame unmodified",
        },
        "invalid_output_handling": {
            "refusal": "recorded as {'error':'refusal', category}",
            "no_text": "recorded as {'error':'no_text', stop_reason}",
            "bad_json": "recorded as {'error':'bad_json', message}",
            "batch_error": "recorded as {'error': <result type>, message}",
            "cached_errors_are_not_done": "is_done() treats an error file as "
                                          "not done, so reruns retry it",
            "scoring": "score_vlm.py counts an unusable response as a failure "
                       "with an empty prediction, never drops it",
        },
        "pricing_used_for_projection": {
            "input_usd_per_mtok": PRICE_IN, "output_usd_per_mtok": PRICE_OUT,
            "note": "list rates committed in this file; the projection is an "
                    "ESTIMATE, not an invoice",
        },
    }
    p = out / "request_provenance.json"
    p.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"wrote {p}")
    return p


def run_batch(client, names, args, cfg, out: Path) -> None:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    # Price the run BEFORE submitting. A batch cannot be un-spent, and the
    # difference between effort levels here is 67x.
    pending = sum(
        1 for rep in range(args.repeat) for nm in names
        if not is_done(out / f"rep{rep}" / f"{Path(nm).stem}.json"))
    est, out_tok = project_cost(pending, args.thinking, args.effort)
    print(f"projected: {pending} requests x ~{out_tok:,} output tok "
          f"= ~${est:.2f} batched "
          f"(thinking={'on' if args.thinking else 'OFF'}, effort={args.effort})")
    if est > args.max_spend:
        sys.exit(f"ABORT: ~${est:.2f} exceeds --max-spend ${args.max_spend:.2f}. "
                 f"Nothing submitted.\n"
                 f"  Lower --repeat, drop --thinking, or raise --max-spend.")

    state_path = out / "batches.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    for rep in range(args.repeat):
        rep_dir = out / f"rep{rep}"
        rep_dir.mkdir(parents=True, exist_ok=True)
        todo = [nm for nm in names if not is_done(rep_dir / f"{Path(nm).stem}.json")]
        if not todo:
            print(f"rep{rep}: already complete")
            continue

        # State is a LIST of batch ids per repeat, not one id. With a single
        # id, a partially-failed batch could never be retried: the resume path
        # re-read the same finished batch, re-wrote the same errors, and no
        # retry batch was ever submitted. Observed exactly that on 19 of 190
        # images clipped by max_tokens.
        bids = state.get(str(rep)) or []
        if isinstance(bids, str):                # migrate the old one-id shape
            bids = [bids]
        known = set(bids)

        # Drain every batch already submitted for this repeat first; that may
        # satisfy `todo` without spending anything.
        for bid in bids:
            harvest(client, bid, rep_dir, args, label=f"rep{rep} {bid}")
        todo = [nm for nm in names
                if not is_done(rep_dir / f"{Path(nm).stem}.json")]
        if not todo:
            print(f"rep{rep}: complete after draining {len(bids)} batch(es)")
            continue

        # Build the tasks explicitly rather than inline, so the EXACT bytes
        # sent for each image can be hashed. For variant B those bytes carry
        # our detector's boxes, so this manifest is what pins which detector
        # state the anchor was measured against.
        import hashlib
        reqs, manifest = [], {}
        for nm in todo:
            stem = Path(nm).stem
            task = build_task(stem, args.variant, cfg)
            manifest[stem] = {
                "image_sha256": hashlib.sha256(task["image_bytes"]).hexdigest(),
                "media_type": task["media_type"],
                "bytes": len(task["image_bytes"]),
                "n_detections": task["n_detections"],
                "n_components": task["n_components"],
                "user_text_sha256": hashlib.sha256(
                    task["text"].encode()).hexdigest(),
            }
            reqs.append(Request(
                custom_id=stem,
                params=MessageCreateParamsNonStreaming(
                    **params_for(task, args.effort, args.thinking))))
        (out / f"request_manifest_rep{rep}.json").write_text(
            json.dumps({"rep": rep, "n": len(manifest),
                        "detections_dir": cfg["detect"]["cache_dir"],
                        "images_dir": cfg["preprocess"]["images_dir"],
                        "images": manifest}, indent=1) + "\n")

        batch = client.messages.batches.create(requests=reqs)
        bid = batch.id
        bids.append(bid)
        state[str(rep)] = bids
        state_path.write_text(json.dumps(state, indent=1))
        print(f"rep{rep}: submitted {len(reqs)} requests as {bid}"
              + (f" (retry #{len(bids)-1})" if known else ""))

        while True:
            b = client.messages.batches.retrieve(bid)
            if b.processing_status == "ended":
                break
            c = b.request_counts
            print(f"  rep{rep} {b.processing_status}: "
                  f"processing={c.processing} succeeded={c.succeeded} "
                  f"errored={c.errored}", flush=True)
            time.sleep(args.poll_seconds)
        harvest(client, bid, rep_dir, args, label=f"rep{rep} {bid}")


def harvest(client, bid: str, rep_dir: Path, args, label: str) -> None:
    """Write every result of one ended batch into the per-image cache."""
    b = client.messages.batches.retrieve(bid)
    if b.processing_status != "ended":
        return
    n_ok = n_err = 0
    for r in client.messages.batches.results(bid):
        # Results come back in ARBITRARY order — key on custom_id, never on
        # position.
        dst = rep_dir / f"{r.custom_id}.json"
        if r.result.type == "succeeded":
            res = parse_message(r.result.message)
        else:
            res = {"error": r.result.type,
                   "message": str(getattr(r.result, "error", ""))[:300]}
        n_ok += "error" not in res
        n_err += "error" in res
        # Never let an older batch's failure clobber a good result already
        # harvested from a retry.
        if "error" in res and is_done(dst):
            continue
        dst.write_text(json.dumps(res, indent=1))
    print(f"{label}: {n_ok} ok, {n_err} errored")


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
                    **params_for(build_task(stem, args.variant, cfg),
                                  args.effort, args.thinking)) as s:
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
    ap.add_argument("--split", default="val",
                    help="exploration/oracle-injection, so it reads val by "
                         "default; --split test only for a reported number")
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--repeat", type=int, default=3,
                    help="independent passes; a single pass is not a measurement")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--sync", action="store_true",
                    help="live requests instead of the Batches API (2x the cost)")
    ap.add_argument("--workers", type=int, default=4, help="--sync only")
    ap.add_argument("--thinking", action="store_true",
                    help="enable adaptive thinking. Costs ~9-85x more: "
                         "thinking tokens are invisible but billed as output "
                         "and are 99%% of the bill. Off by default.")
    ap.add_argument("--max-spend", type=float, default=10.0,
                    help="abort before submitting if the measured projection "
                         "exceeds this many USD")
    ap.add_argument("--effort", default="low",
                    choices=["low", "medium", "high", "xhigh", "max"],
                    help="thinking depth. 99%% of output tokens (and so ~97%% of "
                         "cost) is thinking, so this is the cost dial. Measure "
                         "before lowering it.")
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--dry-run", action="store_true",
                    help="build one request, save it, make no API call")
    args = ap.parse_args()

    from schematic2netlist.config import load_config
    cfg = load_config(args.config)
    names = split_names(cfg, args.split, args.limit, args.splits_dir)
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

    record_provenance(out, args, cfg, names)

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
