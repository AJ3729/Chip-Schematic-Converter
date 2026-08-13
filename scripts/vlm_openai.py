#!/usr/bin/env python3
"""The external anchor, OpenAI side. Same task, same metrics, different lab.

``vlm_baseline.py`` runs the Anthropic model; this runs an OpenAI one. Both
import the task from ``vlm_task.py`` and build nothing of their own, so the two
models get a byte-identical prompt, image and output schema. Both write the
same cached response shape, so ``score_vlm.py`` scores them through the same
metric cascade with no per-provider branch.

Why a second lab is worth the money. The paper's headline is that the residual
connectivity error is information-limited, and that currently rests entirely on
OUR methods failing — a reviewer can always say we didn't try hard enough. One
frontier model failing too is corroboration. Two frontier models from different
labs, failing on the SAME images, is very hard to attribute to our
implementation, our prompt, or one vendor's blind spot.

Uses the Batch API (50% cheaper, 24h window), one batch per repeat, with the
batch id checkpointed so an interrupted poll resumes. Results are keyed by
custom_id because batch output is not in submission order.

Model IDs move faster than this file. Run --list-models to see what your key
can reach and pass --model explicitly; nothing is guessed for you.

Usage:
    python scripts/vlm_openai.py --list-models
    python scripts/vlm_openai.py --variant b --model <id> --dry-run
    python scripts/vlm_openai.py --variant b --model <id> --repeat 3
    python scripts/score_vlm.py --run-dir results/vlm/openai_b --variant b
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from vlm_task import build_task, split_names


def record_provenance(out: Path, args, cfg, names: list) -> Path:
    """Everything needed to judge or repeat this run, written BEFORE submission.

    The mirror of the same function in ``vlm_baseline.py``. The earlier runs of
    BOTH providers left their resolved reasoning settings unrecorded --
    ``results/vlm/PROVENANCE.md`` section 4 marks them MISSING and notes the
    OpenAI gap too (code's measured 1681 output tokens at effort=low against an
    observed mean of 1272). One JSON write at submission closes it.
    """
    import hashlib
    import platform
    import subprocess

    prompts_path = ROOT / "configs/vlm_prompts.json"
    probe = build_task(Path(names[0]).stem, args.variant, cfg)
    body = body_for(probe, args.model, args.max_tokens, args.effort)

    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             timeout=5, cwd=ROOT).stdout.decode().strip()
    except Exception:
        git = "unknown"
    try:
        import openai as _oai
        sdk = _oai.__version__
    except Exception:
        sdk = "unknown"

    manifest = {}
    for nm in names:
        stem = Path(nm).stem
        t = build_task(stem, args.variant, cfg)
        manifest[stem] = {
            "image_sha256": hashlib.sha256(t["image_bytes"]).hexdigest(),
            "media_type": t["media_type"], "bytes": len(t["image_bytes"]),
            "n_detections": t["n_detections"], "n_components": t["n_components"],
            "user_text_sha256": hashlib.sha256(t["text"].encode()).hexdigest(),
        }

    rec = {
        "_what": "resolved request provenance, written before submission. "
                 "Section 4 of PROVENANCE.md was MISSING for the earlier run.",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": git,
        "model_requested": args.model,
        "sdk": {"openai": sdk, "python": platform.python_version(),
                "platform": platform.platform()},
        "reasoning_settings": {
            "reasoning_effort": body.get("reasoning_effort",
                                         "omitted (provider default)"),
            "max_completion_tokens": body["max_completion_tokens"],
            "temperature": "not set -- several reasoning models reject it",
            "seed": "not set -- runs are NOT deterministic",
        },
        "output_schema": body["response_format"]["json_schema"]["schema"],
        "structured_output": "json_schema, strict=True",
        "prompts": {
            "file": str(prompts_path.relative_to(ROOT)),
            "sha256": hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
            "system": probe["system"],
            "user_template_rendered_for_first_image": probe["text"],
        },
        "run_shape": {
            "variant": args.variant, "split": args.split,
            "n_images": len(names), "n_repeats": args.repeat,
            "n_requests": len(names) * args.repeat,
            "mode": "batch", "batch_discount": 0.5,
            "completion_window": args.completion_window,
        },
        "inputs": {
            "images_dir": cfg["preprocess"]["images_dir"],
            "detections_dir": cfg["detect"]["cache_dir"],
            "_variant_b_note": (
                "variant B renders OUR detected boxes onto the frame, so the "
                "bytes sent depend on the detector; the hashes below pin it"
            ) if args.variant == "b" else "variant A sends the frame unmodified",
            "images": manifest,
        },
        "invalid_output_handling": {
            "non_success": "recorded as {'error': ...} in the per-image cache",
            "cached_errors_are_not_done": "is_done() treats an error file as "
                                          "not done, so reruns retry it",
            "scoring": "score_vlm.py counts an unusable response as a failure "
                       "with an empty prediction, never drops it",
        },
        "pricing_used_for_projection": {
            "input_usd_per_mtok": args.price_in,
            "output_usd_per_mtok": args.price_out,
            "note": "argparse defaults are a deliberately PESSIMISTIC flagship "
                    "rate; the projection is an ESTIMATE, not an invoice",
        },
    }
    p = out / "request_provenance.json"
    p.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"wrote {p} ({len(manifest)} image hashes)")
    return p


def body_for(task: dict, model: str, max_tokens: int,
             effort: str | None = None) -> dict:
    """Chat Completions body for one image.

    Chat Completions rather than Responses because the Batch API supports it
    most broadly. ``max_completion_tokens`` rather than ``max_tokens``: current
    reasoning models reject the latter. No temperature is set — several
    reasoning models reject it, and the anchor wants the model's default
    behaviour anyway.
    """
    data_url = (f"data:{task['media_type']};base64,"
                f"{base64.b64encode(task['image_bytes']).decode()}")
    body = {
        "model": model,
        "max_completion_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": task["system"]},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": task["text"]},
            ]},
        ],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "circuit_netlist", "strict": True,
            "schema": task["schema"]}},
    }
    if effort:
        body["reasoning_effort"] = effort
    return body


def parse_line(rec: dict) -> dict:
    """One batch output line -> the cached response shape score_vlm.py reads."""
    if rec.get("error"):
        return {"error": "batch_error", "message": str(rec["error"])[:300]}
    resp = rec.get("response") or {}
    if resp.get("status_code") != 200:
        return {"error": f"http_{resp.get('status_code')}",
                "message": str(resp.get("body"))[:300]}
    body = resp.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return {"error": "no_choices"}
    ch = choices[0]
    if ch.get("finish_reason") == "content_filter":
        return {"error": "content_filter"}
    text = (ch.get("message") or {}).get("content")
    if not text:
        # length-capped reasoning models return an empty content string
        return {"error": "empty_content",
                "finish_reason": ch.get("finish_reason")}
    try:
        out = json.loads(text)
    except json.JSONDecodeError as e:
        return {"error": "bad_json", "message": str(e)[:200]}
    u = body.get("usage") or {}
    out["_usage"] = {"input": u.get("prompt_tokens"),
                     "output": u.get("completion_tokens")}
    out["_model"] = body.get("model")
    return out


# Measured on this task, gpt-5.5, variant B, reasoning_effort=low.
MEASURED_IN, MEASURED_OUT = 1981, 1681


def is_done(path: Path) -> bool:
    """A cached ERROR does not count as done, so reruns retry it."""
    return path.exists() and "error" not in json.loads(path.read_text())


def drain(client, bid: str, rep_dir: Path, label: str) -> None:
    """Write every result of one ended batch into the per-image cache."""
    b = client.batches.retrieve(bid)
    if b.status not in ("completed", "failed", "expired", "cancelled"):
        return
    n_ok = n_err = 0
    for fid in (b.output_file_id, b.error_file_id):
        if not fid:
            continue
        for raw in client.files.content(fid).text.splitlines():
            if not raw.strip():
                continue
            rec = json.loads(raw)
            # Batch output is NOT in submission order — key on custom_id.
            res = parse_line(rec)
            n_ok += "error" not in res
            n_err += "error" in res
            dst = rep_dir / f"{rec['custom_id']}.json"
            # Never let an older batch's failure clobber a retry's good result.
            if "error" in res and is_done(dst):
                continue
            dst.write_text(json.dumps(res, indent=1))
    print(f"{label}: {n_ok} ok, {n_err} errored")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["a", "b"], default="b")
    ap.add_argument("--model", default=None,
                    help="exact model id; see --list-models. Nothing is guessed.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="val",
                    help="exploration/oracle-injection, so it reads val by "
                         "default; --split test only for a reported number")
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--effort", default=None,
                    choices=["minimal", "low", "medium", "high"],
                    help="reasoning_effort. 96%% of output tokens are reasoning, "
                         "so this is the cost dial.")
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--completion-window", default="24h")
    ap.add_argument("--jsonl-dir", default=None,
                    help="where to stage the upload payload (default: system "
                         "temp). Never results/ — that directory is tracked.")
    ap.add_argument("--keep-jsonl", action="store_true",
                    help="keep the local payload after upload (debugging)")
    ap.add_argument("--keep-remote", action="store_true",
                    help="do not delete the uploaded input file when done")
    ap.add_argument("--price-in", type=float, default=10.0,
                    help="USD per Mtok input. Default is a deliberately "
                         "pessimistic flagship rate — set your real one.")
    ap.add_argument("--price-out", type=float, default=30.0,
                    help="USD per Mtok output (reasoning tokens bill here)")
    ap.add_argument("--max-spend", type=float, default=10.0,
                    help="abort before submitting if the projection exceeds this")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    try:
        import openai
    except ImportError:
        sys.exit("pip install openai")

    if args.list_models:
        client = openai.OpenAI()
        ids = sorted(m.id for m in client.models.list())
        print(f"{len(ids)} models reachable with this key:\n")
        for i in ids:
            print(f"  {i}")
        print("\nPick a vision-capable flagship and pass it as --model.")
        return

    from schematic2netlist.config import load_config
    cfg = load_config(args.config)
    names = split_names(cfg, args.split, args.limit, args.splits_dir)
    out = Path(args.out_dir or f"results/vlm/openai_{args.variant}")
    out.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        stem = Path(names[0]).stem
        task = build_task(stem, args.variant, cfg)
        body = body_for(task, args.model or "<MODEL>", args.max_tokens, args.effort)
        ext = "png" if task["media_type"].endswith("png") else "jpg"
        (out / f"dryrun_image.{ext}").write_bytes(task["image_bytes"])
        (out / "dryrun_prompt.txt").write_text(
            f"MODEL: {body['model']}\n\nSYSTEM:\n{task['system']}\n\n"
            f"{'='*60}\nUSER:\n{task['text']}")
        line = json.dumps({"custom_id": stem, "method": "POST",
                           "url": "/v1/chat/completions", "body": body})
        mb = len(line) / 1e6
        print(f"variant {args.variant}: {len(names)} images x {args.repeat} "
              f"repeats = {len(names)*args.repeat} requests")
        print(f"  model         : {body['model']}")
        print(f"  first image   : {stem}, {task['n_detections']} detections "
              f"({task['n_components']} components)")
        print(f"  jsonl line    : {mb:.2f} MB -> {mb*len(names):.0f} MB per batch")
        print(f"  wrote {out}/dryrun_image.{ext} and dryrun_prompt.txt")
        print("\nNo API call made.")
        return

    if not args.model:
        sys.exit("--model is required. Run --list-models to see what your key "
                 "can reach; this script will not guess a model id for you.")

    client = openai.OpenAI()
    record_provenance(out, args, cfg, names)

    pending = sum(1 for rep in range(args.repeat) for nm in names
                  if not is_done(out / f"rep{rep}" / f"{Path(nm).stem}.json"))
    est = ((MEASURED_IN * args.price_in + MEASURED_OUT * args.price_out)
           / 1e6 * pending * 0.5)
    print(f"projected: {pending} requests x ~{MEASURED_OUT:,} output tok "
          f"= ~${est:.2f} batched at ${args.price_in}/${args.price_out} per Mtok")
    if est > args.max_spend:
        sys.exit(f"ABORT: ~${est:.2f} exceeds --max-spend ${args.max_spend:.2f}. "
                 f"Nothing submitted.")

    state_path = out / "batches.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    for rep in range(args.repeat):
        rep_dir = out / f"rep{rep}"
        rep_dir.mkdir(parents=True, exist_ok=True)
        todo = [nm for nm in names
                if not is_done(rep_dir / f"{Path(nm).stem}.json")]
        if not todo:
            print(f"rep{rep}: already complete")
            continue

        # A LIST of submissions per repeat, not one. With a single id a
        # partially-failed batch can never be retried — the resume path just
        # re-reads the same finished batch. Cost that lesson once already on
        # the Anthropic side (19 of 190 clipped by max_tokens).
        entries = state.get(str(rep)) or []
        if isinstance(entries, dict):                 # migrate old shape
            entries = [entries]
        for e in entries:
            drain(client, e["batch_id"], rep_dir, f"rep{rep} {e['batch_id']}")
        todo = [nm for nm in names
                if not is_done(rep_dir / f"{Path(nm).stem}.json")]
        if not todo:
            print(f"rep{rep}: complete after draining {len(entries)} batch(es)")
            continue
        input_fid = None
        if True:
            # The payload is base64 images — ~47 MB per repeat for variant B,
            # ~200 MB across both variants at 3 repeats. It must NOT land in
            # results/, which is a tracked directory: one `git add results/`
            # and that is in the history forever. It is also fully regenerable
            # from the committed prompts and code, so it goes to a temp dir and
            # is deleted once uploaded.
            jl = Path(args.jsonl_dir or tempfile.gettempdir()) / \
                f"vlm_{args.variant}_rep{rep}_{os.getpid()}.jsonl"
            with jl.open("w") as fh:
                for nm in todo:
                    stem = Path(nm).stem
                    fh.write(json.dumps({
                        "custom_id": stem, "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body_for(build_task(stem, args.variant, cfg),
                                         args.model, args.max_tokens,
                                         args.effort)}) + "\n")
            size_mb = jl.stat().st_size / 1e6
            with jl.open("rb") as fh:
                up = client.files.create(file=fh, purpose="batch")
            batch = client.batches.create(
                input_file_id=up.id, endpoint="/v1/chat/completions",
                completion_window=args.completion_window)
            bid, input_fid = batch.id, up.id
            entries.append({"batch_id": bid, "input_file_id": input_fid,
                            "model": args.model})
            state[str(rep)] = entries
            state_path.write_text(json.dumps(state, indent=1))
            if args.keep_jsonl:
                print(f"rep{rep}: kept local payload at {jl}")
            else:
                jl.unlink(missing_ok=True)
            print(f"rep{rep}: submitted {len(todo)} requests as {bid} "
                  f"({size_mb:.0f} MB uploaded)")

        while True:
            b = client.batches.retrieve(bid)
            if b.status in ("completed", "failed", "expired", "cancelled"):
                break
            c = b.request_counts
            print(f"  rep{rep} {b.status}: completed={c.completed} "
                  f"failed={c.failed} total={c.total}", flush=True)
            time.sleep(args.poll_seconds)

        if b.status != "completed":
            print(f"rep{rep}: batch ended {b.status} — {b.errors}")
        # The uploaded input is the big artifact (base64 images) and counts
        # against the account's file storage. It is regenerable, so drop it
        # once the batch has ended. The OUTPUT file is left alone — that is
        # the data, and it is small.
        if input_fid and not args.keep_remote:
            try:
                client.files.delete(input_fid)
                print(f"rep{rep}: deleted uploaded input file {input_fid}")
            except Exception as e:  # noqa: BLE001
                print(f"rep{rep}: could not delete {input_fid}: "
                      f"{type(e).__name__}")
        drain(client, bid, rep_dir, f"rep{rep} {bid}")

    errs = [p for p in out.rglob("rep*/*.json")
            if "error" in json.loads(p.read_text())]
    print(f"\nwrote {out}  ({len(errs)} errored)")
    print(f"score with: python scripts/score_vlm.py --run-dir {out} "
          f"--variant {args.variant}")


if __name__ == "__main__":
    main()
