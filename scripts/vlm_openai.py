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


def body_for(task: dict, model: str, max_tokens: int) -> dict:
    """Chat Completions body for one image.

    Chat Completions rather than Responses because the Batch API supports it
    most broadly. ``max_completion_tokens`` rather than ``max_tokens``: current
    reasoning models reject the latter. No temperature is set — several
    reasoning models reject it, and the anchor wants the model's default
    behaviour anyway.
    """
    data_url = (f"data:{task['media_type']};base64,"
                f"{base64.b64encode(task['image_bytes']).decode()}")
    return {
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


def is_done(path: Path) -> bool:
    """A cached ERROR does not count as done, so reruns retry it."""
    return path.exists() and "error" not in json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["a", "b"], default="b")
    ap.add_argument("--model", default=None,
                    help="exact model id; see --list-models. Nothing is guessed.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--completion-window", default="24h")
    ap.add_argument("--jsonl-dir", default=None,
                    help="where to stage the upload payload (default: system "
                         "temp). Never results/ — that directory is tracked.")
    ap.add_argument("--keep-jsonl", action="store_true",
                    help="keep the local payload after upload (debugging)")
    ap.add_argument("--keep-remote", action="store_true",
                    help="do not delete the uploaded input file when done")
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
    names = split_names(cfg, args.split, args.limit)
    out = Path(args.out_dir or f"results/vlm/openai_{args.variant}")
    out.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        stem = Path(names[0]).stem
        task = build_task(stem, args.variant, cfg)
        body = body_for(task, args.model or "<MODEL>", args.max_tokens)
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

        entry = state.get(str(rep)) or {}
        bid, input_fid = entry.get("batch_id"), entry.get("input_file_id")
        if not bid:
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
                                         args.model, args.max_tokens)}) + "\n")
            size_mb = jl.stat().st_size / 1e6
            with jl.open("rb") as fh:
                up = client.files.create(file=fh, purpose="batch")
            batch = client.batches.create(
                input_file_id=up.id, endpoint="/v1/chat/completions",
                completion_window=args.completion_window)
            bid, input_fid = batch.id, up.id
            state[str(rep)] = {"batch_id": bid, "input_file_id": input_fid,
                               "model": args.model}
            state_path.write_text(json.dumps(state, indent=1))
            if args.keep_jsonl:
                print(f"rep{rep}: kept local payload at {jl}")
            else:
                jl.unlink(missing_ok=True)
            print(f"rep{rep}: submitted {len(todo)} requests as {bid} "
                  f"({size_mb:.0f} MB uploaded)")
        else:
            print(f"rep{rep}: resuming {bid}")

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
                (rep_dir / f"{rec['custom_id']}.json").write_text(
                    json.dumps(res, indent=1))
        print(f"rep{rep}: wrote {n_ok} ok, {n_err} errored")

    errs = [p for p in out.rglob("rep*/*.json")
            if "error" in json.loads(p.read_text())]
    print(f"\nwrote {out}  ({len(errs)} errored)")
    print(f"score with: python scripts/score_vlm.py --run-dir {out} "
          f"--variant {args.variant}")


if __name__ == "__main__":
    main()
