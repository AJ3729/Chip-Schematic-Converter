#!/usr/bin/env python3
"""Local annotation tool for CGHD netlists (task C1).

A keyboard-driven browser tool for tracing connectivity from scratch. Runs
entirely on localhost against the local filesystem; nothing leaves the machine.

THE RULE THIS TOOL ENFORCES STRUCTURALLY: it never displays, pre-fills, or
suggests anything derived from pipeline output. Not detections, not predicted
nets, not confidences. Circular evaluation is the failure that rule prevents,
and the manuscript already commits to it for Digitize-HCD, so the tool refuses
to load a predictions directory even if one is present.

What it records:
  * as-drawn topology -- components, terminals, and the net each terminal is on
  * intersection adjudications: junction / crossing / edge_group / none
  * interventions the annotator WOULD apply, in a separate field, never folded
    into topology
  * per-circuit annotation seconds, and which pass this is (for E4)

Usage:
    python tools/annotator/server.py            # http://127.0.0.1:8765
    python tools/annotator/server.py --tutorial # 3 Digitize-HCD circuits
                                                # whose answers are known
"""

from __future__ import annotations

import argparse
import http.server
import json
import socketserver
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"

QUEUE = ROOT / "data/cghd/annotation_queue.json"
CGHD_IMG = ROOT / "data/cghd_1024/images"
HCD_IMG = ROOT / "data/cleaned_1024"
HCD_GT = ROOT / "data/gt_test_1024"
OUTBOX = ROOT / "data/cghd/annotations/incoming"
DRAFTS = ROOT / "data/cghd/annotations/drafts"

# Tutorial circuits: Digitize-HCD, ground truth already known, so the annotator
# can calibrate against a right answer before touching CGHD.
TUTORIAL = ["circuit_1013", "circuit_1022", "circuit_1025"]


def load_queue() -> list[dict]:
    if not QUEUE.exists():
        return []
    return json.loads(QUEUE.read_text())["queue"]


def work_items(tutorial: bool) -> list[dict]:
    if tutorial:
        return [{"rank": i + 1, "id": s, "image": s, "corpus": "digitize-hcd",
                 "tutorial": True, "drafter": None, "n_captures": 1,
                 "captures": [s]}
                for i, s in enumerate(TUTORIAL)]
    out = []
    for q in load_queue():
        # Annotate the FIRST capture; the netlist applies to the drawing, and
        # B7's grouping carries it to the other three.
        caps = q["captures"]
        out.append({"rank": q["rank"], "id": q["drawing_group"],
                    "image": caps[0], "corpus": "cghd", "tutorial": False,
                    "drafter": q["drafter"], "n_captures": len(caps),
                    "captures": caps})
    return out


def image_path(item: dict) -> Path:
    base = HCD_IMG if item["corpus"] == "digitize-hcd" else CGHD_IMG
    return base / f"{item['image']}.jpg"


def status(items: list[dict]) -> dict:
    OUTBOX.mkdir(parents=True, exist_ok=True)
    DRAFTS.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in OUTBOX.glob("*.json")}
    done |= {p.stem for p in (ROOT / "data/cghd/annotations/accepted").glob("*.json")}
    drafted = {p.stem for p in DRAFTS.glob("*.json")}
    return {"done": sorted(done), "drafts": sorted(drafted),
            "total": len(items)}


class Handler(http.server.SimpleHTTPRequestHandler):
    items: list[dict] = []
    tutorial: bool = False

    def _send(self, obj, code=200, ctype="application/json"):
        body = (json.dumps(obj) if ctype == "application/json"
                else obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):                                        # noqa: N802
        u = urllib.parse.urlparse(self.path)
        p, q = u.path, urllib.parse.parse_qs(u.query)

        if p in ("/", "/index.html"):
            return self._send((STATIC / "index.html").read_text(),
                              ctype="text/html; charset=utf-8")
        if p == "/app.js":
            return self._send((STATIC / "app.js").read_text(),
                              ctype="application/javascript")
        if p == "/api/items":
            return self._send({"items": self.items, "tutorial": self.tutorial,
                               "status": status(self.items)})
        if p == "/api/draft":
            f = DRAFTS / f"{q.get('id',[''])[0]}.json"
            return self._send(json.loads(f.read_text()) if f.exists() else {})
        if p == "/api/truth":
            # tutorial only: the known answer, revealed on request AFTER the
            # annotator has committed their own
            if not self.tutorial:
                return self._send({"error": "not in tutorial mode"}, 403)
            f = HCD_GT / f"{q.get('id',[''])[0]}.json"
            return self._send(json.loads(f.read_text()) if f.exists() else {})
        if p.startswith("/img/"):
            stem = p[len("/img/"):]
            item = next((i for i in self.items if i["image"] == stem
                         or stem in i.get("captures", [])), None)
            if not item:
                return self._send({"error": "unknown image"}, 404)
            base = HCD_IMG if item["corpus"] == "digitize-hcd" else CGHD_IMG
            f = base / f"{stem}.jpg"
            if not f.exists():
                return self._send({"error": f"missing {f}"}, 404)
            data = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        return self._send({"error": "not found"}, 404)

    def do_POST(self):                                       # noqa: N802
        u = urllib.parse.urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")

        if u.path == "/api/draft":
            DRAFTS.mkdir(parents=True, exist_ok=True)
            (DRAFTS / f"{body['id']}.json").write_text(json.dumps(body, indent=1))
            return self._send({"ok": True})

        if u.path == "/api/submit":
            OUTBOX.mkdir(parents=True, exist_ok=True)
            rec = body.get("record") or {}
            stem = body["id"]
            (OUTBOX / f"{stem}.json").write_text(json.dumps(rec, indent=1) + "\n")
            (DRAFTS / f"{stem}.json").unlink(missing_ok=True)
            return self._send({"ok": True, "written": str(
                (OUTBOX / f"{stem}.json").relative_to(ROOT))})

        return self._send({"error": "not found"}, 404)

    def log_message(self, *a):       # quiet
        return


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--tutorial", action="store_true",
                    help="3 Digitize-HCD circuits with known answers")
    a = ap.parse_args()

    items = work_items(a.tutorial)
    if not items:
        sys.exit("no work items: run scripts/annotation_sampling_design.py")

    missing = [i["id"] for i in items if not image_path(i).exists()]
    if missing:
        print(f"WARNING: {len(missing)} items have no image on disk, "
              f"e.g. {missing[:3]}")

    Handler.items = items
    Handler.tutorial = a.tutorial
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", a.port), Handler) as httpd:
        mode = "TUTORIAL (Digitize-HCD, answers known)" if a.tutorial else "CGHD"
        print(f"annotation tool [{mode}]  {len(items)} items")
        print(f"  http://127.0.0.1:{a.port}")
        print(f"  submissions -> {OUTBOX.relative_to(ROOT)}")
        print("  Ctrl-C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
