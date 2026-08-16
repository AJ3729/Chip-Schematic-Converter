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
    python tools/annotator/server.py --blind    # the 58-circuit blind packet

``--blind`` serves the independent second annotation of the Digitize-HCD test
split (``results/blind_review/packet/``). It is the same tool with two things
locked down, because that packet measures whether an unaided reader reaches the
same topology:

  * the images come from the packet's own ``frames_1024/``, so coordinates are
    in the frame ``scripts/compare_annotations.py`` expects, and
  * ``/api/truth`` stays refused, exactly as outside tutorial mode.

Output goes to a separate outbox so a blind pass can never be mixed into the
CGHD campaign. Convert it for scoring with ``scripts/annotator_to_gt.py``.
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

BLIND_PACKET = ROOT / "results/blind_review/packet"
BLIND_IMG = BLIND_PACKET / "frames_1024"
BLIND_OUTBOX = ROOT / "data/blind_review/incoming"
BLIND_DRAFTS = ROOT / "data/blind_review/drafts"

# Tutorial circuits: Digitize-HCD, ground truth already known, so the annotator
# can calibrate against a right answer before touching CGHD.
TUTORIAL = ["circuit_1013", "circuit_1022", "circuit_1025"]


def load_queue() -> list[dict]:
    if not QUEUE.exists():
        return []
    return json.loads(QUEUE.read_text())["queue"]


def blind_stems() -> list[str]:
    f = BLIND_PACKET / "circuits.txt"
    if not f.exists():
        return []
    return [s.strip() for s in f.read_text().splitlines() if s.strip()]


def work_items(tutorial: bool, blind: bool = False) -> list[dict]:
    if blind:
        # circuits.txt is shuffled on purpose -- the packet withholds which
        # stratum a circuit came from, so the order carries no signal either.
        return [{"rank": i + 1, "id": s, "image": s, "corpus": "blind-review",
                 "tutorial": False, "drafter": None, "n_captures": 1,
                 "captures": [s]}
                for i, s in enumerate(blind_stems())]
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


IMG_BASE = {"digitize-hcd": HCD_IMG, "cghd": CGHD_IMG, "blind-review": BLIND_IMG}


def image_path(item: dict) -> Path:
    return IMG_BASE[item["corpus"]] / f"{item['image']}.jpg"


def status(items: list[dict], outbox: Path, drafts: Path) -> dict:
    outbox.mkdir(parents=True, exist_ok=True)
    drafts.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in outbox.glob("*.json")}
    accepted = outbox.parent / "accepted"
    if accepted.is_dir():
        done |= {p.stem for p in accepted.glob("*.json")}
    drafted = {p.stem for p in drafts.glob("*.json")}
    return {"done": sorted(done), "drafts": sorted(drafted),
            "total": len(items)}


class Handler(http.server.SimpleHTTPRequestHandler):
    items: list[dict] = []
    tutorial: bool = False
    blind: bool = False
    outbox: Path = OUTBOX
    drafts: Path = DRAFTS

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
                               "blind": self.blind,
                               "status": status(self.items, self.outbox,
                                                self.drafts)})
        if p == "/api/draft":
            f = self.drafts / f"{q.get('id',[''])[0]}.json"
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
            f = IMG_BASE[item["corpus"]] / f"{stem}.jpg"
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
            self.drafts.mkdir(parents=True, exist_ok=True)
            (self.drafts / f"{body['id']}.json").write_text(json.dumps(body, indent=1))
            return self._send({"ok": True})

        if u.path == "/api/submit":
            self.outbox.mkdir(parents=True, exist_ok=True)
            rec = body.get("record") or {}
            stem = body["id"]
            (self.outbox / f"{stem}.json").write_text(json.dumps(rec, indent=1) + "\n")
            (self.drafts / f"{stem}.json").unlink(missing_ok=True)
            return self._send({"ok": True, "written": str(
                (self.outbox / f"{stem}.json").relative_to(ROOT))})

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
    ap.add_argument("--blind", action="store_true",
                    help="the 58-circuit independent second annotation packet")
    a = ap.parse_args()

    if a.blind and a.tutorial:
        sys.exit("--blind and --tutorial are different jobs; pick one")

    items = work_items(a.tutorial, a.blind)
    if not items:
        sys.exit("no work items: run scripts/make_blind_packet.py" if a.blind
                 else "no work items: run scripts/annotation_sampling_design.py")

    missing = [i["id"] for i in items if not image_path(i).exists()]
    if missing:
        print(f"WARNING: {len(missing)} items have no image on disk, "
              f"e.g. {missing[:3]}")
        if a.blind:
            sys.exit("the blind packet is incomplete; rebuild it with "
                     "scripts/make_blind_packet.py before annotating")

    Handler.items = items
    Handler.tutorial = a.tutorial
    Handler.blind = a.blind
    Handler.outbox = BLIND_OUTBOX if a.blind else OUTBOX
    Handler.drafts = BLIND_DRAFTS if a.blind else DRAFTS
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", a.port), Handler) as httpd:
        mode = ("BLIND second annotation (Digitize-HCD test split)" if a.blind
                else "TUTORIAL (Digitize-HCD, answers known)" if a.tutorial
                else "CGHD")
        print(f"annotation tool [{mode}]  {len(items)} items")
        print(f"  http://127.0.0.1:{a.port}")
        print(f"  submissions -> {Handler.outbox.relative_to(ROOT)}")
        if a.blind:
            print("  convert for scoring -> scripts/annotator_to_gt.py")
        print("  Ctrl-C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
