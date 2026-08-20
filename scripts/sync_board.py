#!/usr/bin/env python3
"""Task board sync: ingest delivered human artifacts, unblock what they unblock.

Run at the start of every working session. It does three things:

1. Ingests anything new in ``data/cghd/annotations/incoming/``: validates each
   file against the annotation schema, moves valid files to ``accepted/`` and
   invalid ones to ``rejected/`` with a specific, actionable message.
2. Re-evaluates every task's prerequisite. A task whose prerequisite artifact
   now exists flips BLOCKED -> READY.
3. Prints the board.

The point of this script is that human work never blocks machine work. A task
with a missing prerequisite is recorded as BLOCKED with the exact path that is
missing, and the next READY task is selected instead.

Usage:
    python scripts/sync_board.py            # ingest, re-evaluate, print board
    python scripts/sync_board.py --report   # add the missing-path report
    python scripts/sync_board.py --json     # machine-readable board
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOARD = ROOT / "TASK_BOARD.md"
INBOX = ROOT / "data/cghd/annotations/incoming"
ACCEPTED = ROOT / "data/cghd/annotations/accepted"
REJECTED = ROOT / "data/cghd/annotations/rejected"

# (id, title, prerequisite_path_or_None, human_gate)
#   prerequisite: a path that must exist before the task can start
#   human_gate:   "review" if the output needs author approval before use
TASKS: list[tuple[str, str, str | None, str | None]] = [
    ("A0", "Board and sync infrastructure", None, None),
    ("A1", "Repository and artifact inventory", None, None),
    ("A2", "Reproduce and freeze", None, None),
    ("A3", "Statistics utilities", None, None),
    ("A4", "Bibliography and template scrub", None, "review"),

    ("B1", "Acquire and inventory CGHD", None, None),
    ("B2", "Class taxonomy mapping", None, "review"),
    ("B3", "Format adapter", None, None),
    ("B4", "Imaging characterization of both corpora", None, None),
    ("B5", "Detection transfer, scored immediately", None, None),
    ("B6", "Netlist prediction over the evaluable pool", None, None),
    ("B7", "Capture invariance experiment", None, None),
    ("B8", "Annotation sampling design", None, "review"),

    ("C1", "Annotation tool", None, None),
    ("C2", "Schema validator and ingest", None, None),
    ("C3", "Annotation dashboard", None, None),

    ("D1", "Pin symmetry template (author authors the real file)", None, "review"),
    ("D2", "Pin aware scorer", "spec/pin_symmetry.yaml", None),
    ("D3", "Perturbation sensitivity and tolerance definitions", None, "review"),
    ("D4", "Multistability control", None, None),
    ("D5", "Multi condition agreement", None, None),
    ("D6", "Full structural recomputation", "metrics/pin_aware.py", None),
    ("D7", "Repair separation", None, None),
    # The prerequisite used to name a CGHD path, which was wrong: the repair
    # ledger exists for the DIGITIZE-HCD test split, and the second annotator's
    # blind packet is 58 of those same circuits carrying 249 declared repairs.
    # Their intervention records are the independent reference D8 needs, and
    # they arrive with the blind pass rather than with a CGHD campaign.
    ("D8", "Repair intent evaluation",
     "data/blind_review/gt_b/decisions", None),

    ("E1", "Incremental scoring harness", None, None),
    ("E2", "Cross corpus transfer results",
     "data/cghd/annotations/accepted/*.json", None),
    ("E3", "Drafter generalization analysis",
     "data/cghd/annotations/accepted/*.json", None),
    ("E4", "Inter annotator agreement",
     "data/cghd/annotations/double/*.json", None),

    ("F1", "Table and figure regeneration", None, None),
    ("F2", "Qualitative figure", "spec/qualitative_circuit.txt", None),
    ("F3", "Consistency pass", None, None),
    ("F4", "Reproducibility packet", None, None),
    ("F5", "Section drafts", None, "review"),
]

# States the board may record. AWAITING_REVIEW means the artifact exists but a
# downstream consumer must not use it until the author approves.
STATES = ("READY", "IN_PROGRESS", "BLOCKED", "AWAITING_REVIEW", "DONE")


def prereq_met(prereq: str | None) -> bool:
    if prereq is None:
        return True
    if "*" in prereq:
        parent = ROOT / Path(prereq).parent
        return parent.is_dir() and any(parent.glob(Path(prereq).name))
    return (ROOT / prereq).exists()


# --------------------------------------------------------------- ingest


def validate_annotation(path: Path) -> list[str]:
    """Return a list of problems; empty means the file is acceptable.

    Deliberately strict about structure and deliberately silent about
    plausibility: this flags, it never auto-corrects. An annotation that
    disagrees with the detector is a finding, not an error.
    """
    problems: list[str] = []
    try:
        d = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return [f"not valid JSON: {e}"]

    for key in ("schema_version", "image", "components", "sites"):
        if key not in d:
            problems.append(f"missing required top-level key '{key}'")
    if problems:
        return problems

    seen_terminals: set[tuple] = set()
    for i, c in enumerate(d["components"]):
        for k in ("id", "class", "terminals"):
            if k not in c:
                problems.append(f"component[{i}] missing '{k}'")
        if "terminals" not in c:
            continue
        for j, t in enumerate(c["terminals"]):
            if not isinstance(t, dict) or "net" not in t:
                problems.append(f"component[{i}].terminals[{j}] has no 'net'")
                continue
            if t["net"] is None:
                problems.append(
                    f"component[{i}].terminals[{j}] net is null — every "
                    f"terminal must be assigned to exactly one net")
            key = (c.get("id"), j)
            if key in seen_terminals:
                problems.append(f"component[{i}] terminal {j} declared twice")
            seen_terminals.add(key)

        nets = [t.get("net") for t in c["terminals"] if isinstance(t, dict)]
        if len(nets) > 1 and len(set(nets)) == 1 and c.get("class") not in (
                "GND", "V-DC (one port)"):
            problems.append(
                f"component[{i}] ({c.get('class')}) has every terminal on net "
                f"'{nets[0]}' — a component shorted through its own body. If "
                f"the drawing really shows this, set "
                f"\"allow_self_short\": true on that component.")
            if c.get("allow_self_short"):
                problems.pop()

    for s in d.get("sites", []):
        if s.get("kind") not in ("junction", "crossing", "edge_group", "none"):
            problems.append(
                f"site {s.get('id')} has kind {s.get('kind')!r}; must be one of "
                f"junction / crossing / edge_group / none")
    return problems


def ingest() -> dict:
    for p in (INBOX, ACCEPTED, REJECTED):
        p.mkdir(parents=True, exist_ok=True)
    accepted, rejected = [], []
    for f in sorted(INBOX.glob("*.json")):
        problems = validate_annotation(f)
        if problems:
            shutil.move(str(f), REJECTED / f.name)
            (REJECTED / f"{f.stem}.errors.txt").write_text(
                "\n".join(f"- {p}" for p in problems) + "\n")
            rejected.append((f.name, problems))
        else:
            shutil.move(str(f), ACCEPTED / f.name)
            accepted.append(f.name)
    return {"accepted": accepted, "rejected": rejected,
            "n_accepted_total": len(list(ACCEPTED.glob("*.json")))}



# --------------------------------------------------------------- dashboard


def annotation_dashboard(ing: dict) -> Path:
    """reports/annotation_progress.md (task C3).

    Refreshed on every sync. Answers the questions the author actually needs
    while annotating: how far in, is the sample still balanced, how consistent
    am I with myself, and when will this be done.
    """
    import statistics
    accepted = sorted(ACCEPTED.glob("*.json"))
    recs = []
    for f in accepted:
        try:
            recs.append(json.loads(f.read_text()))
        except json.JSONDecodeError:
            continue

    q_path = ROOT / "data/cghd/annotation_queue.json"
    design = json.loads(q_path.read_text()) if q_path.exists() else {}
    target = design.get("queue_length", 0)
    designed_drafters = len({q["drafter"] for q in design.get("queue", [])}) \
        if design.get("queue") else 0

    drafters = {r.get("drafter") for r in recs if r.get("drafter") is not None}
    times = [r["annotation_seconds"] for r in recs
             if isinstance(r.get("annotation_seconds"), (int, float))
             and r["annotation_seconds"] > 0]
    med = statistics.median(times) if times else None

    # self agreement on double-annotated drawings (pass 2 present)
    by_group: dict[str, list] = {}
    for r in recs:
        by_group.setdefault(r.get("drawing_group"), []).append(r)
    doubles = {g: v for g, v in by_group.items() if len(v) >= 2}
    kappa_txt = "not yet — no drawing annotated twice"
    if doubles:
        try:
            sys.path.insert(0, str(ROOT))
            from stats.kappa import cohens_kappa
            a, b = [], []
            for g, v in doubles.items():
                s1 = {s["id"]: s["kind"] for s in v[0].get("sites", [])}
                s2 = {s["id"]: s["kind"] for s in v[1].get("sites", [])}
                for k in sorted(set(s1) & set(s2)):
                    a.append(s1[k]); b.append(s2[k])
            if a:
                kr = cohens_kappa(a, b)
                kappa_txt = (f"kappa = {kr.kappa:.3f} ({kr.interpret()}) over "
                             f"{kr.n} intersection sites in {len(doubles)} "
                             f"re-annotated drawings")
        except Exception as e:                                # noqa: BLE001
            kappa_txt = f"could not compute: {type(e).__name__}"

    eta = "unknown — need at least 2 timed circuits"
    if med and target and len(recs) < target:
        remaining = (target - len(recs)) * med / 3600.0
        eta = (f"{remaining:.1f} hours of annotation remain at the current "
               f"median of {med/60:.1f} min per circuit")

    lines = [
        "# Annotation progress",
        "",
        "Regenerated by `scripts/sync_board.py` on every run. "
        "Do not edit by hand.",
        "",
        f"- **completed** {len(recs)} of {target} drawings"
        + (f" ({len(recs)/target:.0%})" if target else ""),
        f"- **rejected on ingest** {len(list(REJECTED.glob('*.json')))}"
        " (see the .errors.txt beside each)",
        f"- **awaiting ingest** {len(list(INBOX.glob('*.json')))}",
        f"- **distinct drafters covered** {len(drafters)}"
        + (f" of {designed_drafters} designed" if designed_drafters else ""),
        f"- **median time per circuit** "
        + (f"{med/60:.1f} min" if med else "n/a"),
        f"- **self agreement** {kappa_txt}",
        f"- **projection** {eta}",
        "",
        "## Stratification achieved vs designed",
        "",
        "| | achieved | designed |",
        "| --- | --- | --- |",
        f"| drawings | {len(recs)} | {target} |",
        f"| drafters | {len(drafters)} | {designed_drafters} |",
        "",
        "The queue is ordered so every prefix is stratified, so a partial "
        "column here is still a usable sample rather than a biased one.",
    ]
    if not recs:
        lines += ["", "_No annotations yet. The tool writes to "
                  "`data/cghd/annotations/incoming/`; this file fills in as "
                  "they arrive._"]
    dst = ROOT / "reports/annotation_progress.md"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + "\n")
    return dst


# --------------------------------------------------------------- board


def read_states() -> dict[str, tuple[str, str]]:
    """Parse the existing board so manual state edits survive a sync."""
    states: dict[str, tuple[str, str]] = {}
    if not BOARD.exists():
        return states
    for line in BOARD.read_text().splitlines():
        if not line.startswith("| ") or line.startswith("| ID"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 5 and cells[0] and cells[0][0] in "ABCDEF":
            states[cells[0]] = (cells[2], cells[4])
    return states


def build_board(ing: dict) -> list[dict]:
    prior = read_states()
    board = []
    for tid, title, prereq, gate in TASKS:
        old_state, commit = prior.get(tid, ("", ""))
        met = prereq_met(prereq)
        if old_state == "DONE":
            state = "DONE"
        elif old_state == "AWAITING_REVIEW":
            state = "AWAITING_REVIEW"
        elif not met:
            state = "BLOCKED"
        elif old_state in ("IN_PROGRESS",):
            state = old_state
        else:
            state = "READY"
        board.append({"id": tid, "title": title, "state": state,
                      "prereq": prereq or "", "met": met,
                      "gate": gate or "", "commit": commit})
    return board


def write_board(board: list[dict], ing: dict) -> None:
    lines = [
        "# Tier 1 task board",
        "",
        "Generated by `scripts/sync_board.py`. Run it at the start of every",
        "session. A BLOCKED row names the exact artifact that is missing; the",
        "correct response is to work the next READY task, never to wait.",
        "",
        f"Annotations accepted so far: **{ing['n_accepted_total']}**",
        "",
        "| ID | Title | State | Prerequisite (if blocked) | Commit |",
        "| --- | --- | --- | --- | --- |",
    ]
    for t in board:
        pre = "" if t["met"] else f"`{t['prereq']}`"
        gate = " *(needs review)*" if t["gate"] else ""
        lines.append(
            f"| {t['id']} | {t['title']}{gate} | {t['state']} | {pre} | "
            f"{t['commit']} |")
    counts: dict[str, int] = {}
    for t in board:
        counts[t["state"]] = counts.get(t["state"], 0) + 1
    lines += ["", "## Counts", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(counts.items())]
    BOARD.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true",
                    help="print the exact missing path for every blocked task")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--set", metavar="ID=STATE[:COMMIT]", action="append",
                    default=[],
                    help="record a state change, e.g. --set A0=DONE:abc1234")
    a = ap.parse_args()

    ing = ingest()
    annotation_dashboard(ing)
    board = build_board(ing)

    for spec in a.set:
        tid, _, rest = spec.partition("=")
        state, _, commit = rest.partition(":")
        if state not in STATES:
            sys.exit(f"unknown state {state!r}; must be one of {STATES}")
        for t in board:
            if t["id"] == tid.strip():
                t["state"] = state
                if commit:
                    t["commit"] = commit
                break
        else:
            sys.exit(f"unknown task id {tid!r}")

    write_board(board, ing)

    if a.json:
        print(json.dumps({"ingest": ing, "board": board}, indent=1))
        return 0

    if ing["accepted"] or ing["rejected"]:
        print(f"ingest: {len(ing['accepted'])} accepted, "
              f"{len(ing['rejected'])} rejected")
        for name, probs in ing["rejected"]:
            print(f"  REJECTED {name}: {probs[0]}")
    print(f"annotations accepted to date: {ing['n_accepted_total']}\n")

    for t in board:
        mark = {"READY": " ", "DONE": "x", "BLOCKED": "!",
                "AWAITING_REVIEW": "~", "IN_PROGRESS": ">"}[t["state"]]
        print(f"  [{mark}] {t['id']:3s} {t['title'][:52]:52s} {t['state']}")

    if a.report:
        print("\nblocked tasks and the exact path each is waiting on:")
        blocked = [t for t in board if t["state"] == "BLOCKED"]
        if not blocked:
            print("  (none)")
        for t in blocked:
            print(f"  {t['id']}: {t['prereq']}")

    ready = [t["id"] for t in board if t["state"] == "READY"]
    print(f"\n{len(ready)} READY: {', '.join(ready) if ready else '(none)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
