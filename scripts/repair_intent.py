#!/usr/bin/env python3
"""Do the pipeline's declared repairs recover what a reader would have done? (D8)

The repair pass makes a netlist simulatable by adding grounds and tying floating
nodes, and it declares every intervention. Declaring them proves they are
visible; it does not prove they are RIGHT. The only reference that can settle
that is a person who read the same drawing and wrote down what they would have
changed -- which is what the second annotator records in the Interventions box.

WHAT IS COMPARABLE, AND WHAT IS NOT. Of the six issues the pipeline declares,
only some have a human counterpart at all, and scoring the rest would produce a
number that is wrong in a flattering-looking direction:

  COMPARABLE      no_dc_path_to_ground. The circuit has no usable return and the
                  pipeline supplies one. A reader faced with the same drawing
                  either would or would not do the same thing, so precision and
                  recall mean something.

  NOT A REPAIR    placeholder_values fires on every circuit because no OCR is
                  performed; it is a property of the system, not a judgement
                  about a drawing, and the annotator is explicitly told not to
                  record values. Counted, never scored.

  GAUGE           ground_selection, ideal_inductor_dc, unset_current_direction,
                  unsnapped_terminal change no behaviour. No reader would call
                  them repairs. Scoring them would make the pipeline look like
                  it over-repairs ~2.5 times per circuit when nearly all of that
                  count is bookkeeping. Reported separately, never scored.

THE OTHER DIRECTION IS THE HALF NOBODY HAS MEASURED. The annotator's taxonomy
carries three labels the pipeline declares NOWHERE -- inferred_polarity,
missing_connection and as_drawn_short. Every one of those a reader records is by
construction a repair the pipeline did not make and does not know it did not
make. That is the under-repair measurement, and it cannot come from anywhere
else.

AN EMPTY INTERVENTION LIST IS AN ANSWER. The annotator is told so explicitly. A
delivered circuit with no interventions means "nothing needed repairing", and is
scored as such; a circuit with no delivered file is missing data and is excluded
and counted. Conflating the two would silently turn every unfinished circuit
into a human "no repair needed" and inflate the pipeline's false-positive rate.

Usage:
    python scripts/repair_intent.py
    python scripts/repair_intent.py --self-test
    python scripts/repair_intent.py --decisions <dir> --ledgers <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from stats.bootstrap import bootstrap_rate  # noqa: E402

DECISIONS = ROOT / "data/blind_review/gt_b/decisions"
LEDGERS = ROOT / "results/final/benchmark/seed0/ledgers"
OUT = ROOT / "results/repair_intent.json"

# Human labels that mean "this circuit has no usable return and I would supply
# one". All three map to the single pipeline issue that means the same thing.
HUMAN_GROUNDING = {"assumed_ground", "no_dc_path_to_ground", "floating_node"}
PIPELINE_GROUNDING = {"no_dc_path_to_ground"}

# Human labels the pipeline has no counterpart for. Each one recorded is a
# repair the pipeline did not make.
HUMAN_UNDETECTED = {"inferred_polarity", "missing_connection", "as_drawn_short"}

# Declared by the pipeline, never scored, for the reasons in the docstring.
PIPELINE_NOT_A_REPAIR = {"placeholder_values"}
PIPELINE_GAUGE = {"ground_selection", "ideal_inductor_dc",
                  "unset_current_direction", "unsnapped_terminal"}


def load_ledger(p: Path) -> list[dict]:
    try:
        return json.loads(p.read_text()).get("entries", []) or []
    except Exception:                                         # noqa: BLE001
        return []


def load_interventions(p: Path) -> list[dict] | None:
    """The annotator's records, or None when the circuit was never delivered."""
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text()).get("interventions", []) or []
    except Exception:                                         # noqa: BLE001
        return None


def norm(t: str | None) -> str:
    return (t or "").strip().lower().replace(" ", "_").replace("-", "_")


def compare(stem: str, human: list[dict], ledger: list[dict]) -> dict:
    h_types = [norm(i.get("type")) for i in human]
    p_types = [e["issue"] for e in ledger]

    h_ground = any(t in HUMAN_GROUNDING for t in h_types)
    p_ground = any(t in PIPELINE_GROUNDING for t in p_types)

    # Where both name a net, do they name the SAME one? A weaker agreement than
    # "both said grounding", and the only place location can be checked.
    h_nets = {norm(i.get("target")) for i in human
              if norm(i.get("type")) in HUMAN_GROUNDING and i.get("target")}
    p_nets = set()
    for e in ledger:
        if e["issue"] in PIPELINE_GROUNDING:
            p_nets |= {norm(n) for n in (e.get("location") or {}).get("nets", [])}
    loc = None
    if h_nets and p_nets:
        loc = bool(h_nets & p_nets)

    missed = [i for i in human if norm(i.get("type")) in HUMAN_UNDETECTED]
    other = [i for i in human
             if norm(i.get("type")) not in HUMAN_GROUNDING | HUMAN_UNDETECTED]

    return {
        "stem": stem,
        "human_says_grounding_repair": h_ground,
        "pipeline_declared_grounding_repair": p_ground,
        "agreement": ("both" if h_ground and p_ground else
                      "pipeline_only" if p_ground else
                      "human_only" if h_ground else "neither"),
        "location_agrees": loc,
        "human_nets": sorted(n for n in h_nets if n),
        "pipeline_nets": sorted(n for n in p_nets if n),
        "undetected_by_pipeline": [
            {"type": norm(i.get("type")), "target": i.get("target"),
             "note": (i.get("note") or "")[:200]} for i in missed],
        "unclassified_human_records": [
            {"type": norm(i.get("type")), "note": (i.get("note") or "")[:200]}
            for i in other],
        "pipeline_gauge_declared": sorted(
            t for t in set(p_types) if t in PIPELINE_GAUGE),
    }


def summarise(rows: list[dict], missing: list[str]) -> dict:
    n = len(rows)
    agree = Counter(r["agreement"] for r in rows)
    tp, fp, fn = agree["both"], agree["pipeline_only"], agree["human_only"]
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec else None

    out: dict = {
        "circuits_scored": n,
        "circuits_not_delivered": len(missing),
        "_denominator": ("only circuits the annotator delivered are scored; an "
                         "empty intervention list is 'nothing needed repairing' "
                         "and IS scored, an absent file is missing data and is "
                         "not"),
        "grounding_repair": {
            "_comparable_because": ("the pipeline's no_dc_path_to_ground and the "
                                    "reader's grounding labels mean the same "
                                    "thing about the same drawing"),
            "both": tp, "pipeline_only": fp, "human_only": fn,
            "neither": agree["neither"],
            "precision": prec, "recall": rec, "f1": f1,
            "_null_means_undefined": (
                "precision is null when the pipeline declared no grounding "
                "repair anywhere; recall is null when the reader recorded none. "
                "Neither is 0.0 -- there was nothing to be right or wrong "
                "about, and a zero would be read as a failure."),
        },
    }
    if n >= 5:
        for name, vals in (("precision", [r["agreement"] == "both" for r in rows
                                          if r["pipeline_declared_grounding_repair"]]),
                           ("recall", [r["agreement"] == "both" for r in rows
                                       if r["human_says_grounding_repair"]])):
            if vals:
                iv = bootstrap_rate(vals)
                out["grounding_repair"][f"{name}_ci95"] = [iv.lo, iv.hi]

    loc = [r["location_agrees"] for r in rows if r["location_agrees"] is not None]
    out["grounding_repair"]["location_checked"] = len(loc)
    out["grounding_repair"]["location_agrees"] = sum(1 for x in loc if x)

    under = [i for r in rows for i in r["undetected_by_pipeline"]]
    out["under_repair"] = {
        "_meaning": ("repairs the reader would make that the pipeline declares "
                     "nowhere. Each is a gap the pipeline cannot see in itself."),
        "n_records": len(under),
        "circuits_with_at_least_one": sum(
            1 for r in rows if r["undetected_by_pipeline"]),
        "rate": (sum(1 for r in rows if r["undetected_by_pipeline"]) / n
                 if n else None),
        "by_type": dict(Counter(i["type"] for i in under)),
        "examples": under[:20],
    }
    out["not_scored"] = {
        "_why": ("placeholder_values fires on every circuit because no OCR is "
                 "performed; the gauge issues change no behaviour. Neither has a "
                 "human counterpart, and scoring them would misreport the "
                 "pipeline as over-repairing when the count is bookkeeping."),
        "not_a_repair": sorted(PIPELINE_NOT_A_REPAIR),
        "gauge": sorted(PIPELINE_GAUGE),
    }
    out["unclassified_human_records"] = [
        {"stem": r["stem"], **i}
        for r in rows for i in r["unclassified_human_records"]][:30]
    return out


# ------------------------------------------------------------------ self-test

def self_test(ledgers: Path) -> int:
    """Score two synthetic annotators whose right answer is known.

    A PERFECT reader reports exactly the grounding repairs the pipeline
    declared, and must score precision and recall 1.0. A SILENT reader reports
    nothing, and must score recall 0 with every pipeline call a false positive.
    Neither needs real data, so the scoring path is validated before the
    annotator delivers -- which is the only time a fault in it is cheap.
    """
    stems = sorted(p.stem for p in ledgers.glob("*.json"))[:40]
    perfect, silent = [], []
    for s in stems:
        led = load_ledger(ledgers / f"{s}.json")
        pg = [e for e in led if e["issue"] in PIPELINE_GROUNDING]
        human = [{"type": "no_dc_path_to_ground",
                  "target": ((e.get("location") or {}).get("nets") or [None])[0],
                  "note": "synthetic"} for e in pg]
        perfect.append(compare(s, human, led))
        silent.append(compare(s, [], led))

    ok = True
    sp = summarise(perfect, [])["grounding_repair"]
    ss = summarise(silent, [])["grounding_repair"]
    p_ok = sp["precision"] == 1.0 and sp["recall"] == 1.0
    ok &= p_ok
    print(f"  perfect reader: precision={sp['precision']}, recall={sp['recall']}"
          f"  {'OK' if p_ok else 'FAIL'}")
    # Recall is UNDEFINED for a silent reader, not zero: with no human positives
    # there is nothing to recall, and reporting 0.0 would read as "the pipeline
    # missed everything" when the truth is "there was nothing to miss". Precision
    # is defined and must be 0.0 -- every pipeline call is unmatched.
    s_ok = (ss["recall"] is None and ss["precision"] == 0.0
            and ss["both"] == 0 and ss["human_only"] == 0
            and ss["pipeline_only"] > 0)
    ok &= s_ok
    print(f"  silent reader : precision={ss['precision']}, "
          f"recall={ss['recall']} (undefined, correctly), "
          f"pipeline_only={ss['pipeline_only']}  {'OK' if s_ok else 'FAIL'}")

    # an empty list must be scored, an absent file must not
    scored_empty = summarise([compare("x", [], [])], [])["circuits_scored"]
    e_ok = scored_empty == 1
    ok &= e_ok
    print(f"  empty intervention list is scored, not skipped  "
          f"{'OK' if e_ok else 'FAIL'}")
    assert load_interventions(ROOT / "does_not_exist.json") is None
    print("  absent file returns None (missing data, not 'no repair')  OK")

    u = summarise([compare("y", [{"type": "inferred_polarity", "note": "n"}], [])],
                  [])["under_repair"]
    u_ok = u["n_records"] == 1 and u["by_type"] == {"inferred_polarity": 1}
    ok &= u_ok
    print(f"  under-repair labels counted separately  {'OK' if u_ok else 'FAIL'}")

    print(f"\nself-test: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decisions", default=str(DECISIONS.relative_to(ROOT)))
    ap.add_argument("--ledgers", default=str(LEDGERS.relative_to(ROOT)))
    ap.add_argument("--out", default=str(OUT.relative_to(ROOT)))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    ledgers = ROOT / a.ledgers
    if a.self_test:
        return self_test(ledgers)

    dec = ROOT / a.decisions
    stems = sorted(p.stem for p in ledgers.glob("*.json"))
    rows, missing = [], []
    for s in stems:
        human = load_interventions(dec / f"{s}.json")
        if human is None:
            missing.append(s)
            continue
        rows.append(compare(s, human, load_ledger(ledgers / f"{s}.json")))

    report = {
        "_what": ("Whether the pipeline's declared repairs match what an "
                  "independent reader would have done (task D8)."),
        "decisions_dir": a.decisions,
        "ledgers_dir": a.ledgers,
        "summary": summarise(rows, missing) if rows else None,
        "per_circuit": rows,
        "not_delivered": missing,
    }
    out_p = ROOT / a.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=1) + "\n")

    if not rows:
        print(f"no annotator decisions in {a.decisions} yet -- harness verified, "
              f"wrote {a.out}")
        print("  run again once the second annotation is converted with "
              "scripts/annotator_to_gt.py")
        return 0

    s = report["summary"]
    g = s["grounding_repair"]
    print(f"scored {s['circuits_scored']} circuits "
          f"({s['circuits_not_delivered']} not delivered) -> {a.out}")
    print(f"  grounding repair: precision {g['precision']}, recall {g['recall']}"
          f"  (both {g['both']}, pipeline-only {g['pipeline_only']}, "
          f"human-only {g['human_only']})")
    print(f"  location agrees on {g['location_agrees']}/{g['location_checked']} "
          f"where both name a net")
    u = s["under_repair"]
    print(f"  repairs the pipeline never declares: {u['n_records']} across "
          f"{u['circuits_with_at_least_one']} circuits {u['by_type']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
