#!/usr/bin/env python3
"""Every number in the manuscript, and the file it comes from (tasks F1 and F3).

The project rule is that no number in the paper is hand-typed. Enforcing it
needs two things that are really one thing: a machine-readable statement of
where each quantity lives, and a check that the prose agrees with it. Both are
built from the REGISTRY below, so they cannot drift apart -- a generator and a
checker maintained separately would eventually disagree, and the paper would
follow whichever one nobody was reading.

    --emit    write paper/generated/numbers.tex, one \\newcommand per quantity
    --check   read a .tex and verify every registered quantity appears in it
              with the value its source file actually holds

WHAT --check CATCHES, AND WHAT IT CANNOT

It catches the failure that matters: a number that was correct when it was typed
and is now stale because the run that produced it was redone. That is the whole
history of this project -- the detector retrain moved every downstream figure at
once -- and it is invisible to proofreading, because a stale number looks exactly
like a fresh one.

It cannot certify a number it does not know about. So it also reports every
\\num{} in the .tex that no registry entry explains, as a WORK LIST rather than
as an error: many are legitimately not results (split sizes, percentages, years).
The list shrinking over time is the actual measure of progress here, and it is
printed so it cannot be quietly ignored.

It also checks only that the correct value appears SOMEWHERE in the document,
not that it appears in the right cell. A number moved into the wrong row of the
right table would pass. That is a deliberate stopping point rather than an
oversight: locating each quantity would mean parsing the tables, and a parser
that silently mismatched rows would report false failures on a correct paper --
which is how a checker gets switched off. Row placement is what proofreading is
for; staleness is what proofreading cannot do.

Usage:
    python scripts/manuscript_numbers.py --emit
    python scripts/manuscript_numbers.py --check path/to/access.tex
    python scripts/manuscript_numbers.py --check path/to/access.tex --unregistered
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "paper" / "generated"

SEEDS = (0, 1, 2)


@dataclass
class Q:
    """One quantity: a macro name, where it comes from, and how it prints."""
    name: str
    source: str                     # path, may contain {seed}
    key: str                        # dotted path into the JSON
    fmt: str = "{:.4f}"
    over_seeds: str | None = None   # None | "mean" | "sd"
    scale: float = 1.0
    note: str = ""
    table: str = ""


def dotted(obj, key: str):
    cur = obj
    for part in key.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


def read_value(q: Q):
    if q.over_seeds:
        vals = []
        for s in SEEDS:
            p = ROOT / q.source.format(seed=s)
            vals.append(float(dotted(json.loads(p.read_text()), q.key)))
        return (statistics.mean(vals) if q.over_seeds == "mean"
                else statistics.stdev(vals))
    p = ROOT / q.source
    return float(dotted(json.loads(p.read_text()), q.key))


# ---------------------------------------------------------------------------
# THE REGISTRY
# ---------------------------------------------------------------------------
# Grouped by the table or paragraph each quantity appears in, so a table that
# gains a row gains a registry entry beside it rather than a hand-typed number.

BENCH = "results/final/benchmark/seed{seed}/summary.json"
BENCH0 = "results/final/benchmark/seed0/summary.json"
VAL = "results/final/benchmark_val/summary.json"
DET = "results/final/detection/seed{seed}/test/summary.json"
DET0 = "results/final/detection/seed0/test/summary.json"
PINS = "results/final/pin_order/summary.json"
LADDER = "results/pin_aware_ladder.json"
MULTI = "results/multistability.json"
MCOND = "results/multi_condition_agreement.json"
ROB = "results/robustness/SUMMARY.json"
DELIV = "results/robustness/delivered_lossless.json"
VLMSIG = "results/table5_significance.json"
VLMREP = "results/final/vlm_repeat_significance/summary.json"

REGISTRY: list[Q] = [
    # ---- tab:main, end-to-end reconstruction -----------------------------
    Q("MainTermPairF1", BENCH, "topology.terminal_pair_f1.mean",
      over_seeds="mean", table="tab:main"),
    Q("MainTermPairF1SD", BENCH, "topology.terminal_pair_f1.mean",
      over_seeds="sd", table="tab:main"),
    Q("MainNetFone", BENCH, "topology.net_f1.mean",
      over_seeds="mean", table="tab:main"),
    Q("MainNetFoneSD", BENCH, "topology.net_f1.mean",
      over_seeds="sd", table="tab:main"),
    Q("MainPerComp", BENCH, "topology.per_component_connected_acc.mean",
      over_seeds="mean", table="tab:main"),
    Q("MainPerCompSD", BENCH, "topology.per_component_connected_acc.mean",
      over_seeds="sd", table="tab:main"),
    Q("MainNGED", BENCH, "topology.nged.mean", over_seeds="mean",
      table="tab:main"),
    Q("MainNGEDSD", BENCH, "topology.nged.mean", over_seeds="sd",
      table="tab:main"),
    Q("MainStrict", BENCH, "topology.strict_success.mean",
      over_seeds="mean", table="tab:main"),
    Q("MainStrictSD", BENCH, "topology.strict_success.mean",
      over_seeds="sd", table="tab:main"),
    Q("MainSpiceValid", BENCH0, "repair.spice_valid_rate", table="tab:main"),
    Q("MainSolvableBefore", BENCH0, "repair.solvable_before_rate",
      table="tab:main"),
    Q("MainSolvableAfter", BENCH0, "repair.solvable_after_rate",
      table="tab:main"),
    Q("MainAssumptions", BENCH0, "repair.mean_assumptions", fmt="{:.2f}",
      table="tab:main"),

    # validation column, shown for reference only
    Q("ValTermPairF1", VAL, "topology.terminal_pair_f1.mean", table="tab:main"),
    Q("ValNetFone", VAL, "topology.net_f1.mean", table="tab:main"),
    Q("ValPerComp", VAL, "topology.per_component_connected_acc.mean",
      table="tab:main"),
    Q("ValNGED", VAL, "topology.nged.mean", table="tab:main"),
    Q("ValStrict", VAL, "topology.strict_success.mean", table="tab:main"),
    Q("ValSpiceValid", VAL, "repair.spice_valid_rate", table="tab:main"),
    Q("ValSolvableBefore", VAL, "repair.solvable_before_rate", table="tab:main"),
    Q("ValSolvableAfter", VAL, "repair.solvable_after_rate", table="tab:main"),
    Q("ValAssumptions", VAL, "repair.mean_assumptions", fmt="{:.2f}",
      table="tab:main"),

    # ---- tab:detector ----------------------------------------------------
    Q("DetMapMean", DET, "map50", over_seeds="mean", table="tab:detector"),
    Q("DetMapSD", DET, "map50", over_seeds="sd", table="tab:detector"),
    Q("DetMapNinetyFiveMean", DET, "map50_95", over_seeds="mean",
      table="tab:detector"),
    Q("DetPrecisionSeedZero", DET.format(seed=0), "precision",
      table="tab:detector"),
    Q("DetRecallSeedZero", DET.format(seed=0), "recall", table="tab:detector"),

    # ---- tab:pins --------------------------------------------------------
    Q("PinsDecidable", PINS, "templates_only.decidable", fmt="{:.0f}",
      table="tab:pins"),
    Q("PinsMatched", PINS, "templates_only.multi_terminal_matched",
      fmt="{:.0f}", table="tab:pins"),
    Q("PinsTemplateOverall", PINS, "templates_only.accuracy",
      table="tab:pins"),
    Q("PinsTemplateBJTNPN", PINS, "templates_only.by_class.BJT-NPN.accuracy",
      table="tab:pins"),
    Q("PinsTemplateBJTPNP", PINS, "templates_only.by_class.BJT-PNP.accuracy",
      table="tab:pins"),
    Q("PinsTemplateMOSN", PINS, "templates_only.by_class.MOSFET-N.accuracy",
      table="tab:pins"),
    Q("PinsTemplateMOSP", PINS, "templates_only.by_class.MOSFET-P.accuracy",
      table="tab:pins"),
    Q("PinsTemplateOpAmp", PINS, "templates_only.by_class.Op-Amp.accuracy",
      table="tab:pins"),

    # ---- tab:ladder, the paper's central result --------------------------
    Q("LadderPinBlindStrict", LADDER, "ladder.pin_blind_strict_success",
      table="tab:ladder"),
    Q("LadderPinAwareStrict", LADDER, "ladder.pin_aware_strict_success",
      table="tab:ladder"),
    Q("LadderBlindPerfectAlsoAware", LADDER,
      "ladder.of_pin_blind_perfect_also_pin_aware_perfect", table="tab:ladder"),
    Q("LadderAwarePerfectOpAgrees", LADDER,
      "ladder.of_pin_aware_perfect_op_agrees", table="tab:ladder"),
    Q("LadderComponentAcc", LADDER, "mean_component_accuracy",
      table="tab:ladder"),
    Q("LadderCircuits", LADDER, "n_circuits", fmt="{:.0f}", table="tab:ladder"),
    Q("LadderBlindOnly", LADDER, "mcnemar_blind_vs_aware.pin_blind_only",
      fmt="{:.0f}", table="tab:ladder"),
    Q("LadderAwareOnly", LADDER, "mcnemar_blind_vs_aware.pin_aware_only",
      fmt="{:.0f}", table="tab:ladder"),

    # ---- multistability control (D4) --------------------------------------
    Q("MultiTested", MULTI, "n_circuits_tested", fmt="{:.0f}"),
    Q("MultiFlagged", MULTI, "n_flagged_multistable_or_order_dependent",
      fmt="{:.0f}"),
    Q("MultiPerfect", MULTI, "headline_all_circuits.topologically_perfect",
      fmt="{:.0f}"),
    Q("MultiOpDisagrees", MULTI, "headline_all_circuits.of_those_op_disagrees",
      fmt="{:.0f}"),
    Q("MultiRate", MULTI, "headline_all_circuits.rate"),
    Q("MultiRateExFlagged", MULTI, "headline_excluding_flagged.rate"),

    # ---- D5, multi-condition agreement -----------------------------------
    Q("McondOpFone", MCOND, "summary.op_primary.mean_f1.mean"),
    Q("McondOpExact", MCOND, "summary.op_primary.exact_rate.mean"),
    Q("McondOpN", MCOND, "summary.op_primary.n_informative", fmt="{:.0f}"),
    Q("McondLowBiasFone", MCOND, "summary.op_low_bias.mean_f1.mean"),
    Q("McondLowBiasExact", MCOND, "summary.op_low_bias.exact_rate.mean"),
    Q("McondAcFone", MCOND, "summary.ac_1khz.mean_f1.mean"),
    Q("McondAcExact", MCOND, "summary.ac_1khz.exact_rate.mean"),
    Q("McondAcN", MCOND, "summary.ac_1khz.n_informative", fmt="{:.0f}"),
    Q("McondEither", MCOND, "summary._coverage.informative_under_either",
      fmt="{:.0f}"),
    Q("McondAddedByAc", MCOND, "summary._coverage.n_added_by_ac_alone",
      fmt="{:.0f}"),

    # ---- tab:vlm, the frontier-model anchor -------------------------------
    # These were unregistered until the Claude variant-B row was found to pair
    # a mean strict success with a delta and p computed from the majority vote
    # of the same three repeats -- two correct numbers describing different
    # systems, in a row that therefore did not subtract. Nothing caught it
    # because nothing was checking this table.
    Q("VlmClaudeAStrict", VLMSIG, "comparisons.claude_A.vlm_strict_success",
      table="tab:vlm"),
    Q("VlmGptAStrict", VLMSIG, "comparisons.gpt_A.vlm_strict_success",
      table="tab:vlm"),
    Q("VlmGptBStrict", VLMSIG, "comparisons.gpt_B.vlm_strict_success",
      table="tab:vlm"),
    Q("VlmGptBDelta", VLMSIG, "comparisons.gpt_B.pipeline_minus_vlm",
      table="tab:vlm"),
    Q("VlmClaudeADelta", VLMSIG, "comparisons.claude_A.pipeline_minus_vlm",
      table="tab:vlm"),
    Q("VlmGptADelta", VLMSIG, "comparisons.gpt_A.pipeline_minus_vlm",
      table="tab:vlm"),
    # variant B, Claude: single-query mean, and the delta that matches it
    Q("VlmClaudeBStrict", VLMREP, "single_query.mean_strict_success",
      table="tab:vlm"),
    Q("VlmClaudeBDelta", VLMREP, "single_query.delta_ours_minus_theirs",
      table="tab:vlm"),
    Q("VlmClaudeBPMin", VLMREP, "single_query.mcnemar_p_min", fmt="{:.2f}",
      table="tab:vlm"),
    Q("VlmClaudeBPMax", VLMREP, "single_query.mcnemar_p_max", fmt="{:.2f}",
      table="tab:vlm"),
    Q("VlmClaudeBRepLo", VLMREP, "per_repeat.rep2.vlm_strict_success",
      table="tab:vlm"),
    Q("VlmClaudeBRepMid", VLMREP, "per_repeat.rep1.vlm_strict_success",
      table="tab:vlm"),
    Q("VlmClaudeBRepHi", VLMREP, "per_repeat.rep0.vlm_strict_success",
      table="tab:vlm"),
    Q("VlmClaudeBCiLo", VLMREP, "single_query.ci_widest.0", table="tab:vlm"),
    Q("VlmClaudeBCiHi", VLMREP, "single_query.ci_widest.1", table="tab:vlm"),

    # ---- tab:robustness, simulated capture degradation --------------------
    # The control entry is the one that matters: if it ever stops equalling
    # the published 0.5312 the whole sweep is measuring its own plumbing, and
    # a checker that verifies every other row but not this one would not say so.
    Q("RobControl", ROB, "conditions.clean.strict_success", table="tab:robustness"),
    Q("RobContrast2", ROB, "conditions.contrast_s2.strict_success", table="tab:robustness"),
    Q("RobJpeg1", ROB, "conditions.jpeg_s1.strict_success", table="tab:robustness"),
    Q("RobDownscale3", ROB, "conditions.downscale_s3.strict_success", table="tab:robustness"),
    Q("RobBright3", ROB, "conditions.brightness_s3.strict_success", table="tab:robustness"),
    Q("RobBlur3", ROB, "conditions.blur_s3.strict_success", table="tab:robustness"),
    Q("RobRotate1", ROB, "conditions.rotate_s1.strict_success", table="tab:robustness"),
    Q("RobPersp1", ROB, "conditions.perspective_s1.strict_success", table="tab:robustness"),
    Q("RobPersp3", ROB, "conditions.perspective_s3.strict_success", table="tab:robustness"),
    Q("RobScaleCghd", "results/robustness/scale_f71/summary.json",
      "topology.strict_success.mean", table="tab:robustness"),
    Q("RobScaleTiny", "results/robustness/scale_f36/summary.json",
      "topology.strict_success.mean", table="tab:robustness"),

    # Delivered, not requested. The manuscript quotes these because clipping
    # against a 249.7-mean page and a JPEG re-encode together remove more than
    # half the requested sigma, and the requested figure would overstate three
    # conditions by more than twofold.
    Q("DelivGauss1", DELIV, "conditions.gauss_noise_s1_lossless.sigma_delivered", fmt="{:.1f}"),
    Q("DelivGauss2", DELIV, "conditions.gauss_noise_s2_lossless.sigma_delivered", fmt="{:.1f}"),
    Q("DelivGauss3", DELIV, "conditions.gauss_noise_s3_lossless.sigma_delivered", fmt="{:.1f}"),
    Q("DelivSpeckle1", "results/robustness/delivered_corruption.json",
      "conditions.speckle_s1.changed_frac_on_disk", fmt="{:.2f}", scale=100.0),
    Q("DelivSpeckle3", "results/robustness/delivered_corruption.json",
      "conditions.speckle_s3.changed_frac_on_disk", fmt="{:.2f}", scale=100.0),
    # re-seeded and lossless arms
    Q("RobCleanLossless", "results/robustness/clean_lossless/summary.json",
      "topology.strict_success.mean", table="tab:robustness"),
    Q("RobGaussL1", "results/robustness/gauss_noise_s1_lossless/summary.json",
      "topology.strict_success.mean", table="tab:robustness"),
    Q("RobGaussL3", "results/robustness/gauss_noise_s3_lossless/summary.json",
      "topology.strict_success.mean", table="tab:robustness"),
    Q("RobSpeckle1b", "results/robustness/speckle_s1/summary.json",
      "topology.strict_success.mean", table="tab:robustness"),
    Q("RobSpeckle3b", "results/robustness/speckle_s3/summary.json",
      "topology.strict_success.mean", table="tab:robustness"),
    Q("RobSpeckle1FixF1", "results/robustness/speckle_s1_fix/summary.json",
      "topology.terminal_pair_f1.mean"),
    Q("RobSpeckle1F1", "results/robustness/speckle_s1/summary.json",
      "topology.terminal_pair_f1.mean"),
]


def _ablation_deltas() -> dict[str, float]:
    """Per-arm change in strict success, computed rather than transcribed.

    The manuscript's first draft named the top-two contributors from the arms'
    internal directory names and got it wrong: `v5_plus_crossover_DEFAULT` adds
    port templates, not crossover, and crossover moves strict success by exactly
    zero. Deriving the deltas here means a sentence about which stage matters
    cannot disagree with the table beside it.
    """
    idx = json.loads((ROOT / "results/final/ablation/index.json").read_text())
    out, prev = {}, None
    for a in idx["arms"]["ablation"]:
        s = float(a["topology"]["strict_success"]["mean"])
        out[a["label"]] = 0.0 if prev is None else s - prev
        prev = s
    return out


class AblationQ(Q):
    """A quantity read from the ablation deltas rather than from a JSON key."""


# Validation-split ablation deltas (mentor fix 1d). The ablation is now
# reported on val; test is the comparison column, so both are registered.
class AblationValQ(Q):
    """A quantity read from the VALIDATION ablation deltas."""


def _ablation_val_deltas() -> dict[str, float]:
    idx = json.loads(
        (ROOT / "results/final/ablation_val/index.json").read_text())
    out, prev = {}, None
    for a in idx["arms"]["ablation"]:
        s = float(a["topology"]["strict_success"]["mean"])
        out[a["label"]] = 0.0 if prev is None else s - prev
        prev = s
    return out


for _label, _macro in (("v5_plus_crossover_DEFAULT", "AblValPortTemplates"),
                       ("v2_ink_boundary_snap", "AblValInkSnap"),
                       ("v6_plus_bridge_span7", "AblValBridgeSpan"),
                       ("v12_plus_head_ensemble", "AblValHeadEnsemble")):
    REGISTRY.append(AblationValQ(_macro,
                                 "results/final/ablation_val/index.json",
                                 f"__delta__.{_label}", table="fig:ablation"))

# The two residual circuits, resolved (task F2 precursor).
RESID = "results/residual_circuits.json"
REGISTRY.extend([
    Q("ResidUngrounded", RESID,
      "population_check.circuits_without_drawn_ground", fmt="{:.0f}",
      table="fig:qualitative"),
    Q("ResidUngroundedDisagree", RESID,
      "population_check.of_those_op_disagrees", fmt="{:.0f}",
      table="fig:qualitative"),
    Q("ResidGroundedN", RESID,
      "population_check.circuits_with_drawn_ground", fmt="{:.0f}",
      table="fig:qualitative"),
])

# Split-overlap audit (mentor fix 1c) and its sensitivity table.
SPLITDUP = "results/split_duplicate_audit.json"
REGISTRY.extend([
    Q("LeakTestShareVal", SPLITDUP,
      "topology.impact.test_sharing_topology_with_val", fmt="{:.0f}",
      table="tab:leakage"),
    Q("LeakTestImages", SPLITDUP, "topology.impact.test_images", fmt="{:.0f}",
      table="tab:leakage"),
    Q("LeakDistinctTopologies", SPLITDUP, "topology.distinct_topologies",
      fmt="{:.0f}", table="tab:leakage"),
    Q("LeakImagesWithGT", SPLITDUP, "topology.images_with_gt", fmt="{:.0f}",
      table="tab:leakage"),
    Q("LeakLargestGroup", SPLITDUP, "topology.impact.largest_group_size",
      fmt="{:.0f}", table="tab:leakage"),
])

for _label, _macro in (("v5_plus_crossover_DEFAULT", "AblPortTemplates"),
                       ("v2_ink_boundary_snap", "AblInkSnap"),
                       ("v4_plus_crossover", "AblCrossover"),
                       ("v6_plus_bridge_span7", "AblBridgeSpan")):
    REGISTRY.append(AblationQ(_macro, "results/final/ablation/index.json",
                              f"__delta__.{_label}", table="fig:ablation"))


def value_of(q: Q) -> tuple[float | None, str]:
    if isinstance(q, AblationValQ):
        try:
            return _ablation_val_deltas()[q.key.split(".", 1)[1]], ""
        except Exception as e:                                # noqa: BLE001
            return None, f"{type(e).__name__}: {e}"
    if isinstance(q, AblationQ):
        try:
            return _ablation_deltas()[q.key.split(".", 1)[1]], ""
        except Exception as e:                                # noqa: BLE001
            return None, f"{type(e).__name__}: {e}"
    try:
        return read_value(q) * q.scale, ""
    except FileNotFoundError as e:
        return None, f"missing source: {e.filename}"
    except (KeyError, IndexError) as e:
        return None, f"key not found: {q.key} ({e})"
    except Exception as e:                                    # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def emit() -> int:
    GEN.mkdir(parents=True, exist_ok=True)
    lines = [
        "% GENERATED by scripts/manuscript_numbers.py -- do not edit.",
        "% Each macro's source file and key is stated beside it. Regenerate",
        "% after any results change, then run --check against the manuscript.",
        "",
    ]
    bad = []
    for q in REGISTRY:
        v, err = value_of(q)
        if v is None:
            bad.append((q.name, err))
            continue
        src = q.source.format(seed="{0,1,2}") if q.over_seeds else q.source
        agg = f", {q.over_seeds} over seeds" if q.over_seeds else ""
        lines.append(f"% {src} :: {q.key}{agg}")
        lines.append(f"\\newcommand{{\\{q.name}}}{{{q.fmt.format(v)}}}")
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote paper/generated/numbers.tex -- "
          f"{len(REGISTRY) - len(bad)}/{len(REGISTRY)} quantities")
    for name, err in bad:
        print(f"  UNRESOLVED {name}: {err}")
    return 1 if bad else 0


_NUM_RE = re.compile(r"\\num\{([-+]?[0-9][0-9.eE+\-]*)\}")


def check(tex_path: Path, show_unregistered: bool) -> int:
    text = tex_path.read_text()
    literals = _NUM_RE.findall(text)
    lit_set = {s.strip() for s in literals}

    ok, stale, unresolved = [], [], []
    for q in REGISTRY:
        v, err = value_of(q)
        if v is None:
            unresolved.append((q, err))
            continue
        want = q.fmt.format(v)
        # A negative quantity is conventionally typeset with the sign OUTSIDE
        # the \num{}, as \(-\num{0.6466}\), so the literal in the source has no
        # sign. Accept either spelling rather than reporting a formatting
        # convention as a stale number -- false alarms are how a checker stops
        # being run.
        spellings = {want, want.lstrip("-")} if v < 0 else {want}
        hit = spellings & lit_set
        if hit:
            ok.append((q, sorted(hit)[0]))
        else:
            # Find what the manuscript says instead, if anything close.
            near = [s for s in lit_set
                    if s.replace("-", "").replace("+", "")[:3]
                    == want.replace("-", "").replace("+", "")[:3]]
            stale.append((q, want, near[:3]))

    print(f"manuscript: {tex_path}")
    print(f"  registered quantities   {len(REGISTRY)}")
    print(f"  present and correct     {len(ok)}")
    print(f"  NOT FOUND in the text   {len(stale)}")
    print(f"  unresolved sources      {len(unresolved)}")

    for q, want, near in stale:
        print(f"\n  MISSING/STALE  \\{q.name}"
              f"{' [' + q.table + ']' if q.table else ''}")
        print(f"    source says   {want}   ({q.source} :: {q.key})")
        print(f"    nearest in text: {near if near else 'nothing similar'}")
    for q, err in unresolved:
        print(f"\n  UNRESOLVED  \\{q.name}: {err}")

    if show_unregistered:
        explained = {w for _, w in ok} | {w for _, w, _ in stale}
        rest = sorted(lit_set - explained,
                      key=lambda s: (len(s), s), reverse=True)
        print(f"\n  {len(rest)} \\num{{}} literal(s) with no registry entry. "
              "Not errors -- split sizes, percentages and counts live here too "
              "-- but this is the work list for finishing F1:")
        print("   ", ", ".join(rest[:40]))
        if len(rest) > 40:
            print(f"    ... and {len(rest) - 40} more")

    return 1 if (stale or unresolved) else 0


_CITE_RE = re.compile(r"\\cite\{([^}]*)\}")
_BIBKEY_RE = re.compile(r"^\s*@\w+\s*\{\s*([^,\s]+)\s*,", re.M)


def check_cites(tex_path: Path, bib_path: Path | None) -> int:
    """Every \\cite key, and whether the bibliography defines it.

    An undefined key compiles to a bold [?] that is easy to miss in a long
    document and impossible to miss in review. The bibliography is maintained
    outside this repository, so without --bib this only enumerates what the .bib
    must contain.
    """
    text = tex_path.read_text()
    keys: list[str] = []
    for group in _CITE_RE.findall(text):
        keys.extend(k.strip() for k in group.split(",") if k.strip())
    used = sorted(set(keys))
    print(f"\ncitations: {len(used)} distinct keys, {len(keys)} uses")

    if bib_path is None:
        print("  no --bib given; the bibliography must define each of:")
        for k in used:
            print(f"    {k}  ({keys.count(k)} use(s))")
        return 0

    defined = set(_BIBKEY_RE.findall(bib_path.read_text()))
    missing = [k for k in used if k not in defined]
    unused = sorted(defined - set(used))
    print(f"  bibliography: {bib_path} defines {len(defined)} entries")
    print(f"  MISSING (would compile to [?]): {len(missing)}")
    for k in missing:
        print(f"    {k}")
    if unused:
        print(f"  defined but never cited: {len(unused)} "
              f"({', '.join(unused[:8])}{' ...' if len(unused) > 8 else ''})")
    return 1 if missing else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--check", metavar="TEX")
    ap.add_argument("--unregistered", action="store_true",
                    help="also list \\num{} literals no registry entry explains")
    ap.add_argument("--cites", action="store_true",
                    help="with --check, also audit \\cite keys")
    ap.add_argument("--bib", metavar="BIB",
                    help="bibliography to resolve \\cite keys against")
    a = ap.parse_args()
    if not a.emit and not a.check:
        ap.error("give --emit or --check")
    rc = 0
    if a.emit:
        rc |= emit()
    if a.check:
        rc |= check(Path(a.check), a.unregistered)
        if a.cites or a.bib:
            rc |= check_cites(Path(a.check),
                              Path(a.bib) if a.bib else None)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
