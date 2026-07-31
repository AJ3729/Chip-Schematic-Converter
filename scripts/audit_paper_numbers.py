#!/usr/bin/env python3
"""Enforce the no-hand-typed-numbers rule in the manuscript (Phase G).

Every reported quantity must reach the paper through a macro generated
by ``scripts/make_paper_tables.py`` or through an ``\\input`` table, so
that changing a result changes the paper. This script fails when a
result-shaped number is typed directly into prose.

What counts as result-shaped: a decimal (``0.637``), a percentage
(``95.14\\%``), or a bare integer of three digits or more (``1277``).
Ordinary structural numbers are allowed — section/figure references,
LaTeX lengths, small counts written as words or single digits — and an
explicit allowlist covers facts that are properties of the DATASET or
of CITED WORK rather than results of ours (dataset sizes, a cited
paper's headline figure), which must stay verifiable against their
source rather than regenerated from our runs.

Auto-generated files are skipped: they are the output of the rule, not
a violation of it.

Usage:
    python scripts/audit_paper_numbers.py
    python scripts/audit_paper_numbers.py --strict   # also flag allowlisted
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"

# Numbers that are properties of the dataset or of cited work. Each needs
# a source, not a pipeline run. Keep this list short and justified.
ALLOWED = {
    "1277": "Digitize-HCD image count (dataset fact)",
    "18600": "Digitize-HCD box count (dataset fact)",
    "18": "17-class vocabulary / misc structural",
    "895": "frozen train split size",
    "192": "frozen val split size",
    "512": "preprocessing canvas size",
    "640": "detector training resolution",
    "320": "port-crop size",
    "3255": "CGHD image count (dataset fact)",
    "1317": "Wang et al. benchmark size (cited work)",
    "95.14": "Wang et al. reported success rate (cited work)",
    "98.2": "Reddy & Panicker reported mAP (cited work)",
    "96.47": "SINA reported F1 (cited work)",
    "2.93": "Wang et al. reported latency (cited work)",
    "250": "abstract word limit (instruction to authors)",
    "0.3": "IoU alignment threshold (a method parameter, stated in Setup)",
    "0.05": "significance level / CI notation",
    "1.0": "trivial constant",
    "0.10": "port-accuracy reporting threshold (method parameter)",
    "0.15": "port-accuracy reporting threshold (method parameter)",
    "53": "CGHD class-dictionary size (dataset fact)",
    "17": "Digitize-HCD class count (dataset fact)",
    "2000": "bootstrap resample count (method parameter)",
    "1000": "bootstrap resample count (method parameter)",
    "100": "percentage base",
    "2021": "citation year", "2022": "citation year",
    "2023": "citation year", "2024": "citation year",
    "2025": "citation year", "2026": "citation year",
}

NUM = re.compile(r"(?<![\\\w.])(\d+\.\d+|\d{3,})(?![\w])")


def strip_noise(text: str) -> str:
    text = re.sub(r"(?<!\\)%.*", "", text)              # comments
    # LaTeX thousand separators: 1{,}277 is one number, not "1" and "277"
    text = re.sub(r"(\d)\{,\}(\d)", r"\1\2", text)
    # names that merely contain digits, and unit/licence boilerplate
    # metric and product NAMES that contain digits are not reported values
    text = re.sub(r"mAP@0\.5(?::0\.95)?|AP@0\.5(?::0\.95)?|"
                  r"RTX[~ ]?\d+|M1|SHA-?256|CC[~ ]?BY[~ ]?4\.0|YOLOv\d+[a-z]?|"
                  r"\d+(?:\.\d+)?\s*\\,?\s*(?:GB|MB|px|pt|s|ms)\b", " ", text)
    text = re.sub(r"\\(?:label|ref|cite|input|include)\{[^}]*\}", "", text)
    text = re.sub(r"\\(?:todoa|draftnote)\{", "{", text)  # keep inner prose
    text = re.sub(r"\\[a-zA-Z]+(?:\[[^\]]*\])?", " ", text)  # other commands
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true",
                    help="report allowlisted numbers too")
    args = ap.parse_args()

    # figures/ holds TikZ coordinates — geometry, not reported quantities
    targets = sorted(
        p for p in PAPER.rglob("*.tex")
        if not {"generated", "tables", "figures"} & set(p.parts)
    )
    findings, allowed_hits = [], 0
    for p in targets:
        for lineno, raw in enumerate(p.read_text().splitlines(), 1):
            for m in NUM.finditer(strip_noise(raw)):
                tok = m.group(1)
                if tok in ALLOWED and not args.strict:
                    allowed_hits += 1
                    continue
                findings.append((p.relative_to(ROOT), lineno, tok, raw.strip()))

    macros = PAPER / "generated" / "numbers.tex"
    n_macros = len(re.findall(r"\\newcommand", macros.read_text())) \
        if macros.exists() else 0

    print(f"scanned {len(targets)} .tex files; {n_macros} generated macros "
          f"available; {allowed_hits} allowlisted literals")

    # A macro the prose \inputs but nobody generated is invisible to the
    # literal check -- it is not a number, so the audit passes while LaTeX
    # fails on an undefined control sequence. That is not hypothetical:
    # running benchmark_repair.py without --verify silently dropped five
    # macros (30 -> 25) and nothing complained.
    generated = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}",
                               macros.read_text())) if macros.exists() else set()
    referenced: set[str] = set()
    for p in targets:
        if "generated" in str(p):
            continue
        referenced |= set(re.findall(r"\\([A-Z][A-Za-z]{4,})\{\}",
                                     p.read_text(encoding="utf-8")))
    missing = sorted(m for m in referenced if m not in generated)
    if missing:
        print(f"\nFAIL — referenced but never generated: {', '.join(missing)}")
        print("  run scripts/make_paper_tables.py; if a macro is still absent "
              "its source run has not been produced")
        sys.exit(1)

    if not findings:
        print(f"PASS — no hand-typed result numbers, and all "
              f"{len(referenced)} referenced macros are generated")
        return
    print(f"\nFAIL — {len(findings)} literal number(s) in prose:\n")
    for path, lineno, tok, line in findings:
        print(f"  {path}:{lineno}: {tok!r}")
        print(f"      {line[:110]}")
    print("\nEither route the value through scripts/make_paper_tables.py, "
          "or add it to ALLOWED here with the reason it is not a result.")
    sys.exit(1)


if __name__ == "__main__":
    main()
