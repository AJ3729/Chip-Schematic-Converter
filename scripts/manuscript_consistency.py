#!/usr/bin/env python3
"""Structural consistency of the manuscript (task F3).

``scripts/manuscript_numbers.py`` checks that each number matches the artifact
it came from. This checks the things that are wrong even when every number is
right: a reference to a label that does not exist, a table nobody points at, a
figure that appears twice in the float order, a placeholder still in the text.

None of these produce a LaTeX error. An undefined \\ref typesets as a bold
``??`` that survives a skim; a table that is never referenced reads as
deliberate; a \\TODO renders in red only because this manuscript defines it to.
They are found by looking, and looking is exactly what stops happening in the
week before a deadline.

Checks:
  1. every \\ref resolves to a \\label, and no label is defined twice
  2. every table and figure is referenced at least once in the prose
  3. \\TODO inventory, classified by whether the author or the machine is the
     blocker -- a list of what is left, not a count
  4. numbers stated in two forms (0.5312 and 53.1\\%) agree with each other
  5. no leftover template boilerplate

Exit status is non-zero only for 1 and 5, which are unambiguous defects. The
rest are reported and left to judgement, because a checker whose complaints must
be overridden is a checker that gets switched off.

Usage:
    python scripts/manuscript_consistency.py paper/access.tex
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Strings the IEEE template ships that mean the file was not finished.
BOILERPLATE = (
    (r"Author\s+\\headeretal", "running head still says \"Author\""),
    (r"First\s+A\.\s+Author", "template author name"),
    (r"\bLorem ipsum\b", "placeholder body text"),
    (r"ACCESS\.20\d\d\.DOI", None),   # the DOI placeholder is expected pre-acceptance
    (r"\\author\{\\uppercase\{Author", "template author block"),
)

_LABEL = re.compile(r"\\label\{([^}]*)\}")
_REF = re.compile(r"\\(?:ref|autoref|eqref)\{([^}]*)\}")
_TODO = re.compile(r"\\TODO\{")
_NUM = re.compile(r"\\num\{([-+]?[0-9][0-9.eE+\-]*)\}")

# A \TODO whose text names one of these is waiting on a person, not on code.
AUTHOR_BLOCKED = ("annotation", "annotator", "acknowledg", "version decision",
                  "human review", "supply a circuit", "determinism was measured",
                  "DOI")


def strip_comments(text: str) -> str:
    """Drop % comments so commented-out examples are not read as content."""
    out = []
    for line in text.splitlines():
        i, esc = 0, False
        cut = len(line)
        while i < len(line):
            if line[i] == "\\":
                esc = not esc
            elif line[i] == "%" and not esc:
                cut = i
                break
            else:
                esc = False
            i += 1
        out.append(line[:cut])
    return "\n".join(out)


def check_labels(text: str) -> tuple[list[str], list[str]]:
    labels = _LABEL.findall(text)
    dupes = [k for k, n in Counter(labels).items() if n > 1]
    undefined = sorted(set(_REF.findall(text)) - set(labels))
    return undefined, sorted(dupes)


def check_floats_referenced(text: str) -> list[str]:
    """Tables and figures the prose never points at."""
    refs = set(_REF.findall(text))
    floats = [lab for lab in _LABEL.findall(text)
              if lab.startswith(("tab:", "fig:"))]
    return sorted(lab for lab in floats if lab not in refs)


def todo_inventory(raw: str) -> list[tuple[int, str, str]]:
    """(line, blocker, first sentence) for each \\TODO, in document order."""
    out = []
    for i, line in enumerate(raw.splitlines(), 1):
        if not _TODO.search(line):
            continue
        if line.lstrip().startswith("%"):
            continue          # the header comment explaining what \TODO is for
        # take a window so the classification sees the whole TODO
        body = line
        blocker = ("author" if any(k in body.lower() for k in AUTHOR_BLOCKED)
                   else "machine")
        snippet = re.sub(r".*\\TODO\{", "", body).strip()[:90]
        out.append((i, blocker, snippet))
    return out


_PCT = re.compile(r"\\num\{([0-9.]+)\}\s*\\%")

# How close two literals must be, in characters, to plausibly be the same
# quantity written twice. A sentence.
PROXIMITY_CHARS = 220


def check_percent_forms(text: str) -> list[str]:
    """A quantity written as both a decimal and a percentage must agree.

    PROXIMITY IS THE WHOLE CHECK. Comparing every decimal in the document
    against every percentage finds dozens of pairs that are near each other
    numerically and unrelated in meaning -- 0.8095 beside an unconnected 81%
    somewhere else entirely -- and a check that reports those is one nobody
    reads twice. Only literals within the same sentence-sized window are
    compared, because that is the situation where one was written from the
    other and only one of them got updated.

    An exact match is silent. The interesting case is 0.5312 sitting beside
    53.5%, which is what a half-finished update looks like.
    """
    decs = [(m.start(), m.group(1), float(m.group(1)))
            for m in _NUM.finditer(text)
            if _is_float(m.group(1)) and 0 < float(m.group(1)) < 1]
    pcts = [(m.start(), m.group(1), float(m.group(1)))
            for m in _PCT.finditer(text) if _is_float(m.group(1))]

    problems = []
    for dpos, ds, dv in decs:
        for ppos, ps, pv in pcts:
            if abs(dpos - ppos) > PROXIMITY_CHARS:
                continue
            d = abs(dv * 100 - pv)
            # An integer percentage that is the correct rounding of the decimal
            # is a deliberate choice of precision, not a mismatch. "0.6152
            # post-repair ... 62% of decks" is right, and flagging it teaches
            # the reader to skip this section.
            if "." not in ps and round(dv * 100) == pv:
                continue
            if 0 < d <= 0.6:
                ctx = " ".join(text[min(dpos, ppos):
                                    max(dpos, ppos) + 20].split())[:100]
                problems.append(
                    f"{ds} and {ps}\\% differ by {d:.2f} points, "
                    f"{abs(dpos - ppos)} chars apart -- \"{ctx}\"")
    return sorted(set(problems))


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


_BEGIN = re.compile(r"\\begin\{([^}]*)\}")
_END = re.compile(r"\\end\{([^}]*)\}")


def check_environments(text: str) -> list[str]:
    """Unbalanced \\begin/\\end, and the graphics each figure includes.

    This is not a substitute for compiling -- it cannot be. It is what can be
    checked without a TeX installation, and an unclosed environment is both the
    most common way a document stops compiling and the least informative error
    LaTeX gives for it.
    """
    problems = []
    stack: list[tuple[str, int]] = []
    for i, line in enumerate(text.splitlines(), 1):
        for env in _BEGIN.findall(line):
            stack.append((env, i))
        for env in _END.findall(line):
            if not stack:
                problems.append(f"L{i}: \\end{{{env}}} with nothing open")
            elif stack[-1][0] != env:
                opened, oline = stack[-1]
                problems.append(
                    f"L{i}: \\end{{{env}}} closes \\begin{{{opened}}} "
                    f"opened at L{oline}")
                stack.pop()
            else:
                stack.pop()
    for env, line in stack:
        problems.append(f"L{line}: \\begin{{{env}}} is never closed")

    # every \includegraphics target must exist
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", text):
        rel = m.group(1)
        candidates = [ROOT / "paper" / rel, ROOT / rel]
        if not rel.endswith((".pdf", ".png", ".jpg", ".eps")):
            candidates += [ROOT / "paper" / (rel + ".pdf"),
                           ROOT / (rel + ".pdf")]
        if not any(c.exists() for c in candidates):
            problems.append(f"missing graphic: {rel}")
    return problems


def check_boilerplate(text: str) -> list[str]:
    found = []
    for pattern, why in BOILERPLATE:
        if why is None:
            continue
        if re.search(pattern, text):
            found.append(why)
    return found


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tex")
    a = ap.parse_args()
    raw = Path(a.tex).read_text()
    text = strip_comments(raw)

    print(f"consistency: {a.tex}")
    fatal = 0

    undefined, dupes = check_labels(text)
    print(f"\n1. cross-references: {len(set(_REF.findall(text)))} refs against "
          f"{len(set(_LABEL.findall(text)))} labels")
    if undefined:
        fatal += 1
        print(f"   UNDEFINED (typeset as ??): {undefined}")
    if dupes:
        fatal += 1
        print(f"   DEFINED TWICE (refs silently take the last): {dupes}")
    if not undefined and not dupes:
        print("   all resolve, none duplicated  OK")

    orphans = check_floats_referenced(text)
    print(f"\n2. floats never referenced in prose: {len(orphans)}")
    for o in orphans:
        print(f"   {o}")
    if not orphans:
        print("   every table and figure is pointed at  OK")

    todos = todo_inventory(raw)
    by = Counter(b for _, b, _ in todos)
    print(f"\n3. placeholders: {len(todos)} "
          f"({by.get('author', 0)} waiting on the author, "
          f"{by.get('machine', 0)} on further work)")
    for line, blocker, snip in todos:
        print(f"   L{line:<5} [{blocker:7s}] {snip}")

    pcts = check_percent_forms(text)
    print(f"\n4. decimal/percentage pairs that nearly agree: {len(pcts)}")
    for p in pcts:
        print(f"   {p}")
    if not pcts:
        print("   none  OK")

    envs = check_environments(text)
    print(f"\n5. environments and graphics: {len(envs)} problem(s)")
    for e in envs:
        fatal += 1
        print(f"   {e}")
    if not envs:
        print("   all \\begin/\\end balanced, every \\includegraphics resolves  OK")

    bp = check_boilerplate(text)
    print(f"\n6. template boilerplate: {len(bp)}")
    for b in bp:
        fatal += 1
        print(f"   {b}")
    if not bp:
        print("   none  OK")

    print(f"\n{'FAIL' if fatal else 'PASS'} "
          f"({fatal} unambiguous defect(s); sections 2-4 are for judgement)")
    return 1 if fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())
