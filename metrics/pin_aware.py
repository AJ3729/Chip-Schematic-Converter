"""Pin-aware structural scoring (task D2).

WHY THIS EXISTS. The published metrics canonicalise terminal order away:
`benchmark.canonicalize_terminals` sorts a component's terminals by a
connectivity signature computed identically on both sides, so a swap cancels
and is invisible. That is correct for net NAMES, which are arbitrary. It is
wrong for pin IDENTITY, which is physical -- `netlist.py` writes
`Q<c> <b> <e>` off the terminal order, so a reversed transistor is emitted as
a reversed transistor and simulates wrongly while every topology metric
reports success.

This module scores the same reconstruction with pin identity intact.

A component is CORRECT when its port-to-net assignment matches the reference
under some permutation in that class's declared symmetry group
(`spec/pin_symmetry.yaml`). Passives are symmetric, so reversing a resistor is
not an error. Transistors, diodes, op-amps and sources are asymmetric, so
reversing one is.

Pin-aware strict success = every component correct under a SINGLE consistent
net correspondence. It is strictly harder than the published strict success:
anything it accepts, the published metric accepts too.
"""

from __future__ import annotations

import collections
import itertools
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

SPEC = Path(__file__).resolve().parent.parent / "spec/pin_symmetry.yaml"


@lru_cache(maxsize=1)
def load_symmetry(path: str | None = None) -> dict[str, dict]:
    """Class -> {ports, group}. Cached; the spec is frozen during a run."""
    p = Path(path) if path else SPEC
    if not p.exists():
        raise FileNotFoundError(
            f"{p} is missing. It is authored by the project owner and declares, "
            f"per class, which terminals are genuinely interchangeable. Without "
            f"it there is no principled way to decide whether a reversed "
            f"component is an error.")
    d = yaml.safe_load(p.read_text())
    return {k: {"ports": v.get("ports") or [],
                "group": [tuple(g) for g in (v.get("group") or [])]}
            for k, v in d["classes"].items()}


def permutations_for(cls: str, n: int, sym: dict[str, dict]) -> list[tuple]:
    """Every permutation of terminal indices that the class deems equivalent.

    Always includes the identity. A class absent from the spec is treated as
    fully asymmetric -- the conservative choice, since silently forgiving a
    swap is the failure this module exists to prevent.
    """
    ident = tuple(range(n))
    entry = sym.get(cls)
    if not entry:
        return [ident]
    out = [ident]
    for g in entry["group"]:
        if len(g) == n and sorted(g) == list(range(n)) and tuple(g) != ident:
            out.append(tuple(g))
    return out


@dataclass
class PinAwareResult:
    n_components: int = 0
    n_correct: int = 0
    n_scored: int = 0                 # components with a usable comparison
    n_pred_unmatched: int = 0         # invented components (false positives)
    strict_success: bool = False
    per_class: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)

    @property
    def component_accuracy(self) -> float:
        return self.n_correct / self.n_scored if self.n_scored else 0.0


def _induce_correspondence(pred: list[dict], ref: list[dict],
                           matched: list[tuple[int, int]]) -> dict[str, str]:
    """Net correspondence induced by component matching, INDEPENDENT of pin order.

    This must not use terminal order. Deriving the mapping by zipping a
    component's predicted terminals against its reference terminals assumes the
    pin order is already correct -- which is precisely what the caller is
    trying to measure. Under that scheme a swapped component votes for a
    corrupted mapping, and the corruption then propagates to every other
    component sharing those nets: swapping three passives in a chain made two
    of them look wrong and one look right, none of which is true.

    Instead a net is identified by WHICH matched components touch it, as an
    unordered signature. Terminal order cannot influence that, so the
    correspondence is fixed before pin identity is judged and the two are
    measured independently.
    """
    by_ref = {c["id"]: c for c in ref}
    by_pred = {c["id"]: c for c in pred}
    p2r = dict(matched)

    def signatures(comps, key=lambda cid: cid) -> dict[str, frozenset]:
        touch: dict[str, set] = collections.defaultdict(set)
        for c in comps:
            cid = key(c["id"])
            if cid is None:
                continue
            for n in (c.get("nets") or []):
                if n is not None:
                    touch[str(n)].add(cid)
        return {n: frozenset(v) for n, v in touch.items()}

    # predicted components are keyed by their REFERENCE partner so the two
    # signature spaces are comparable
    psig = signatures([c for c in pred if c["id"] in p2r],
                      key=lambda cid: p2r.get(cid))
    rsig = signatures([by_ref[r] for _, r in matched if r in by_ref])

    by_rsig: dict[frozenset, list[str]] = collections.defaultdict(list)
    for n, s in rsig.items():
        by_rsig[s].append(n)

    corr: dict[str, str] = {}
    for pn, s in psig.items():
        cands = by_rsig.get(s, [])
        # Only an unambiguous signature yields a correspondence. Two reference
        # nets touched by exactly the same components are indistinguishable
        # here, and guessing between them would invent agreement.
        if len(cands) == 1:
            corr[pn] = cands[0]
    return corr


def score_pin_aware(pred: list[dict], ref: list[dict],
                    matched: list[tuple[int, int]],
                    sym: dict[str, dict] | None = None,
                    corr: dict[str, str] | None = None) -> PinAwareResult:
    """Score one circuit with pin identity intact.

    ``pred`` / ``ref``  : [{id, class, nets: [net per terminal, in port order]}]
    ``matched``         : [(pred_id, ref_id)] from the existing Hungarian
                          assignment at IoU 0.3 within class
    ``corr``            : net correspondence; induced from ``matched`` if absent
    """
    sym = sym if sym is not None else load_symmetry()
    corr = corr if corr is not None else _induce_correspondence(pred, ref, matched)

    by_ref = {c["id"]: c for c in ref}
    by_pred = {c["id"]: c for c in pred}
    res = PinAwareResult(n_components=len(ref),
                         n_pred_unmatched=max(0, len(pred) - len(matched)))
    per: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])

    for pid, rid in matched:
        p, r = by_pred.get(pid), by_ref.get(rid)
        if not p or not r:
            continue
        cls = r.get("class", "")
        pn = [None if x is None else str(x) for x in (p.get("nets") or [])]
        rn = [None if x is None else str(x) for x in (r.get("nets") or [])]
        if not rn or len(pn) != len(rn):
            # A differing terminal count is a real failure, not a skip.
            res.n_scored += 1
            per[cls][1] += 1
            res.errors.append({"ref_id": rid, "class": cls,
                               "why": "terminal count differs",
                               "pred": pn, "ref": rn})
            continue

        mapped = [corr.get(x) if x is not None else None for x in pn]
        ok = False
        for perm in permutations_for(cls, len(rn), sym):
            if all(mapped[perm[i]] is not None and mapped[perm[i]] == rn[i]
                   for i in range(len(rn))):
                ok = True
                break
        res.n_scored += 1
        per[cls][1] += 1
        if ok:
            res.n_correct += 1
            per[cls][0] += 1
        else:
            res.errors.append({"ref_id": rid, "class": cls,
                               "why": "no permitted permutation matches",
                               "pred_mapped": mapped, "ref": rn})

    res.per_class = {k: {"correct": v[0], "total": v[1],
                         "accuracy": v[0] / v[1] if v[1] else 0.0}
                     for k, v in sorted(per.items())}
    # Strict = every REFERENCE component correct AND nothing invented.
    #
    # Both halves are needed. Omitting the first would reward a prediction that
    # simply dropped the hard parts. Omitting the SECOND makes this metric
    # non-comparable with the published strict success, which fails a circuit
    # carrying a false positive -- and then a supposedly harder metric can
    # accept a circuit the easier one rejects, which is incoherent. Observed on
    # circuit_308 and circuit_731, both 7 predictions against 6 reference
    # components.
    res.strict_success = (res.n_correct == res.n_components
                          and res.n_components > 0
                          and res.n_pred_unmatched == 0)
    return res


def swap_is_detectable(cls: str, i: int, j: int,
                       sym: dict[str, dict] | None = None) -> bool:
    """Would exchanging terminals i and j of this class be scored as an error?

    False for a symmetric pair -- and that is the passive control: swapping a
    resistor's leads must never register.
    """
    sym = sym if sym is not None else load_symmetry()
    entry = sym.get(cls)
    if not entry:
        return True
    n = len(entry["ports"]) or max(i, j) + 1
    swap = list(range(n))
    swap[i], swap[j] = swap[j], swap[i]
    return tuple(swap) not in set(permutations_for(cls, n, sym))
