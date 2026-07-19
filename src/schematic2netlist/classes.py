"""Canonical class vocabulary (Phase C prerequisite).

The benchmark vocabulary is the 17 published Digitize-HCD category
names. Legacy Roboflow-era names are accepted as aliases and
canonicalized at detection-load time, so every stage downstream of
detect.py sees canonical names only. Pipeline logic (ground handling,
SPICE element choice) branches on each class's *role*, never on name
substrings.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

CLASS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "class_names.yaml"


@lru_cache(maxsize=1)
def _load() -> tuple[dict, dict]:
    """Returns (canonical: name -> {role, terminals}, alias_lut: lower -> name)."""
    with open(CLASS_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    canonical = cfg["canonical"]
    lut: dict[str, str] = {}
    for name in canonical:
        lut[name.lower()] = name
    for alias, target in cfg.get("aliases", {}).items():
        if target not in canonical:
            raise ValueError(f"alias {alias!r} -> unknown canonical {target!r}")
        lut[alias.lower()] = target
    return canonical, lut


def canonical_classes() -> list[str]:
    """The 17 canonical names, in stable (YAML) order."""
    canonical, _ = _load()
    return list(canonical)


def canonical_class(name: str) -> str:
    """Map any known class name (canonical or alias, any case) to its
    canonical form. Unknown names pass through unchanged so novel
    classes fail loudly downstream rather than silently here."""
    _, lut = _load()
    return lut.get(name.strip().lower(), name)


def class_role(name: str) -> str:
    """Pipeline role for a class ("ground", "resistor", "nmos", ...).
    Unknown classes get role "unknown"."""
    canonical, _ = _load()
    entry = canonical.get(canonical_class(name))
    return entry["role"] if entry else "unknown"


def class_terminals(name: str) -> int:
    """Electrical terminal count for a class (0 for non-components)."""
    canonical, _ = _load()
    entry = canonical.get(canonical_class(name))
    return int(entry["terminals"]) if entry else 2


def is_ground(name: str) -> bool:
    return class_role(name) == "ground"
