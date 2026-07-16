"""Configuration loading and override handling.

configs/default.yaml is the single source of truth for every pipeline
threshold. Ablations (Phase E) run by deep-updating that dict, never by
editing code.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load a YAML config; defaults to configs/default.yaml."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {cfg_path} did not parse to a mapping")
    return cfg


def deep_update(base: dict, overrides: dict) -> dict:
    """Return a new dict with `overrides` recursively merged into `base`."""
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def set_by_dotted_key(cfg: dict, dotted_key: str, value) -> dict:
    """Return a copy of `cfg` with e.g. "wires.min_blob_area" set to `value`."""
    keys = dotted_key.split(".")
    override: dict = {keys[-1]: value}
    for key in reversed(keys[:-1]):
        override = {key: override}
    return deep_update(cfg, override)


def config_hash(cfg: dict) -> str:
    """Stable short hash identifying a configuration (keys ablation rows)."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
