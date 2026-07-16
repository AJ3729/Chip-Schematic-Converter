"""Determinism plumbing (Phase A4).

- ``set_global_seed`` seeds every RNG in play (random, numpy, torch when
  present) and requests deterministic torch algorithms.
- ``write_run_metadata`` records config + git SHA + seed + environment
  into the run directory so every experiment is reproducible.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def set_global_seed(seed: int) -> int:
    """Seed random, numpy, and (if installed) torch. Returns the seed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
    return seed


def get_git_sha() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                timeout=5,
                cwd=Path(__file__).resolve().parents[2],
            )
            .stdout.decode()
            .strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def collect_env() -> dict:
    env = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    for mod_name in ("numpy", "cv2", "networkx", "scipy", "yaml"):
        try:
            mod = __import__(mod_name)
            env[mod_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            env[mod_name] = "not installed"
    return env


def write_run_metadata(
    out_dir: str | Path, cfg: dict, seed: int, extra: dict | None = None
) -> Path:
    """Write run_meta.json (config + git SHA + seed + env) to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
        "seed": seed,
        "config": cfg,
        "env": collect_env(),
    }
    if extra:
        meta.update(extra)
    path = out_dir / "run_meta.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return path
