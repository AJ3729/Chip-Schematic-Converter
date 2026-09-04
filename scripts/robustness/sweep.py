#!/usr/bin/env python3
"""Drive every corruption condition, a few at a time, and collect the results.

Conditions are independent, so they run concurrently; detection is CPU-bound
and the box has 8 cores, so three at once is roughly the throughput knee.
Already-finished conditions are skipped, which makes the sweep resumable after
an interrupt rather than a 2.6-hour all-or-nothing.

Usage:
    python scripts/robustness/sweep.py --jobs 3
    python scripts/robustness/sweep.py --only gauss_noise_s2,blur_s3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import corruptions as C  # noqa: E402

RES = ROOT / "results/robustness"


def done(cond: str) -> bool:
    f = RES / cond / "summary.json"
    if not f.exists():
        return False
    try:
        return json.loads(f.read_text()).get("scored", 0) >= 190
    except Exception:
        return False


def run(cond: str) -> tuple[str, bool, float]:
    t = time.time()
    r = subprocess.run(
        [sys.executable, "scripts/robustness/run_condition.py",
         "--condition", cond, "--split", "test"],
        cwd=ROOT, capture_output=True, text=True)
    return cond, r.returncode == 0, time.time() - t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()

    conds = [c for c, _, _ in C.conditions()]
    if a.only:
        conds = [c.strip() for c in a.only.split(",") if c.strip()]
    todo = [c for c in conds if not done(c)]
    print(f"{len(conds)} conditions, {len(conds) - len(todo)} already complete, "
          f"{len(todo)} to run, {a.jobs} at a time", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for i, (cond, ok, dt) in enumerate(ex.map(run, todo), 1):
            print(f"  [{i}/{len(todo)}] {cond:18} "
                  f"{'ok' if ok else 'FAILED'}  {dt:.0f}s "
                  f"(elapsed {(time.time() - t0) / 60:.0f}m)", flush=True)
    print(f"sweep finished in {(time.time() - t0) / 60:.0f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
