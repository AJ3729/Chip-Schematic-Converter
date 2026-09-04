#!/usr/bin/env python3
"""record_transforms.py, but driven by preprocess_v2 (the impulse-prefilter copy).

A shim, not a fork. It substitutes the v2 module into sys.modules under the
name the recorder imports, then hands over to the recorder's own main(). That
way there is exactly one copy of the preprocessing logic under test and exactly
one copy of the recording logic, and neither file had to be edited to make this
arm runnable.

Usage: identical to scripts/record_transforms.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from schematic2netlist import preprocess_v2                      # noqa: E402
sys.modules["schematic2netlist.preprocess"] = preprocess_v2      # the swap

import record_transforms                                          # noqa: E402

if __name__ == "__main__":
    # prove the swap took, rather than assuming it
    assert record_transforms.preprocess_image_meta.__module__.endswith("preprocess_v2"), \
        "v2 swap did not take -- the recorder is still on the shipped module"
    raise SystemExit(record_transforms.main())
