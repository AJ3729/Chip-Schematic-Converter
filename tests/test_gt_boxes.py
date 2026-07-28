"""GT box correction (C1): geometry may change, topology may not.

The whole justification for rewriting GT boxes is that the human
verified net assignments, not box extents. These tests pin that
boundary — if a future change to the correction touches anything but
``bbox``, the verified ground truth has been silently altered.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
V2 = REPO / "data/gt_netlists_verified_v2"
V3 = REPO / "data/gt_netlists_verified_v3"

pytestmark = pytest.mark.skipif(
    not (V2.is_dir() and V3.is_dir()),
    reason="verified GT directories not present (data/ is gitignored)",
)

TOPOLOGY_FIELDS = ("id", "class", "terminals", "unconnected")
FILE_FIELDS = ("schema_version", "image", "verified", "annotator", "notes")


def _pairs():
    for p in sorted(V2.glob("circuit_*.json")):
        q = V3 / p.name
        if q.exists():
            yield json.loads(p.read_text()), json.loads(q.read_text()), p.name


def test_v3_covers_v2():
    v2 = {p.name for p in V2.glob("circuit_*.json")}
    v3 = {p.name for p in V3.glob("circuit_*.json")}
    assert v2 == v3


def test_topology_and_provenance_unchanged():
    for a, b, name in _pairs():
        for f in FILE_FIELDS:
            assert a.get(f) == b.get(f), f"{name}: file field {f} changed"
        assert len(a["components"]) == len(b["components"]), name
        for ca, cb in zip(a["components"], b["components"]):
            for f in TOPOLOGY_FIELDS:
                assert ca.get(f) == cb.get(f), f"{name}: component {f} changed"


def test_boxes_are_better_shaped():
    """The point of the correction: fewer square boxes around elongated
    symbols, with centres essentially unmoved."""
    def squarish(b):
        return abs(b[2] - b[3]) / max(b[2], b[3]) < 0.05

    sq_a = sq_b = 0
    max_centre_shift = 0.0
    for a, b, _name in _pairs():
        for ca, cb in zip(a["components"], b["components"]):
            if "bbox" not in ca or "bbox" not in cb:
                continue
            sq_a += squarish(ca["bbox"])
            sq_b += squarish(cb["bbox"])
            max_centre_shift = max(
                max_centre_shift,
                abs(ca["bbox"][0] - cb["bbox"][0]),
                abs(ca["bbox"][1] - cb["bbox"][1]),
            )
    assert sq_b < sq_a, "correction did not reduce square-ish boxes"
    # a handful of components legitimately move (bootstrap mis-assignment);
    # the correction must not be a wholesale re-registration
    assert max_centre_shift < 200


def test_script_dry_run_writes_nothing(tmp_path):
    out = tmp_path / "should_not_exist"
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts/fix_gt_boxes.py"),
         "--out-dir", str(out)],
        cwd=REPO, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "DRY RUN" in r.stdout
    assert not out.exists()
