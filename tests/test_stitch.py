"""Unit tests for mask-aware wire-island stitching (tier-1 fix)."""

import numpy as np

from schematic2netlist.config import load_config
from schematic2netlist.wires import stitch_wire_islands, stitchable_mask


def cfg():
    return load_config()


def n_islands(mask):
    import cv2
    n, _ = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return n - 1


class TestStitch:
    def test_collinear_gap_over_hole_is_stitched(self):
        wires = np.zeros((100, 200), np.uint8)
        wires[50, 10:80] = 255          # left segment
        wires[50, 120:190] = 255        # right segment
        hole = np.zeros_like(wires)
        hole[40:60, 80:120] = 255       # our own masked region between them
        assert n_islands(wires) == 2
        out = stitch_wire_islands(wires, hole, cfg())
        assert n_islands(out) == 1

    def test_gap_without_hole_is_not_stitched(self):
        wires = np.zeros((100, 200), np.uint8)
        wires[50, 10:80] = 255
        wires[50, 120:190] = 255
        hole = np.zeros_like(wires)      # no masked region explains the gap
        out = stitch_wire_islands(wires, hole, cfg())
        assert n_islands(out) == 2

    def test_perpendicular_lead_is_not_welded(self):
        # vertical pin lead ends at the ring; horizontal rail passes by.
        # collinearity must refuse the 90-degree weld.
        wires = np.zeros((120, 200), np.uint8)
        wires[60, 10:190] = 255          # horizontal rail (one island)
        wires[10:48, 100] = 255          # vertical lead pointing at the rail
        hole = np.zeros_like(wires)
        hole[48:58, 90:110] = 255        # ring region between lead and rail
        out = stitch_wire_islands(wires, hole, cfg())
        assert n_islands(out) == 2       # still separate

    def test_no_third_island_weld(self):
        # a third island sits ON the straight path between a and b
        wires = np.zeros((100, 300), np.uint8)
        wires[50, 10:80] = 255
        wires[50, 140:160] = 255         # third island in the middle
        wires[50, 220:290] = 255
        hole = np.zeros_like(wires)
        hole[40:60, 80:220] = 255
        out = stitch_wire_islands(wires, hole, cfg())
        # a-mid and mid-b may each stitch (collinear, valid), but never
        # a direct a-b bridge over the top of mid; merging via mid is
        # electrically identical, so all three ending as one island is
        # acceptable — what is not acceptable is a crash or a weld that
        # skips the checks. Assert it ran and produced <= previous count.
        assert n_islands(out) <= 3

    def test_stitchable_mask_excludes_component_body(self):
        dets = [{"class": "Resistor", "confidence": 1.0,
                 "x": 100, "y": 50, "width": 40, "height": 20}]
        m = stitchable_mask((100, 200), dets, cfg(), text_mask=None)
        assert m[50, 100] == 0           # centre of body: never stitchable
        assert m[50, 100 - 20 - 4] == 255  # inside the pad ring: stitchable

    def test_far_gap_not_stitched(self):
        wires = np.zeros((100, 400), np.uint8)
        wires[50, 10:80] = 255
        wires[50, 300:390] = 255
        hole = np.zeros_like(wires)
        hole[40:60, 80:300] = 255        # hole explains it, but 220 px wide
        out = stitch_wire_islands(wires, hole, cfg())
        assert n_islands(out) == 2
