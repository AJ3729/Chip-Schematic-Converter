"""Stroke-intersection detection (C2 prerequisite).

Net assembly can only ask "do these strokes connect?" at places it
knows strokes meet, so under-detecting intersections silently caps the
learned path. Both bugs pinned here produced exactly that: zero sites on
shapes that plainly have one.
"""

import numpy as np

from schematic2netlist.skeleton import crop_site, intersection_sites, thin


def _cross(n=41, w=5):
    m = np.zeros((n, n), np.uint8)
    c = n // 2
    m[c - w // 2:c + w // 2 + 1, :] = 255
    m[:, c - w // 2:c + w // 2 + 1] = 255
    return m


def test_thinning_reduces_to_thin_skeleton():
    m = _cross()
    sk = thin(m)
    assert sk.max() == 1
    assert 0 < sk.sum() < (m > 0).sum() / 3     # substantially thinned


def test_cross_has_one_intersection():
    assert len(intersection_sites(_cross())) == 1


def test_tee_has_one_intersection():
    m = np.zeros((41, 41), np.uint8)
    m[18:23, :] = 255
    m[18:, 18:23] = 255
    assert len(intersection_sites(m)) == 1


def test_straight_line_has_none():
    m = np.zeros((41, 41), np.uint8)
    m[18:23, :] = 255
    assert intersection_sites(m) == []


def test_two_separate_crossings_are_two_sites():
    m = np.zeros((60, 120), np.uint8)
    m[28:32, :] = 255
    m[:, 18:22] = 255
    m[:, 98:102] = 255
    assert len(intersection_sites(m)) == 2


def test_blank_mask_is_safe():
    assert intersection_sites(np.zeros((30, 30), np.uint8)) == []


def test_crop_site_pads_at_the_edge():
    m = _cross()
    patch = crop_site(m, 0, 0, 12, 64)        # centred off the corner
    assert patch.shape == (64, 64)
    patch2 = crop_site(m, 20, 20, 12, 64)
    assert patch2.shape == (64, 64) and patch2.any()
