"""The bridging step must close gaps ALONG a stroke and not weld neighbours.

``_bridge_collinear`` in ``closing`` mode is documented as leaving the
perpendicular direction untouched. It does not: a horizontal closing bridges
horizontal gaps between *any* ink, and the horizontal gap between two
side-by-side VERTICAL strokes is exactly such a gap. Two rails 10 px apart fuse
into one solid band at the shipped span of 18, which is a welded net.

These cases pin both halves of the contract. Each asks whether two marked
pixels end up in the SAME connected component -- not how many components the
frame has, because a closing also deposits specks near the border and a raw
count reads those as a failure to bridge when the wire was in fact bridged.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from schematic2netlist.wires import _bridge_collinear, _line_kernel


def cfg(mode: str, span: int = 18, run: int = 7, diag: bool = True) -> dict:
    return {"wires": {"bridge_span": span, "bridge_mode": mode,
                      "bridge_run": run, "bridge_diagonal": diag}}


def joined(mask: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Are pixels a and b in one connected component of the mask?"""
    n, lab = cv2.connectedComponents(mask, connectivity=8)
    la, lb = int(lab[a[1], a[0]]), int(lab[b[1], b[0]])
    assert la and lb, "probe point landed on background"
    return la == lb


# (name, image, probe a, probe b, must_be_joined)
def _cases() -> list[tuple]:
    out = []

    v = np.zeros((70, 70), np.uint8)
    v[5:65, 20:23] = 255
    v[5:65, 33:36] = 255
    out.append(("parallel vertical rails 10px apart", v, (21, 35), (34, 35), False))

    h = np.zeros((70, 70), np.uint8)
    h[20:23, 5:65] = 255
    h[33:36, 5:65] = 255
    out.append(("parallel horizontal rails 10px apart", h, (35, 21), (35, 34), False))

    dh = np.zeros((70, 70), np.uint8)
    dh[33:36, 5:30] = 255
    dh[33:36, 40:65] = 255
    out.append(("dashed horizontal wire, 10px gap", dh, (10, 34), (60, 34), True))

    dv = np.zeros((70, 70), np.uint8)
    dv[5:30, 33:36] = 255
    dv[40:65, 33:36] = 255
    out.append(("dashed vertical wire, 10px gap", dv, (34, 10), (34, 60), True))

    dd = np.zeros((90, 90), np.uint8)
    cv2.line(dd, (10, 10), (32, 32), 255, 3)
    cv2.line(dd, (42, 42), (70, 70), 255, 3)
    out.append(("dashed diagonal wire, ~14px gap", dd, (11, 11), (69, 69), True))

    # a T-junction must survive: the stem and the bar are one net already, and
    # bridging must not be required to notice it
    t = np.zeros((70, 70), np.uint8)
    t[20:23, 5:65] = 255
    t[20:65, 33:36] = 255
    out.append(("T junction stays one net", t, (10, 21), (34, 60), True))
    return out


@pytest.mark.parametrize("name,img,a,b,want", _cases(),
                         ids=[c[0] for c in _cases()])
def test_directional_bridge(name, img, a, b, want):
    got = joined(_bridge_collinear(img, cfg("directional")), a, b)
    assert got == want, (
        f"{name}: directional bridging {'severed' if want else 'welded'} it "
        f"(joined={got}, want={want})")


def test_closing_mode_welds_parallel_rails():
    """The bug the directional mode exists to fix, pinned so it cannot regress
    silently: legacy mode really does weld two rails 10 px apart."""
    v = np.zeros((70, 70), np.uint8)
    v[5:65, 20:23] = 255
    v[5:65, 33:36] = 255
    assert joined(_bridge_collinear(v, cfg("closing")), (21, 35), (34, 35)), \
        "legacy closing no longer welds parallel rails -- if this is now " \
        "fixed, the directional mode's justification needs restating"


def test_nothing_is_removed():
    """Bridging may only add ink. A stroke too short to seed any orientation
    must be left as it was rather than deleted."""
    m = np.zeros((70, 70), np.uint8)
    m[33:36, 5:40] = 255
    m[33:36, 48:52] = 255          # 4 px stub, shorter than any run length
    out = _bridge_collinear(m, cfg("directional"))
    assert np.all(out[m > 0] > 0), "directional bridging deleted ink"


def test_border_ink_is_not_invented():
    """Padding makes the frame border inert: an empty frame stays empty, and
    a stroke does not grow new material at the edge."""
    empty = np.zeros((70, 70), np.uint8)
    assert _bridge_collinear(empty, cfg("directional")).sum() == 0

    edge = np.zeros((70, 70), np.uint8)
    edge[0:3, 10:60] = 255          # a wire lying on the top border
    out = _bridge_collinear(edge, cfg("directional"))
    assert out[:, :5].sum() == 0 and out[:, 65:].sum() == 0, \
        "closing bled ink past the ends of the stroke at the border"


# --- guarded mode: the closing minus its welds -----------------------------

GUARDED_CASES = [
    # a horizontal wire ENDING at a vertical rail across a pen-lift gap is a
    # real connection. A one-sided weld test deletes it; the two-sided test
    # must keep it. This is the case pure gating gets wrong.
    ("horizontal wire meets rail across a 5px gap",
     (slice(20, 23), slice(5, 28)), (slice(5, 65), slice(33, 36)),
     (10, 21), (34, 50), True),
]


def _two_stroke(a, b) -> np.ndarray:
    m = np.zeros((70, 70), np.uint8)
    m[a[0], a[1]] = 255
    m[b[0], b[1]] = 255
    return m


@pytest.mark.parametrize("name,a,b,pa,pb,want", GUARDED_CASES,
                         ids=[c[0] for c in GUARDED_CASES])
def test_guarded_keeps_real_junctions(name, a, b, pa, pb, want):
    m = _two_stroke(a, b)
    assert joined(_bridge_collinear(m, cfg("guarded")), pa, pb) == want, name
    assert joined(_bridge_collinear(m, cfg("directional")), pa, pb) != want, \
        "directional mode is expected to sever this -- if it no longer does, " \
        "the justification for guarded mode needs restating"


@pytest.mark.parametrize("name,img,a,b,want", _cases(),
                         ids=[c[0] for c in _cases()])
def test_guarded_bridge(name, img, a, b, want):
    got = joined(_bridge_collinear(img, cfg("guarded")), a, b)
    assert got == want, (
        f"{name}: guarded bridging {'severed' if want else 'welded'} it "
        f"(joined={got}, want={want})")


def test_guarded_only_removes_fill():
    """Guarded mode may only withhold pixels a closing ADDED: never original
    ink, and never ink no closing would have produced.

    The upper bound is the union of all four oriented closings, not the legacy
    two, because guarded mode runs the diagonal passes too. The lower bound is
    the ink itself -- and note that the LEGACY closing does not satisfy that
    bound, which is what ``test_closing_deletes_ink_at_even_span`` records."""
    rng = np.random.default_rng(0)
    for _ in range(40):
        m = np.zeros((80, 80), np.uint8)
        for k in range(6):
            x, y = rng.integers(0, 60, 2)
            if k % 2:
                m[y:y + 3, x:x + 25] = 255
            else:
                m[y:y + 25, x:x + 3] = 255
        g = _bridge_collinear(m, cfg("guarded"))
        ceil = np.zeros_like(m)
        for o in ("h", "v", "d1", "d2"):
            ceil = cv2.bitwise_or(ceil, cv2.morphologyEx(
                m, cv2.MORPH_CLOSE, _line_kernel(18, o)))
        assert np.all(g[m > 0] > 0), "guarded deleted original ink"
        assert np.all(ceil[g > 0] > 0), "guarded added ink no closing produced"


def test_closing_deletes_ink_at_even_span():
    """A closing is extensive only when the structuring element is symmetric
    about its anchor. OpenCV anchors an even-length kernel off-centre, so
    dilate and erode stop being adjoint and the closing DELETES ink -- a lone
    pixel is annihilated. Measured on 25 frames at 1024 px, every even span
    destroys 0.87-2.40% of all wire ink and every odd span destroys none.

    _line_kernel now rounds up to odd, so this is pinned against the raw
    OpenCV call rather than against the pipeline, and documents why."""
    lone = np.zeros((21, 21), np.uint8)
    lone[10, 10] = 255
    even = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 1))
    assert not cv2.morphologyEx(lone, cv2.MORPH_CLOSE, even)[10, 10], \
        "an even-length closing no longer annihilates a lone pixel -- if " \
        "OpenCV changed its anchor rule, the odd rounding can be revisited"
    assert _line_kernel(18, "h").shape == (1, 19), "span 18 must round to 19"
    assert np.all(_bridge_collinear(lone, cfg("closing"))[lone > 0] > 0), \
        "the pipeline closing still deletes ink despite odd rounding"
