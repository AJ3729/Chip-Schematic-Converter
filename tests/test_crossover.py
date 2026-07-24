"""Unit tests for crossover-aware net assembly (BUILD-C1 / M2).

A plain '+' of two wires is one connected component (wrongly one net).
With a Wire Crossover box at the intersection, the crossover-aware
builder must keep the horizontal and vertical wires on SEPARATE nets
while keeping each wire whole.
"""

import numpy as np

from schematic2netlist.nodes import (
    build_wire_nodes,
    build_wire_nodes_crossover_aware,
)


def make_plus(size=80, center=40, thickness=3):
    m = np.zeros((size, size), np.uint8)
    h = thickness // 2
    m[center - h:center + h + 1, 5:size - 5] = 255   # horizontal wire
    m[5:size - 5, center - h:center + h + 1] = 255   # vertical wire
    return m


def crossover_box(center=40, w=24, h=24):
    return {"class": "Wire Crossover", "x": center, "y": center,
            "width": w, "height": h}


def nets_touching(node_map, y, x, r=2):
    region = node_map[y - r:y + r + 1, x - r:x + r + 1]
    ids = region[region != -1]
    return set(int(v) for v in np.unique(ids))


class TestCrossover:
    def test_plain_plus_is_one_net_classically(self):
        m = make_plus()
        node_map, n = build_wire_nodes(m)
        assert n == 1                      # the bug: X merges to one net

    def test_crossover_splits_into_two_nets(self):
        m = make_plus()
        node_map, n = build_wire_nodes_crossover_aware(m, [crossover_box()])
        assert n == 2                      # horizontal and vertical separated

    def test_horizontal_wire_stays_whole(self):
        m = make_plus()
        node_map, n = build_wire_nodes_crossover_aware(m, [crossover_box()])
        left = nets_touching(node_map, 40, 8)
        right = nets_touching(node_map, 40, 71)
        assert left and left == right      # both ends of H wire: same net

    def test_vertical_wire_stays_whole_and_differs(self):
        m = make_plus()
        node_map, n = build_wire_nodes_crossover_aware(m, [crossover_box()])
        top = nets_touching(node_map, 8, 40)
        bottom = nets_touching(node_map, 71, 40)
        left = nets_touching(node_map, 40, 8)
        assert top and top == bottom       # V wire whole
        assert top != left                 # V and H are different nets

    def test_no_crossover_boxes_is_plain_cc(self):
        m = make_plus()
        _, n = build_wire_nodes_crossover_aware(m, [])
        assert n == 1                      # nothing to split
