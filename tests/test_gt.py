"""Unit tests for the ground-truth topology schema loader/validator."""

from schematic2netlist.gt import (
    SCHEMA_VERSION,
    bootstrap_from_pipeline,
    gt_to_components,
    load_gt,
    save_gt,
    validate_gt,
)


def make_gt(**overrides):
    gt = {
        "schema_version": SCHEMA_VERSION,
        "image": "toy.jpg",
        "source": "manual",
        "verified": True,
        "annotator": "tester",
        "notes": "",
        "components": [
            {
                "id": 0,
                "class": "resistor",
                "bbox": [50, 50, 30, 10],
                "terminals": [
                    {"index": 0, "net": "n1"},
                    {"index": 1, "net": "0"},
                ],
            },
            {
                "id": 1,
                "class": "DC Supply",
                "bbox": [100, 50, 20, 20],
                "terminals": [
                    {"index": 0, "net": "n1"},
                    {"index": 1, "net": "0"},
                ],
            },
            {
                "id": 2,
                "class": "ground",
                "bbox": [75, 90, 16, 10],
                "terminals": [{"index": 0, "net": "0"}],
            },
        ],
    }
    gt.update(overrides)
    return gt


class TestValidator:
    def test_valid_graph_passes(self):
        assert validate_gt(make_gt()) == []

    def test_missing_keys(self):
        assert validate_gt({"image": "x.jpg"})

    def test_duplicate_ids(self):
        gt = make_gt()
        gt["components"][1]["id"] = 0
        assert any("duplicate" in i for i in validate_gt(gt))

    def test_ground_must_be_net_zero_when_verified(self):
        gt = make_gt()
        gt["components"][2]["terminals"][0]["net"] = "n9"
        assert any("must be '0'" in i for i in validate_gt(gt))

    def test_null_net_fails_strict_but_passes_unverified(self):
        gt = make_gt()
        gt["components"][0]["terminals"][0]["net"] = None
        assert any("no net" in i for i in validate_gt(gt))
        gt["verified"] = False
        assert validate_gt(gt) == []

    def test_unconnected_flag_permits_null_when_verified(self):
        gt = make_gt()
        gt["components"][0]["terminals"][0]["net"] = None
        gt["components"][0]["unconnected"] = True
        # n1 now touches only the supply -> flagged as suspicious
        issues = validate_gt(gt)
        assert not any("no net" in i for i in issues)
        assert any("touches only 1" in i for i in issues)

    def test_bad_terminal_indices(self):
        gt = make_gt()
        gt["components"][0]["terminals"][1]["index"] = 5
        assert any("indices" in i for i in validate_gt(gt))

    def test_ground_with_two_terminals_rejected(self):
        gt = make_gt()
        gt["components"][2]["terminals"] = [
            {"index": 0, "net": "0"},
            {"index": 1, "net": "0"},
        ]
        assert any("expected 1" in i for i in validate_gt(gt))

    def test_wrong_terminal_count_for_transistor(self):
        gt = make_gt()
        gt["components"][0] = {
            "id": 0,
            "class": "MOSFET-N",
            "bbox": [50, 50, 30, 10],
            "terminals": [
                {"index": 0, "net": "n1"},
                {"index": 1, "net": "0"},
            ],
        }
        assert any("expected 3" in i for i in validate_gt(gt))

    def test_wire_crossover_rejected_in_topology(self):
        gt = make_gt()
        gt["components"].append({
            "id": 3,
            "class": "Wire Crossover",
            "bbox": [10, 10, 5, 5],
            "terminals": [{"index": 0, "net": "n1"}],
        })
        assert any("drawing annotation" in i for i in validate_gt(gt))

    def test_class_whitelist(self):
        gt = make_gt()
        issues = validate_gt(gt, class_whitelist={"resistor", "ground"})
        assert any("unknown class 'DC Supply'" in i for i in issues)


class TestLoaderRoundTrip:
    def test_save_load_convert(self, tmp_path):
        gt = make_gt()
        path = tmp_path / "toy.json"
        save_gt(gt, path)
        loaded = load_gt(path)
        assert loaded == gt
        comps = gt_to_components(loaded)
        assert comps[0] == {"id": 0, "class": "resistor", "nets": ["n1", "0"]}
        assert comps[2] == {"id": 2, "class": "ground", "nets": ["0"]}

    def test_terminal_order_independent(self, tmp_path):
        gt = make_gt()
        # terminals listed out of order must land by index
        gt["components"][0]["terminals"] = [
            {"index": 1, "net": "0"},
            {"index": 0, "net": "n1"},
        ]
        comps = gt_to_components(gt)
        assert comps[0]["nets"] == ["n1", "0"]


class TestBootstrap:
    def test_bootstrap_from_pipeline_result(self):
        result = {
            "components": [
                {"id": 0, "class": "resistor", "nodes": [3, 7],
                 "node_names": ["n1", "0"], "kind": "two_terminal"},
                {"id": 1, "class": "ground", "nodes": [7, 7],
                 "node_names": ["0", "0"], "kind": "ground"},
            ],
            "detections": [
                {"class": "resistor", "x": 10, "y": 10, "width": 8, "height": 4},
                {"class": "ground", "x": 30, "y": 30, "width": 6, "height": 4},
            ],
        }
        gt = bootstrap_from_pipeline("img.jpg", result)
        assert gt["verified"] is False
        assert gt["source"] == "pipeline_bootstrap"
        assert len(gt["components"][1]["terminals"]) == 1  # ground: 1 terminal
        assert gt["components"][0]["terminals"][1]["net"] == "0"
        # bootstrap output must pass non-strict validation
        assert validate_gt(gt) == []
