"""Frame size must match preprocess.target_size (adoption of 1024 px).

Image directory and frame size were independently settable — one on the
command line, one in the config — so a 1024-px config pointed at
512-px frames scored the wrong pixels with no error anywhere. Detection
boxes are in frame coordinates too, so the mismatch also corrupts
component alignment. Nothing downstream complains; the numbers are just
quietly wrong, which is the worst failure mode a benchmark can have.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]

from schematic2netlist.frames import (  # noqa: E402
    assert_frames_match_config,
    resolve_and_check,
    resolve_images_dir,
)


def write_frame(d: Path, name: str, long_side: int) -> None:
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / name), np.zeros((long_side // 2, long_side, 3), np.uint8))


def cfg(target: int) -> dict:
    return {"preprocess": {"target_size": target}}


def test_matching_frame_is_accepted(tmp_path):
    write_frame(tmp_path, "a.jpg", 1024)
    assert_frames_match_config(tmp_path, ["a.jpg"], cfg(1024))


def test_undersized_frame_is_rejected(tmp_path):
    """The exact production mistake: 1024 config, 512 frames."""
    write_frame(tmp_path, "a.jpg", 512)
    with pytest.raises(SystemExit, match="frame/config mismatch"):
        assert_frames_match_config(tmp_path, ["a.jpg"], cfg(1024))


def test_oversized_frame_is_rejected(tmp_path):
    write_frame(tmp_path, "a.jpg", 1024)
    with pytest.raises(SystemExit, match="frame/config mismatch"):
        assert_frames_match_config(tmp_path, ["a.jpg"], cfg(512))


def test_missing_files_are_skipped_not_failed(tmp_path):
    """Splits legitimately name images absent from a directory; the
    guard checks frame geometry and must not double as a manifest
    check, or every partial run would abort."""
    write_frame(tmp_path, "b.jpg", 1024)
    assert_frames_match_config(tmp_path, ["absent.jpg", "b.jpg"], cfg(1024))


def test_empty_directory_does_not_raise(tmp_path):
    assert_frames_match_config(tmp_path, ["absent.jpg"], cfg(1024))


def test_shipped_default_config_matches_its_frames():
    """The committed default must be self-consistent — this is the
    check that would have caught the mismatch in the first place."""
    from schematic2netlist.config import load_config

    c = load_config()
    images = REPO / c["preprocess"]["images_dir"]
    if not images.exists():
        pytest.skip(f"{images} not present (data is gitignored)")
    names = sorted(p.name for p in images.glob("*.jpg"))[:1]
    if not names:
        pytest.skip("no frames to check")
    assert_frames_match_config(images, names, c)


def test_resolve_prefers_explicit_then_config():
    c = {"preprocess": {"target_size": 1024, "images_dir": "data/cleaned_1024"}}
    assert resolve_images_dir(None, c) == Path("data/cleaned_1024")
    assert resolve_images_dir("other/dir", c) == Path("other/dir")
    assert resolve_images_dir(None, {"preprocess": {}}) == Path("data/cleaned")


def test_every_pipeline_script_routes_through_the_guard():
    """The guard is only worth having if nothing bypasses it. Any script
    that feeds frames to the pipeline must resolve its directory here,
    not default to a hardcoded data/cleaned."""
    scripts = ["benchmark.py", "oracle.py", "benchmark_repair.py",
               "measure_flagging.py", "benchmark_runtime.py"]
    offenders = []
    for name in scripts:
        src = (REPO / "scripts" / name).read_text()
        if "resolve_and_check" not in src:
            offenders.append(f"{name}: does not call resolve_and_check")
        if 'default="data/cleaned"' in src:
            offenders.append(f"{name}: still hardcodes data/cleaned as a default")
        # passing args.images_dir INTO resolve_and_check is the correct
        # use; building a path out of it directly is the bypass
        for bypass in ("Path(args.images_dir)", "{args.images_dir}"):
            if bypass in src:
                offenders.append(f"{name}: builds a path via {bypass}, "
                                 f"bypassing the guard")
    assert not offenders, "\n".join(offenders)
