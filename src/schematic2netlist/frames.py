"""Resolve the preprocessed-frame directory and verify it matches the config.

Frame size (``preprocess.target_size``) and the directory the frames
actually live in used to be settable independently — one in the config,
one as a per-script ``--images-dir`` flag defaulting to ``data/cleaned``.
Pointing a 1024-px config at 512-px frames therefore scored the wrong
pixels with no error raised anywhere, and because detection boxes are
stored in frame coordinates, component alignment corrupted silently too.
The run completes and the numbers are simply wrong, which is the worst
failure mode an evaluation harness can have.

Every script that feeds images to the pipeline resolves its directory
through :func:`resolve_images_dir`, so the two cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path


def resolve_images_dir(explicit: str | Path | None, cfg: dict) -> Path:
    """CLI override if given, else ``preprocess.images_dir`` from cfg."""
    return Path(explicit or cfg["preprocess"].get("images_dir", "data/cleaned"))


def assert_frames_match_config(
    images_dir: str | Path, names: list[str], cfg: dict
) -> None:
    """Raise SystemExit if the frames were not preprocessed at
    ``preprocess.target_size``.

    Checks the first frame that exists and is readable; a preprocessed
    directory is homogeneous by construction. Missing files are skipped
    rather than treated as errors — splits legitimately name images a
    given directory does not hold, and this guard checks geometry, not
    manifest completeness.
    """
    import cv2

    images_dir = Path(images_dir)
    target = cfg["preprocess"]["target_size"]
    for name in names:
        p = images_dir / name
        if not p.exists():
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        got = max(img.shape[:2])
        if got != target:
            raise SystemExit(
                f"frame/config mismatch: {p} is {img.shape[1]}x{img.shape[0]} "
                f"(long side {got}) but preprocess.target_size is {target}.\n"
                f"Point --images-dir at frames preprocessed for {target} px, "
                f"or set preprocess.target_size to {got}."
            )
        return


def resolve_and_check(
    explicit: str | Path | None, names: list[str], cfg: dict
) -> Path:
    """Resolve the frame directory and validate it in one call."""
    images_dir = resolve_images_dir(explicit, cfg)
    assert_frames_match_config(images_dir, names, cfg)
    return images_dir
