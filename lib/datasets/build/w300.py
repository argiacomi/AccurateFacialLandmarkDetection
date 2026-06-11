"""w300 dataset builder (split from build_quality_dataset)."""

# ruff: noqa: E402, F403, F405
from __future__ import annotations

from lib.datasets.build.core import *  # noqa: F403


def _is_300w_cache_root(path: Path) -> bool:
    return any((path / subset).is_dir() for subset in ("afw", "helen", "lfpw", "ibug"))


def _candidate_300w_cache_roots(root: Path, image_root: str | None) -> tuple[Path, ...]:
    raw_candidates: list[Path] = []
    if image_root:
        raw = Path(image_root)
        raw_candidates.extend(
            (
                raw,
                raw / "300w",
                raw / "data" / "300w" / "300w",
                raw / "extracted" / "data" / "300w" / "300w",
            )
        )
    else:
        raw_candidates.extend(
            (
                root,
                root / "300w",
                root / "data" / "300w" / "300w",
                root.parent / "300w" / "data" / "300w" / "300w",
                root.parent / "300w" / "300w",
                ROOT
                / ".fs_cache"
                / "landmark_quality"
                / "300w"
                / "extracted"
                / "data"
                / "300w"
                / "300w",
                ROOT
                / "data"
                / "datasets"
                / "300w"
                / "extracted"
                / "data"
                / "300w"
                / "300w",
                ROOT / "data" / "datasets" / "300w" / "extracted" / "300w",
                ROOT / "data" / "datasets" / "300w" / "extracted",
                ROOT / "data" / "300w" / "300w",
                ROOT / "data" / "300w",
            )
        )

    out: list[Path] = []
    seen: set[Path] = set()
    for candidate in raw_candidates:
        if not candidate.is_dir():
            continue
        if candidate.name in {"trainset", "testset"} and candidate.parent.name in {
            "helen",
            "lfpw",
        }:
            candidate = candidate.parent.parent
        elif candidate.name in {"afw", "helen", "lfpw", "ibug"}:
            candidate = candidate.parent
        if not _is_300w_cache_root(candidate):
            continue
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            out.append(candidate)
    return tuple(out)


def _download_300w_cache_if_missing() -> tuple[Path, ...]:
    """Download/reuse the default 300W cache for annotation-layer datasets.

    HELEN dense annotations are an overlay on 300W Helen images. Standalone
    build_quality_dataset.py invocations do not go through prepare_landmark_dataset.py,
    so lazily populate data/datasets/300w when no cache is already discoverable.

    Disabled by default so tests and local validation never perform an implicit
    network download. Set LANDMARKS_AUTO_DOWNLOAD_300W=1 for CLI fallback use.
    """

    if os.environ.get("LANDMARKS_AUTO_DOWNLOAD_300W") != "1":
        logger.info(
            "300W image cache not found and auto-download is disabled; "
            "set LANDMARKS_AUTO_DOWNLOAD_300W=1 to enable fallback download"
        )
        return ()

    data_root = ROOT / "data" / "datasets"
    try:
        from tools import download_landmark_datasets as downloader
    except Exception as err:  # noqa: BLE001
        logger.warning("could not import downloader for 300W cache fallback: %s", err)
        return ()

    print(
        f"300W image cache not found; downloading/reusing 300w under {data_root}",
        file=sys.stderr,
    )
    try:
        _, registry = downloader.download_datasets(
            ["300w"],
            output_root=data_root,
            extract=True,
            force=False,
            skip_checksum=False,
            keep_going=False,
        )
    except KeyboardInterrupt:
        raise
    except Exception as err:  # noqa: BLE001
        logger.warning("300W cache fallback download failed: %s", err)
        return ()

    resolved = downloader.resolve_source_dir(registry or {}, "300w", data_root)
    candidates: list[Path] = []
    if resolved is not None:
        candidates.extend(
            (
                resolved,
                resolved / "data" / "300w" / "300w",
                resolved / "300w",
            )
        )

    # Also search the standard roots in case the downloader reused an existing
    # registry or extracted marker.
    candidates.extend(_candidate_300w_cache_roots(data_root / "300w", None))

    out: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if candidate.name in {"trainset", "testset"} and candidate.parent.name in {
            "helen",
            "lfpw",
        }:
            candidate = candidate.parent.parent
        elif candidate.name in {"afw", "helen", "lfpw", "ibug"}:
            candidate = candidate.parent
        if not _is_300w_cache_root(candidate):
            continue
        resolved_candidate = candidate.resolve()
        if resolved_candidate not in seen:
            seen.add(resolved_candidate)
            out.append(candidate)
    return tuple(out)


def _helen_300w_roots(root: Path, image_root: str | None) -> tuple[Path, ...]:
    roots = []
    cache_roots = _candidate_300w_cache_roots(root, image_root)
    if not cache_roots and image_root is None:
        cache_roots = _download_300w_cache_if_missing()

    for cache_root in cache_roots:
        helen_root = cache_root / "helen"
        if helen_root.is_dir():
            roots.append(helen_root)
    return tuple(roots)


# Re-export every module-level name (including the single-underscore build
# helpers) so `from lib.datasets.build.<mod> import *` resolves bare-name
# calls in sibling modules exactly as they did in the original flat module.
__all__ = [_n for _n in dict(globals()) if not _n.startswith("__")]
