"""lapa dataset builder (split from build_quality_dataset)."""

# ruff: noqa: E402, F403, F405
from __future__ import annotations

from lib.datasets.build.core import *  # noqa: F403


def _lapa_release_roots(root: Path) -> list[Path]:
    candidates = [root, root / "LaPa"]
    candidates.extend(sorted(path for path in root.rglob("LaPa") if path.is_dir()))
    out: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        if any(
            (candidate / split / "landmarks").is_dir()
            for split in ("train", "val", "test")
        ):
            seen.add(resolved)
            out.append(candidate)
    return out


def _build_lapa(
    root: Path,
    output_dir: Path,
    *,
    scenario: str,
    scenarios: tuple[str, ...] | None,
    limit: int | None,
    mode: str,
    allow_overlap: bool,
    image_root: str | None,
) -> Path:
    release_roots = _lapa_release_roots(root)
    if not release_roots:
        return _build_expected_schema_dataset(
            root,
            output_dir,
            dataset="lapa",
            expected_schema="2d_106",
            parser_name="lapa_106",
            scenario=scenario,
            scenarios=scenarios,
            limit=limit,
            mode=mode,
            allow_overlap=allow_overlap,
            image_root=image_root,
        )

    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []
    for release_root in release_roots:
        for source_split in ("train", "val", "test"):
            split_dir = release_root / source_split
            landmark_dir = split_dir / "landmarks"
            image_dir = split_dir / "images"
            label_dir = split_dir / "labels"
            if not landmark_dir.is_dir():
                continue
            lapa_landmark_files = sorted(landmark_dir.glob("*.txt"))
            for landmark_path in track(
                lapa_landmark_files,
                desc=f"Build lapa ({source_split})",
                total=len(lapa_landmark_files),
                unit="file",
            ):
                try:
                    points, detected_schema = _load_landmark_file(landmark_path)
                    if detected_schema != "2d_106":
                        raise ValueError(f"LaPa expected 2d_106, got {detected_schema}")
                    roots = [image_dir]
                    if image_root:
                        roots.insert(0, Path(image_root))
                    image = _find_named_image(roots, landmark_path.stem)
                    if image is None:
                        raise FileNotFoundError(
                            f"LaPa image not found for {landmark_path.name}"
                        )
                except Exception as err:  # noqa: BLE001
                    skipped.append(
                        {"sample_id": landmark_path.as_posix(), "reason": str(err)}
                    )
                    continue

                split = _manifest_split_for_source_split(source_split)
                condition, conds = _native_conditions_for_split(scenario, split)
                label_path = label_dir / f"{landmark_path.stem}.png"
                sample_id = f"{source_split}/{landmark_path.stem}"
                metadata = _path_identity_metadata(
                    landmark_path, root=root, dataset="lapa"
                )
                metadata.update(
                    {
                        "dataset_parser": "lapa_release_106",
                        "parser_type": "dataset_specific",
                        "source_split": source_split,
                        "source_schema": "2d_106",
                        "source_image": str(image.resolve()),
                    }
                )
                if label_path.is_file():
                    metadata["semantic_label"] = str(label_path.resolve())
                samples.append(
                    _with_split(
                        _sample(
                            output_dir=output_dir,
                            dataset="lapa",
                            sample_id=sample_id,
                            image=image,
                            points68=points,
                            condition=condition,
                            conditions=conds,
                            source_schema="2d_106",
                            source_id=sample_id,
                            metadata=metadata,
                        ),
                        split,
                    )
                )

    if not samples:
        raise ValueError(
            f"no LaPa native release samples built; skipped={skipped[:10]}"
        )
    return _write_manifest(
        output_dir,
        "lapa",
        scenario,
        _filter(samples, scenarios, limit),
        mode=mode,
        allow_overlap=allow_overlap,
        scenarios=scenarios,
        skipped=skipped,
    )


# Re-export every module-level name (including the single-underscore build
# helpers) so `from lib.datasets.build.<mod> import *` resolves bare-name
# calls in sibling modules exactly as they did in the original flat module.
__all__ = [_n for _n in dict(globals()) if not _n.startswith("__")]
