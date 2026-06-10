"""multipie dataset builder (split from build_quality_dataset)."""

# ruff: noqa: E402, F403, F405
from __future__ import annotations

from lib.datasets.build.core import *  # noqa: F403


# MenpoBenchmark packages (MultiPIE, Menpo2D) share one annotation format:
# ``<Prefix>_*_train.txt`` rows of ``image bbox(4) coarse(5x2) dense`` with 68-point
# or 39-point profile dense blocks. They are parsed by the same builder.
_MENPO_BENCHMARK_PREFIX = {"multipie": "MultiPIE", "menpo2d": "Menpo2D"}


def _menpo_benchmark_prefix(dataset: str) -> str:
    return _MENPO_BENCHMARK_PREFIX.get(_dataset(dataset), _dataset(dataset))


def _find_multipie_root(root: Path, dataset: str = "multipie") -> Path:
    prefix = _menpo_benchmark_prefix(dataset)
    candidates = sorted(
        path.parent for path in root.rglob(f"{prefix}*train.txt") if path.is_file()
    )
    if candidates:
        return candidates[0]
    if (root / "image").is_dir():
        return root
    raise FileNotFoundError(f"{prefix} root not found below {root}")


def _multipie_annotation_files(root: Path, dataset: str = "multipie") -> list[Path]:
    prefix = _menpo_benchmark_prefix(dataset)
    base = _find_multipie_root(root, dataset)
    files = sorted(base.glob(f"{prefix}*train.txt"))
    if not files:
        raise FileNotFoundError(f"{prefix} train txt files not found in {base}")
    return files


def _multipie_conditions(
    annotation_file: Path, image_rel: str, scenario: str
) -> tuple[str, tuple[str, ...]]:
    text = f"{annotation_file.name} {image_rel}".lower()
    labels: list[str] = []
    if "profile" in text:
        labels.append("profile")
    if "semifrontal" in text or "semi_frontal" in text:
        labels.append("semifrontal")
    if "train" in annotation_file.name.lower():
        labels.append("trainset")
    if not labels:
        labels.append(_label(scenario))
    labels = list(dict.fromkeys(_label(item) for item in labels))
    return labels[0], tuple(labels)


def _multipie_parse_line(
    line: str, *, line_no: int, path: Path
) -> tuple[str, np.ndarray, list[float], str]:
    parts = line.strip().split()
    if len(parts) < 2:
        raise ValueError("empty or malformed line")

    image_rel = parts[0].replace("\\", "/")
    try:
        values = [float(item) for item in parts[1:]]
    except ValueError as err:
        raise ValueError(f"non-numeric landmark value on line {line_no}") from err

    header_values = 14  # 4 bbox + 5 detector/reference points * 2
    dense_count = len(values) - header_values

    if dense_count == 78:
        bbox = [float(item) for item in values[:4]]
        raw = values[header_values:]
        points = np.asarray(raw, dtype=np.float32).reshape(39, 2)
        return image_rel, points, bbox, "2d_39"

    if dense_count != 136:
        raise ValueError(
            f"line {line_no} in {path} has {len(values)} numeric values; "
            "expected 150 for 68-point rows or 92 for 39-point profile rows"
        )

    bbox = [float(item) for item in values[:4]]
    raw = values[header_values:]
    points = np.asarray(raw, dtype=np.float32).reshape(68, 2)
    points = normalize_landmarks(points, source_schema="2d_68")
    return image_rel, points, bbox, "2d_68"


def _bbox_from_points(points68: np.ndarray) -> list[float]:
    left, top = np.min(points68, axis=0)
    right, bottom = np.max(points68, axis=0)
    return [float(left), float(top), float(right), float(bottom)]


def _build_multipie(
    root: Path,
    output_dir: Path,
    *,
    dataset: str = "multipie",
    scenario: str,
    scenarios: tuple[str, ...] | None,
    limit: int | None,
    mode: str,
    allow_overlap: bool,
) -> Path:
    dataset = _dataset(dataset)
    multipie_root = _find_multipie_root(root, dataset)
    annotation_files = _multipie_annotation_files(root, dataset)

    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []

    for annotation_file in track(
        annotation_files,
        desc=f"Build {dataset}",
        total=len(annotation_files),
        unit="file",
    ):
        for line_no, line in enumerate(
            annotation_file.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                image_rel, points68, bbox, source_schema = _multipie_parse_line(
                    line,
                    line_no=line_no,
                    path=annotation_file,
                )
                image_path = (multipie_root / image_rel).resolve()
                if not image_path.is_file():
                    raise FileNotFoundError(f"image not found: {image_path}")

                condition, conds = _multipie_conditions(
                    annotation_file, image_rel, scenario
                )
                bbox = bbox or _bbox_from_points(points68)
                sample_id = Path(image_rel).with_suffix("").as_posix()
                normalizer = _normalizer(points68, sample_id)
                split = (
                    "train"
                    if "trainset" in conds
                    else _deterministic_split(dataset, sample_id)
                )

                metadata = {
                    "annotation_file": str(annotation_file.resolve()),
                    "annotation_line": line_no,
                    "image_id": image_rel,
                    "face_bbox": bbox,
                    "face_bbox_source": f"{dataset}_landmark_bounds",
                    "normalizer_source": DEFAULT_NORMALIZER_SOURCE,
                    "source_schema": source_schema,
                    "split": split,
                }

                sample_kwargs = dict(
                    output_dir=output_dir,
                    dataset=dataset,
                    sample_id=sample_id,
                    image=image_path,
                    points68=points68,
                    condition=condition,
                    conditions=conds,
                    source_schema=source_schema,
                    source_id=sample_id,
                    metadata=metadata,
                )
                try:
                    sample = _sample(**sample_kwargs, normalizer=normalizer)
                except TypeError:
                    sample = _sample(**sample_kwargs)

                samples.append(_with_split(sample, split))
            except Exception as err:  # noqa: BLE001
                skipped.append(
                    {
                        "sample_id": f"{annotation_file.as_posix()}:{line_no}",
                        "reason": str(err),
                    }
                )
                continue

    if not samples:
        raise ValueError(f"no {dataset} samples built; skipped={skipped[:5]}")

    return _write_manifest(
        output_dir,
        dataset,
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
