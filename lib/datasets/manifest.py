import hashlib
import json
import os
import stat as stat_module
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from lib.datasets.loader_geometry import (
    as_bool_landmark_mask,
    landmark_mask_from_entry,
)
from lib.core.schema import (
    canonicalize_schema,
    flip_map_for_schema,
    head_name_for_schema,
    infer_schema,
    normalize_landmark_array,
    point_count_for_schema,
)
from lib.evaluation.split_safe import (
    entry_in_eval_split,
    manifest_entry_split,
    normalize_heldout_datasets,
)
from lib.training.auxiliary import synthetic_visibility_from_occluder_mask
from lib.training.heatmap_targets import GenerateHeatmap
from lib.transforms.flip import flip_points

try:
    from lib.training.augmentation import GetAugTransform
except ModuleNotFoundError:

    def GetAugTransform():
        raise ModuleNotFoundError(
            "albumentations is required when schema-aware manifest aug=True"
        )


HARD_NEGATIVE_BUCKET_WEIGHTS = {
    "profile_occlusion": 5.0,
    "rolled_profile_occlusion": 5.0,
    "large_yaw_occlusion": 5.0,
    "profile": 3.0,
    "profile_pose": 3.0,
    "large_yaw_pose": 3.0,
    "large_yaw": 3.0,
    "occlusion": 2.0,
    "occluded": 2.0,
    "single_eye_visible": 2.0,
    "mouth_or_jaw_occluded": 2.0,
    "anchor": 1.0,
    "normal": 1.0,
    "frontal": 1.0,
    "clean": 1.0,
}
DEFAULT_HARD_NEGATIVE_WEIGHT = 1.0
MAX_HARD_NEGATIVE_WEIGHT = 5.0


def _normalize_label(value):
    label = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in label:
        label = label.replace("__", "_")
    return label.strip("_")


def _coerce_conditions(entry, metadata):
    raw_items = []
    for value in (entry.get("conditions"), metadata.get("conditions")):
        if isinstance(value, str):
            raw_items.append(value)
        elif isinstance(value, (list, tuple, set)):
            raw_items.extend(value)
        elif isinstance(value, dict):
            raw_items.extend(key for key, present in value.items() if present)
    for key in ("condition", "scenario", "hard_slice", "yaw_slice"):
        if entry.get(key):
            raw_items.append(entry[key])
    if metadata.get("hard_negative_bucket"):
        raw_items.append(metadata["hard_negative_bucket"])
    if metadata.get("condition"):
        raw_items.append(metadata["condition"])

    labels = []
    for item in raw_items:
        label = _normalize_label(item)
        if label and label not in labels:
            labels.append(label)
    return tuple(labels)


@lru_cache(maxsize=256)
def _resolved_base_dir(base_dir):
    # A manifest's base dir is reused for every entry, so resolve it once.
    # Directory resolution is stable for the process lifetime, making the
    # cache safe and bounded by the number of distinct manifest directories.
    return str(Path(base_dir).resolve())


def _resolve_path(base_dir, value):
    raw = str(value or "")
    if os.path.isabs(raw):
        return str(Path(raw))
    # Equivalent to str((base_dir / value).resolve()) but avoids re-resolving the
    # constant base dir on every entry and uses the lighter os.path realpath.
    return os.path.realpath(os.path.join(_resolved_base_dir(str(base_dir)), raw))


def _prepared_crop_from_entry(base_dir, entry, metadata):
    """Resolve an optional pre-resized 256x256 crop declared by a manifest entry.

    Returns ``(abs_image_path, (orig_h, orig_w))`` when the entry provides both a
    ``prepared_image`` path and the native ``prepared_image_orig_hw`` dimensions
    needed to rescale native-space landmarks, else ``("", None)``. The crop is a
    pure decode-cost optimization: callers fall back to decoding the native
    ``image`` whenever it is absent, unreadable, or not exactly 256x256, so a
    missing or stale crop can never change training output.
    """

    value = entry.get("prepared_image") or metadata.get("prepared_image")
    if not value:
        return "", None
    orig_hw = entry.get("prepared_image_orig_hw") or metadata.get(
        "prepared_image_orig_hw"
    )
    try:
        orig_h, orig_w = int(orig_hw[0]), int(orig_hw[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return "", None
    return _resolve_path(base_dir, value), (orig_h, orig_w)


def _clamp_weight(value):
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return DEFAULT_HARD_NEGATIVE_WEIGHT
    if not np.isfinite(weight) or weight <= 0.0:
        return DEFAULT_HARD_NEGATIVE_WEIGHT
    return float(
        min(max(weight, DEFAULT_HARD_NEGATIVE_WEIGHT), MAX_HARD_NEGATIVE_WEIGHT)
    )


def _weight_from_entry(entry, metadata, conditions):
    if "hard_negative_weight" in metadata:
        return _clamp_weight(metadata.get("hard_negative_weight"))

    bucket = _normalize_label(metadata.get("hard_negative_bucket"))
    if bucket in HARD_NEGATIVE_BUCKET_WEIGHTS:
        return HARD_NEGATIVE_BUCKET_WEIGHTS[bucket]

    labels = set(conditions)
    is_profile = any(
        "profile" in label or "large_yaw" in label or label.startswith("yaw_")
        for label in labels
    )
    is_occlusion = any(
        "occlusion" in label or "occluded" in label or "occlud" in label
        for label in labels
    )
    if is_profile and is_occlusion:
        return HARD_NEGATIVE_BUCKET_WEIGHTS["profile_occlusion"]
    if is_profile:
        return HARD_NEGATIVE_BUCKET_WEIGHTS["profile"]
    if is_occlusion:
        return HARD_NEGATIVE_BUCKET_WEIGHTS["occlusion"]

    condition = _normalize_label(entry.get("condition") or entry.get("scenario"))
    return HARD_NEGATIVE_BUCKET_WEIGHTS.get(condition, DEFAULT_HARD_NEGATIVE_WEIGHT)


def _entry_split(entry):
    return manifest_entry_split(entry)


def _coerce_optional_int(value):
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"invalid landmark_count value: {value!r}")


def _schemas_share_trainable_head_and_count(left_schema, right_schema):
    """Return True when two canonical schemas are shape/head-compatible.

    Some eemantic profile schemas are valid for the same array and map
    to the same trainable head. This helper keeps their aliasing
    narrow: both schemas must have the same point count and trainable head.
    """
    if left_schema == right_schema:
        return True
    try:
        return point_count_for_schema(left_schema) == point_count_for_schema(
            right_schema
        ) and head_name_for_schema(left_schema) == head_name_for_schema(right_schema)
    except ValueError:
        return False


class ManifestContractError(ValueError):
    """Raised when a manifest entry violates the declared training contract."""


def _as_bool_landmark_mask(value, landmark_count=68):
    return as_bool_landmark_mask(value, landmark_count)


def _landmark_mask_from_entry(entry, metadata, landmark_count=68):
    return landmark_mask_from_entry(entry, metadata, landmark_count)


def _as_visibility_target(value, landmark_count):
    if value is None:
        return None
    count = int(landmark_count)
    visible_labels = {"visible", "vis", "v", "1", "true", "yes"}
    occluded_labels = {
        "hidden",
        "invisible",
        "occluded",
        "self_occluded",
        "selfoccluded",
        "self_occlusion",
        "externally_occluded",
        "external_occlusion",
        "not_visible",
        "0",
        "false",
        "no",
    }
    if isinstance(value, dict):
        raw = [value.get(str(i), value.get(i, None)) for i in range(count)]
    else:
        raw = value.tolist() if isinstance(value, np.ndarray) else value
    if not isinstance(raw, (list, tuple)) or len(raw) != count:
        return None

    out = []
    known = 0
    for item in raw:
        if item is None:
            out.append(-1)
            continue
        if isinstance(item, str):
            label = _normalize_label(item)
            if label in visible_labels:
                out.append(1)
                known += 1
            elif label in occluded_labels or "occlud" in label:
                out.append(0)
                known += 1
            else:
                out.append(-1)
        else:
            try:
                numeric = float(item)
            except (TypeError, ValueError):
                out.append(1 if bool(item) else 0)
                known += 1
                continue

            if not np.isfinite(numeric) or numeric < 0:
                out.append(-1)
                continue

            out.append(1 if numeric > 0 else 0)
            known += 1
    if known == 0:
        return None
    return np.asarray(out, dtype=np.int64)


def _visibility_target_from_entry(entry, metadata, landmark_count):
    for key in (
        "visibility_target",
        "visibility",
        "landmark_visibility",
        "visibility_mask",
        "landmark_score_visibility_mask",
        "score_visibility_mask",
    ):
        target = _as_visibility_target(entry.get(key), landmark_count)
        if target is not None:
            return target, f"entry.{key}"
        target = _as_visibility_target(metadata.get(key), landmark_count)
        if target is not None:
            return target, f"metadata.{key}"
    return None, ""


MANIFEST_INDEX_VERSION = 1


def _sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_fingerprint(path, *, include_sha256=False):
    path = Path(path)
    # A single stat() (which follows symlinks, like exists()/is_file()/stat())
    # gives existence, file-ness, size, and mtime without repeated syscalls.
    try:
        stat = path.stat()
    except OSError:
        return {
            "path": str(path.expanduser()),
            "exists": False,
            "is_file": False,
            "size": None,
            "mtime_ns": None,
        }

    is_file = stat_module.S_ISREG(stat.st_mode)
    payload = {
        "path": str(path.resolve()),
        "exists": True,
        "is_file": is_file,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_sha256 and is_file:
        payload["sha256"] = _sha256_path(path)
    return payload


def _manifest_index_path(manifest_path):
    manifest_path = Path(manifest_path)
    return manifest_path.with_name(f"{manifest_path.name}.index.jsonl")


def manifest_index_path(manifest_path):
    return _manifest_index_path(manifest_path)


def _manifest_index_header(manifest_path):
    return {
        "type": "manifest_index_meta",
        "version": MANIFEST_INDEX_VERSION,
        "manifest_fingerprint": _path_fingerprint(manifest_path, include_sha256=True),
    }


def _load_manifest_index(manifest_path):
    index_path = _manifest_index_path(manifest_path)
    if not index_path.is_file():
        return {}

    try:
        lines = index_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return {}
        header = json.loads(lines[0])
        if header.get("type") != "manifest_index_meta":
            return {}
        if int(header.get("version", -1)) != MANIFEST_INDEX_VERSION:
            return {}
        if (
            header.get("manifest_fingerprint")
            != _manifest_index_header(manifest_path)["manifest_fingerprint"]
        ):
            return {}

        records = {}
        for line in lines[1:]:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") != "landmark_contract":
                continue
            records[int(record["entry_index"])] = record
        return records
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {}


def _write_manifest_index(manifest_path, records_by_index):
    if not records_by_index:
        return

    index_path = _manifest_index_path(manifest_path)
    lines = [json.dumps(_manifest_index_header(manifest_path), sort_keys=True)]
    for entry_index in sorted(records_by_index):
        record = records_by_index[entry_index]
        if record.get("type") == "landmark_contract":
            lines.append(json.dumps(record, sort_keys=True))

    tmp_path = index_path.with_name(
        f"{index_path.name}.tmp.{int(time.time() * 1_000_000)}"
    )
    try:
        tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp_path.replace(index_path)
    except OSError as exc:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        print(f"warning: could not write manifest index cache {index_path}: {exc}")


def _cached_landmark_contract(index_records, entry_index, landmarks_path):
    record = index_records.get(int(entry_index))
    if not isinstance(record, dict):
        return None

    if record.get("landmarks_path") != str(landmarks_path):
        return None
    if record.get("landmarks_fingerprint") != _path_fingerprint(landmarks_path):
        return None

    contract = record.get("contract")
    if not isinstance(contract, dict):
        return None
    required = {"schema", "target_schema", "landmark_count", "head_name"}
    if not required.issubset(contract):
        return None
    return contract


def _build_landmark_contract(entry, metadata, landmarks_path, entry_index):
    try:
        landmarks = np.load(landmarks_path)
    except OSError:
        raise FileNotFoundError(
            f"could not read landmarks for manifest entry {entry_index}: {landmarks_path}"
        )

    raw_schema = str(metadata.get("source_schema") or entry.get("source_schema") or "")
    raw_target_schema = entry.get("target_schema") or metadata.get("target_schema")
    sample_label = entry.get("sample_id") or entry.get("id") or entry_index
    raw_points = np.asarray(landmarks)

    declared_schema = canonicalize_schema(raw_schema) if raw_schema else None
    declared_source_trainable = False
    if declared_schema is not None:
        try:
            head_name_for_schema(declared_schema)
            declared_source_trainable = True
        except ValueError:
            declared_source_trainable = False

    try:
        xy_points = raw_points[:, :2]
    except (IndexError, TypeError) as exc:
        raise ManifestContractError(
            "manifest landmark array is not a 2D landmark array: "
            f"sample={sample_label!r} "
            f"source_schema={raw_schema or None!r} "
            f"target_schema={raw_target_schema or None!r} "
            f"shape={tuple(raw_points.shape)!r}"
        ) from exc

    try:
        detected_schema = infer_schema(xy_points)
    except ValueError as exc:
        if raw_target_schema not in (None, "") or declared_source_trainable:
            raise ManifestContractError(
                "manifest landmark array shape is not compatible with declared schema: "
                f"sample={sample_label!r} "
                f"source_schema={raw_schema or None!r} "
                f"target_schema={raw_target_schema or None!r} "
                f"shape={tuple(raw_points.shape)!r}"
            ) from exc
        raise

    if declared_schema is None:
        declared_schema = detected_schema
    target_schema = (
        canonicalize_schema(raw_target_schema)
        if raw_target_schema not in (None, "")
        else detected_schema
    )

    if raw_schema and declared_schema != detected_schema:
        has_explicit_target = raw_target_schema not in (None, "")
        source_schema_matches_loaded_points = _schemas_share_trainable_head_and_count(
            declared_schema,
            detected_schema,
        )
        if not source_schema_matches_loaded_points and not (
            has_explicit_target and target_schema == detected_schema
        ):
            raise ManifestContractError(
                "manifest source_schema does not match loaded landmark array "
                "and no explicit matching target_schema was provided: "
                f"sample={sample_label!r} "
                f"source_schema={declared_schema!r} "
                f"detected_schema={detected_schema!r} "
                f"target_schema={target_schema!r} "
                f"shape={tuple(raw_points.shape)!r}"
            )

    if target_schema != detected_schema:
        raise ManifestContractError(
            "manifest target_schema does not match loaded landmark array: "
            f"sample={sample_label!r} "
            f"target_schema={target_schema!r} actual_schema={detected_schema!r} "
            f"shape={tuple(raw_points.shape)!r}"
        )

    try:
        normalized_landmarks = normalize_landmark_array(xy_points, schema=target_schema)
    except ValueError as exc:
        raise ManifestContractError(
            "manifest landmarks failed target_schema normalization: "
            f"sample={sample_label!r} "
            f"target_schema={target_schema!r} "
            f"shape={tuple(raw_points.shape)!r}: {exc}"
        ) from exc

    actual_schema = infer_schema(normalized_landmarks[:, :2])
    if target_schema != actual_schema:
        raise ManifestContractError(
            "manifest target_schema does not match normalized landmark array: "
            f"sample={sample_label!r} "
            f"target_schema={target_schema!r} actual_schema={actual_schema!r} "
            f"shape={tuple(normalized_landmarks.shape)!r}"
        )

    expected_head_name = head_name_for_schema(target_schema)
    head_name = str(
        entry.get("head_name") or metadata.get("head_name") or expected_head_name
    )
    if head_name != expected_head_name:
        raise ManifestContractError(
            "manifest head_name does not match target_schema: "
            f"sample={sample_label!r} "
            f"head_name={head_name!r} expected={expected_head_name!r}"
        )

    raw_landmark_count = entry.get("landmark_count")
    if raw_landmark_count in (None, ""):
        raw_landmark_count = metadata.get("landmark_count")
    try:
        declared_landmark_count = _coerce_optional_int(raw_landmark_count)
    except ValueError as exc:
        raise ManifestContractError(str(exc)) from exc
    expected_landmark_count = point_count_for_schema(target_schema)
    if (
        declared_landmark_count is not None
        and declared_landmark_count != expected_landmark_count
    ):
        raise ManifestContractError(
            "manifest landmark_count does not match target_schema: "
            f"sample={sample_label!r} "
            f"landmark_count={declared_landmark_count!r} "
            f"expected={expected_landmark_count!r}"
        )

    return {
        "schema": str(target_schema),
        "target_schema": str(target_schema),
        "landmark_count": int(normalized_landmarks.shape[0]),
        "head_name": str(head_name),
    }


def _landmark_contract_for_entry(
    entry, metadata, landmarks_path, entry_index, index_records
):
    cached_contract = _cached_landmark_contract(
        index_records, entry_index, landmarks_path
    )
    if cached_contract is not None:
        return cached_contract, index_records[int(entry_index)], True

    contract = _build_landmark_contract(entry, metadata, landmarks_path, entry_index)
    record = {
        "type": "landmark_contract",
        "version": MANIFEST_INDEX_VERSION,
        "entry_index": int(entry_index),
        "landmarks_path": str(landmarks_path),
        "landmarks_fingerprint": _path_fingerprint(landmarks_path),
        "contract": contract,
    }
    return contract, record, False


def build_manifest_index(manifest_path):
    """Build the landmark contract index beside a schema-aware manifest."""

    manifest_path = Path(manifest_path)
    base_dir = manifest_path.parent
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("samples", payload.get("scenarios", []))
    if not isinstance(entries, list):
        raise ValueError(
            f"manifest {manifest_path} must contain a samples or scenarios list"
        )

    cached_index_records = _load_manifest_index(manifest_path)
    records_by_index = dict(cached_index_records)
    indexed_count = 0
    cache_hit_count = 0
    skipped_count = 0
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            skipped_count += 1
            continue
        metadata = (
            entry.get("metadata", {}) if isinstance(entry.get("metadata"), dict) else {}
        )
        landmarks_value = entry.get("landmarks") or entry.get("ground_truth")
        if not landmarks_value:
            skipped_count += 1
            continue
        landmarks_path = _resolve_path(base_dir, landmarks_value)
        try:
            _, index_record, cache_hit = _landmark_contract_for_entry(
                entry,
                metadata,
                landmarks_path,
                index,
                cached_index_records,
            )
        except ManifestContractError:
            raise
        except ValueError:
            skipped_count += 1
            continue
        records_by_index[int(index)] = index_record
        indexed_count += 1
        if cache_hit:
            cache_hit_count += 1

    _write_manifest_index(manifest_path, records_by_index)
    return {
        "index_path": str(_manifest_index_path(manifest_path)),
        "indexed_count": int(indexed_count),
        "cache_hit_count": int(cache_hit_count),
        "skipped_count": int(skipped_count),
    }


class LandmarkDataset(Dataset):
    """Schema-aware landmark manifest dataset.

    `FS68Manifest` is a backward-compatible data_name alias. The loader accepts
    mixed trainable landmark schemas such as 29, 39, 68, 98, 106, and 194 points
    when `schema_aware_training=True`.

    Expected manifest schema:
      {"samples": [{"image": "...", "landmarks": "...", "source_schema": "...",
                    "target_schema": "...", "metadata": {...}}]}

    Landmark `.npy` arrays are treated as pixel coordinates unless their values
    are normalized to [0, 1], in which case they are scaled to the 256x256 CD-ViT
    crop. Hard-negative and visibility metadata are preserved for weighting,
    slicing, and auxiliary losses.
    """

    def __init__(
        self,
        manifest_path,
        split="train",
        preload=True,
        aug=True,
        heatmap_size=0,
        perturbation=0,
        eval_mode="random_hash",
        heldout_datasets=None,
        include_metadata=False,
        schema_aware_training=False,
        split_policy="declared_or_random_hash",
    ):
        super(LandmarkDataset, self).__init__()
        if perturbation:
            raise ValueError(
                "schema-aware landmark manifests do not support perturbation mode"
            )
        if not manifest_path:
            raise ValueError(
                "schema-aware landmark manifests require --manifest, --train_manifest, or --test_manifest"
            )

        self.manifest_path = Path(manifest_path)
        self.split = split
        self.eval_mode = eval_mode
        self.heldout_datasets = normalize_heldout_datasets(heldout_datasets)
        self.include_metadata = bool(include_metadata)
        self.schema_aware_training = bool(schema_aware_training)
        self.split_policy = split_policy
        self.heatmap_size = int(heatmap_size or 0)
        self.samples = self._load_manifest(
            self.manifest_path,
            split,
            self.eval_mode,
            self.heldout_datasets,
            self.split_policy,
        )
        if not self.samples:
            detail = f" split={split!r} eval_mode={self.eval_mode!r}"
            if self.heldout_datasets:
                detail += f" heldout_datasets={self.heldout_datasets!r}"
            raise ValueError(
                f"no trainable schema-aware samples found in {self.manifest_path} for{detail}"
            )

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.aug_transform = GetAugTransform() if aug else None
        self.generateHM = (
            GenerateHeatmap(self.heatmap_size) if self.heatmap_size > 0 else None
        )
        self.data_list = self.loaditem_list() if preload else None

    def _load_manifest(
        self, manifest_path, split, eval_mode, heldout_datasets, split_policy
    ):
        base_dir = manifest_path.parent
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = payload.get("samples", payload.get("scenarios", []))
        if not isinstance(entries, list):
            raise ValueError(
                f"manifest {manifest_path} must contain a samples or scenarios list"
            )

        declared_splits = {
            _entry_split(entry)
            for entry in entries
            if isinstance(entry, dict) and _entry_split(entry)
        }
        use_split_filter = bool(declared_splits)

        cached_index_records = _load_manifest_index(manifest_path)
        index_records = dict(cached_index_records)
        index_dirty = False

        samples = []
        skipped_non_trainable_schema = 0
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            if not entry_in_eval_split(
                entry,
                index,
                split=split,
                eval_mode=eval_mode,
                heldout_datasets=heldout_datasets,
                has_declared_splits=use_split_filter,
                split_policy=split_policy,
            ):
                continue

            metadata = (
                entry.get("metadata", {})
                if isinstance(entry.get("metadata"), dict)
                else {}
            )
            source = (
                entry.get("source", {}) if isinstance(entry.get("source"), dict) else {}
            )
            landmarks_value = entry.get("landmarks") or entry.get("ground_truth")
            image_value = entry.get("image")
            if not landmarks_value or not image_value:
                continue

            landmarks_path = _resolve_path(base_dir, landmarks_value)
            try:
                contract, index_record, cache_hit = _landmark_contract_for_entry(
                    entry,
                    metadata,
                    landmarks_path,
                    index,
                    cached_index_records,
                )
                if not cache_hit:
                    index_records[int(index)] = index_record
                    index_dirty = True
                schema = str(contract["schema"])
                target_schema = str(contract["target_schema"])
                landmark_count = int(contract["landmark_count"])
                head_name = str(contract["head_name"])
            except ManifestContractError:
                raise
            except ValueError:
                skipped_non_trainable_schema += 1
                continue

            if not self.schema_aware_training and schema != "2d_68":
                skipped_non_trainable_schema += 1
                continue

            if isinstance(entry.get("auxiliary_labels"), dict):
                metadata = dict(metadata)
                metadata.setdefault("auxiliary_labels", entry["auxiliary_labels"])
            occluder_mask_value = (
                entry.get("occluder_mask")
                or entry.get("synthetic_occluder_mask")
                or metadata.get("occluder_mask")
                or metadata.get("synthetic_occluder_mask")
            )
            landmark_mask = _landmark_mask_from_entry(entry, metadata, landmark_count)
            visibility_target, visibility_target_source = _visibility_target_from_entry(
                entry, metadata, landmark_count
            )
            conditions = _coerce_conditions(entry, metadata)
            prepared_image, prepared_orig_hw = _prepared_crop_from_entry(
                base_dir, entry, metadata
            )
            samples.append(
                {
                    "sample_id": str(
                        entry.get("sample_id")
                        or entry.get("id")
                        or entry.get("name")
                        or index
                    ),
                    "image": _resolve_path(base_dir, image_value),
                    "prepared_image": prepared_image,
                    "prepared_image_orig_hw": prepared_orig_hw,
                    "landmarks": landmarks_path,
                    "dataset": str(
                        entry.get("dataset") or metadata.get("dataset") or ""
                    ),
                    "condition": str(
                        entry.get("condition") or entry.get("scenario") or ""
                    ),
                    "conditions": conditions,
                    "source_schema": schema,
                    "target_schema": target_schema,
                    "landmark_count": int(landmark_count),
                    "head_name": head_name,
                    "split": str(entry.get("split") or metadata.get("split") or ""),
                    "split_safe_id": str(
                        entry.get("split_safe_id")
                        or metadata.get("split_safe_id")
                        or ""
                    ),
                    "source": source,
                    "metadata": metadata,
                    "face_bbox": entry.get(
                        "face_bbox",
                        metadata.get(
                            "face_bbox", entry.get("bbox", metadata.get("bbox"))
                        ),
                    ),
                    "bbox_format": entry.get(
                        "bbox_format", metadata.get("bbox_format", "")
                    ),
                    "visibility_target": visibility_target,
                    "visibility_target_source": visibility_target_source,
                    "synthetic_visibility_occluder_mask": _resolve_path(
                        base_dir, occluder_mask_value
                    )
                    if occluder_mask_value
                    else "",
                    "sample_weight": _weight_from_entry(entry, metadata, conditions),
                    "landmark_mask": landmark_mask,
                }
            )

        if index_dirty:
            _write_manifest_index(manifest_path, index_records)

        if skipped_non_trainable_schema:
            reason = "non-trainable" if self.schema_aware_training else "non-68-point"
            print(
                f"schema-aware manifest skipped {skipped_non_trainable_schema} {reason} sample(s) from {manifest_path}"
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_prepared_crop(self, sample):
        """Load a pre-resized 256x256 crop, or return ``None`` to fall back.

        The crop is the byte-lossless result of resizing the native image to
        256x256 with INTER_LINEAR, stored as a BGR PNG. ``cv2.resize`` is a
        per-channel spatial op and the BGR->RGB swap only reorders channels, so
        ``swap(resize(decode))`` (native path) equals ``resize(swap(decode))``
        bit for bit. Landmarks stay in native space and are rescaled with the
        stored original dimensions using the same arithmetic as the native
        branch, so the returned image and landmarks are identical to decoding
        the native image. Any miss (no crop, unreadable, or unexpected shape)
        returns ``None`` so the caller decodes the native image instead.
        """

        prepared_image = sample.get("prepared_image")
        orig_hw = sample.get("prepared_image_orig_hw")
        if not prepared_image or not orig_hw:
            return None
        img = cv2.imread(prepared_image, cv2.IMREAD_COLOR)
        if img is None or img.shape[0] != 256 or img.shape[1] != 256:
            return None
        img = img[:, :, [2, 1, 0]]
        lmk = np.load(sample["landmarks"]).astype(np.float32)[:, :2]
        if float(np.nanmax(lmk)) <= 1.5:
            lmk = lmk * 255.0
        orig_h, orig_w = int(orig_hw[0]), int(orig_hw[1])
        if orig_h != 256 or orig_w != 256:
            scale_x = 256.0 / float(orig_w)
            scale_y = 256.0 / float(orig_h)
            lmk[:, 0] *= scale_x
            lmk[:, 1] *= scale_y
        return img, lmk

    def _load_image_and_landmarks(self, sample):
        prepared = self._load_prepared_crop(sample)
        if prepared is not None:
            return prepared

        img = cv2.imread(sample["image"], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"could not read image {sample['image']}")
        img = img[:, :, [2, 1, 0]]
        lmk = np.load(sample["landmarks"]).astype(np.float32)[:, :2]

        if float(np.nanmax(lmk)) <= 1.5:
            lmk = lmk * 255.0

        h, w = img.shape[:2]
        if h != 256 or w != 256:
            scale_x = 256.0 / float(w)
            scale_y = 256.0 / float(h)
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
            lmk[:, 0] *= scale_x
            lmk[:, 1] *= scale_y

        return img, lmk

    def loaditem_list(self):
        data_list = []
        for sample in self.samples:
            img, lmk = self._load_image_and_landmarks(sample)
            data_list.append((img, lmk, sample["landmark_mask"].copy()))
        return data_list

    def MakeLMKInsideImage(self, img, lmk, landmark_mask=None):
        if landmark_mask is None:
            valid = np.ones((lmk.shape[0],), dtype=bool)
        else:
            valid = np.asarray(landmark_mask, dtype=np.float32) > 0.5
            if valid.shape[0] != lmk.shape[0] or not valid.any():
                valid = np.ones((lmk.shape[0],), dtype=bool)

        finite = np.isfinite(lmk).all(axis=1)
        valid = valid & finite
        if not valid.any():
            raise ValueError("no finite valid landmarks")

        valid_lmk = lmk[valid]
        lt = np.min(valid_lmk, axis=0)
        rb = np.max(valid_lmk, axis=0)

        padding = 0.0
        margin = 5.0
        if lt[0] < margin:
            padding = margin - lt[0]
        if lt[1] < margin:
            padding = max(margin - lt[1], padding)
        if rb[0] > img.shape[1] - margin:
            padding = max(padding, rb[0] - img.shape[1] + margin)
        if rb[1] > img.shape[0] - margin:
            padding = max(padding, rb[1] - img.shape[0] + margin)

        if not np.isfinite(padding):
            raise ValueError(f"non-finite landmark padding: {padding}")

        h, w = int(img.shape[0]), int(img.shape[1])
        padded_h = h + 2 * int(np.ceil(padding))
        padded_w = w + 2 * int(np.ceil(padding))

        # Allow heavily cropped/translated faces, but block pathological samples
        # that would ask OpenCV to allocate huge temporary images.
        max_padded_side = 2048
        max_padded_pixels = 2048 * 2048
        if (
            padded_h > max_padded_side
            or padded_w > max_padded_side
            or padded_h * padded_w > max_padded_pixels
        ):
            raise ValueError(
                f"unreasonable landmark padding: padding={padding:.2f}, "
                f"padded_shape=({padded_h}, {padded_w}), "
                f"lt={lt.tolist()} rb={rb.tolist()} image_shape={img.shape[:2]}"
            )

        if padding > 0:
            padding = int(round(padding))
            new_img = cv2.copyMakeBorder(
                img, padding, padding, padding, padding, cv2.BORDER_CONSTANT
            )
            lmk = lmk + padding
            lmk = lmk * img.shape[0] / new_img.shape[0]
            new_img = cv2.resize(new_img, (img.shape[0], img.shape[1]))
            return new_img, lmk
        return img, lmk

    def __getitem__(self, item):
        sample = self.samples[item]
        visibility_target = sample.get("visibility_target")
        visibility_target_source = sample.get("visibility_target_source", "")
        visibility_target_weight = None
        if self.data_list is None:
            img, lmk = self._load_image_and_landmarks(sample)
            landmark_mask = sample["landmark_mask"].copy()
        else:
            img, lmk, landmark_mask = self.data_list[item]
            img = img.copy()
            lmk = lmk.copy()
            landmark_mask = landmark_mask.copy()
        synthetic_occluder_mask = None
        if visibility_target is None and sample.get(
            "synthetic_visibility_occluder_mask"
        ):
            synthetic_occluder_mask = cv2.imread(
                sample["synthetic_visibility_occluder_mask"],
                cv2.IMREAD_GRAYSCALE,
            )
            if (
                synthetic_occluder_mask is not None
                and synthetic_occluder_mask.shape[:2] != img.shape[:2]
            ):
                synthetic_occluder_mask = cv2.resize(
                    synthetic_occluder_mask,
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

        if self.aug_transform is not None:
            if synthetic_occluder_mask is None:
                transformed = self.aug_transform(image=img, keypoints=lmk)
            else:
                try:
                    transformed = self.aug_transform(
                        image=img,
                        keypoints=lmk,
                        mask=synthetic_occluder_mask,
                    )
                except TypeError:
                    transformed = self.aug_transform(image=img, keypoints=lmk)
                    synthetic_occluder_mask = None

            img = transformed["image"]
            lmk = np.array(transformed["keypoints"], dtype=np.float32)
            if synthetic_occluder_mask is not None:
                synthetic_occluder_mask = transformed.get(
                    "mask", synthetic_occluder_mask
                )

            if np.random.random() < 0.5:
                if self.schema_aware_training:
                    try:
                        flip_index = flip_map_for_schema(sample["source_schema"])
                    except ValueError:
                        flip_index = None
                else:
                    flip_index = np.asarray(flip_points("300W"), dtype=np.int64)
                if flip_index is not None:
                    img = cv2.flip(img, 1)
                    lmk = lmk[flip_index, :]
                    landmark_mask = landmark_mask[flip_index]
                    if visibility_target is not None:
                        visibility_target = np.asarray(visibility_target)[flip_index]
                    if visibility_target_weight is not None:
                        visibility_target_weight = np.asarray(visibility_target_weight)[
                            flip_index
                        ]
                    lmk[:, 0] = 255 - lmk[:, 0]

        if visibility_target is None and synthetic_occluder_mask is not None:
            visibility_target = synthetic_visibility_from_occluder_mask(
                lmk,
                synthetic_occluder_mask,
                valid_mask=landmark_mask,
            )
            visibility_target_source = "synthetic_occluder_mask"
            visibility_target_weight = np.ones(
                (int(visibility_target.shape[0]),),
                dtype=np.float32,
            )

        try:
            img, lmk = self.MakeLMKInsideImage(img, lmk, landmark_mask)
        except Exception as exc:
            meta = (
                sample.get("metadata", {})
                if isinstance(sample.get("metadata"), dict)
                else {}
            )
            raise RuntimeError(
                "MakeLMKInsideImage failed for "
                f"item={item} "
                f"sample_id={sample.get('sample_id', '')!r} "
                f"dataset={sample.get('dataset', meta.get('dataset', ''))!r} "
                f"image={sample.get('image', sample.get('image_path', sample.get('path', '')))!r} "
                f"landmarks={sample.get('landmarks', sample.get('points', ''))!r} "
                f"source_schema={sample.get('source_schema', '')!r} "
                f"target_schema={sample.get('target_schema', '')!r} "
                f"lmk_min={np.nanmin(lmk, axis=0).tolist() if np.size(lmk) else None} "
                f"lmk_max={np.nanmax(lmk, axis=0).tolist() if np.size(lmk) else None}"
            ) from exc
        img = self.transform(img)
        lmk = torch.from_numpy(lmk / 255.0).float()
        landmark_mask_t = torch.from_numpy(
            np.asarray(landmark_mask, dtype=np.float32)
        ).float()

        if self.generateHM is not None:
            heatmap = self.generateHM.Generate((lmk * (self.heatmap_size - 1)).numpy())
            heatmap = torch.from_numpy(heatmap).float()
            heatmap = heatmap * landmark_mask_t.reshape(-1, 1, 1)
            denom = torch.sum(heatmap, dim=(1, 2), keepdim=True).clamp_min(1e-6)
            heatmap = torch.where(
                landmark_mask_t.reshape(-1, 1, 1) > 0.0, heatmap / denom, heatmap
            )
            if self.schema_aware_training:
                metadata = dict(sample.get("metadata", {}))
                metadata.update(
                    {
                        "sample_id": sample.get("sample_id", ""),
                        "dataset": sample.get("dataset", ""),
                        "condition": sample.get("condition", ""),
                        "conditions": list(sample.get("conditions", ())),
                        "source_schema": sample.get("source_schema", ""),
                        "target_schema": sample.get("target_schema", ""),
                        "landmark_count": sample.get("landmark_count", 0),
                        "head_name": sample.get("head_name", ""),
                        "split": sample.get("split", ""),
                        "split_safe_id": sample.get("split_safe_id", ""),
                        "face_bbox": sample.get("face_bbox"),
                        "bbox_format": sample.get("bbox_format", ""),
                        "visibility_target": visibility_target.tolist()
                        if hasattr(visibility_target, "tolist")
                        else visibility_target,
                        "visibility_target_source": visibility_target_source,
                        "hard_negative_bucket": metadata.get(
                            "hard_negative_bucket", ""
                        ),
                    }
                )
                if visibility_target is None:
                    visibility_target_t = torch.full(
                        (int(sample.get("landmark_count", lmk.shape[0])),),
                        -1.0,
                        dtype=torch.float32,
                    )
                else:
                    visibility_target_t = torch.as_tensor(
                        visibility_target,
                        dtype=torch.float32,
                    )
                if visibility_target_weight is None:
                    visibility_target_weight_t = torch.ones_like(
                        visibility_target_t
                    ).float()
                else:
                    visibility_target_weight_t = torch.as_tensor(
                        visibility_target_weight,
                        dtype=torch.float32,
                    )
                return {
                    "image": img,
                    "target": lmk,
                    "heatmap": heatmap,
                    "sample_weight": torch.tensor(
                        sample["sample_weight"], dtype=torch.float32
                    ),
                    "landmark_mask": landmark_mask_t,
                    "visibility_target": visibility_target_t,
                    "visibility_target_weight": visibility_target_weight_t,
                    "visibility_target_provenance": visibility_target_source,
                    "schema": sample["source_schema"],
                    "head_name": sample["head_name"],
                    "metadata": metadata,
                }
            if self.include_metadata:
                metadata = dict(sample.get("metadata", {}))
                metadata.update(
                    {
                        "sample_id": sample.get("sample_id", ""),
                        "dataset": sample.get("dataset", ""),
                        "condition": sample.get("condition", ""),
                        "conditions": list(sample.get("conditions", ())),
                        "source_schema": sample.get("source_schema", ""),
                        "target_schema": sample.get("target_schema", ""),
                        "landmark_count": sample.get("landmark_count", 0),
                        "head_name": sample.get("head_name", ""),
                        "split": sample.get("split", ""),
                        "split_safe_id": sample.get("split_safe_id", ""),
                        "face_bbox": sample.get("face_bbox"),
                        "bbox_format": sample.get("bbox_format", ""),
                        "visibility_target": visibility_target.tolist()
                        if hasattr(visibility_target, "tolist")
                        else visibility_target,
                        "visibility_target_source": visibility_target_source,
                        "hard_negative_bucket": metadata.get(
                            "hard_negative_bucket", ""
                        ),
                    }
                )
                return (
                    img,
                    lmk,
                    heatmap,
                    torch.tensor(sample["sample_weight"], dtype=torch.float32),
                    landmark_mask_t,
                    metadata,
                )
            return (
                img,
                lmk,
                heatmap,
                torch.tensor(sample["sample_weight"], dtype=torch.float32),
                landmark_mask_t,
            )

        if self.include_metadata:
            metadata = dict(sample.get("metadata", {}))
            metadata.update(
                {
                    "sample_id": sample.get("sample_id", ""),
                    "image": sample.get("image", ""),
                    "landmarks": sample.get("landmarks", ""),
                    "dataset": sample.get("dataset", ""),
                    "condition": sample.get("condition", ""),
                    "conditions": list(sample.get("conditions", ())),
                    "source_schema": sample.get("source_schema", ""),
                    "target_schema": sample.get("target_schema", ""),
                    "landmark_count": sample.get("landmark_count", 0),
                    "head_name": sample.get("head_name", ""),
                    "split": sample.get("split", ""),
                    "split_safe_id": sample.get("split_safe_id", ""),
                    "source": sample.get("source", {}),
                    "face_bbox": sample.get("face_bbox"),
                    "bbox_format": sample.get("bbox_format", ""),
                    "visibility_target": sample.get("visibility_target").tolist()
                    if sample.get("visibility_target") is not None
                    else None,
                    "visibility_target_source": sample.get(
                        "visibility_target_source", ""
                    ),
                    "hard_negative_bucket": metadata.get("hard_negative_bucket", ""),
                }
            )
            return img, lmk, landmark_mask_t, metadata

        return img, lmk, landmark_mask_t
