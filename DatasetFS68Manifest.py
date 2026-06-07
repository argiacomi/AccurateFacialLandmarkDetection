import json
import os.path
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from DrawHeatmap import GenerateHeatmap
from RandomFlip import flip_points, random_flip
from lib.landmarks.core.schema import (
    canonicalize_schema,
    flip_map_for_schema,
    head_name_for_schema,
    infer_schema,
    normalize_landmark_array,
)
from lib.landmarks.evaluation.split_safe import (
    entry_in_eval_split,
    manifest_entry_split,
    normalize_heldout_datasets,
)

try:
    from ImageAugmentation import GetAugTransform
except ModuleNotFoundError:
    def GetAugTransform():
        raise ModuleNotFoundError("albumentations is required when FS68Manifest aug=True")


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


def _resolve_path(base_dir, value):
    path = Path(str(value or ""))
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _clamp_weight(value):
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return DEFAULT_HARD_NEGATIVE_WEIGHT
    if not np.isfinite(weight) or weight <= 0.0:
        return DEFAULT_HARD_NEGATIVE_WEIGHT
    return float(min(max(weight, DEFAULT_HARD_NEGATIVE_WEIGHT), MAX_HARD_NEGATIVE_WEIGHT))


def _weight_from_entry(entry, metadata, conditions):
    if "hard_negative_weight" in metadata:
        return _clamp_weight(metadata.get("hard_negative_weight"))

    bucket = _normalize_label(metadata.get("hard_negative_bucket"))
    if bucket in HARD_NEGATIVE_BUCKET_WEIGHTS:
        return HARD_NEGATIVE_BUCKET_WEIGHTS[bucket]

    labels = set(conditions)
    is_profile = any("profile" in label or "large_yaw" in label or label.startswith("yaw_") for label in labels)
    is_occlusion = any("occlusion" in label or "occluded" in label or "occlud" in label for label in labels)
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


def _as_bool_landmark_mask(value, landmark_count=68):
    if value is None:
        return None
    if isinstance(value, dict):
        # Accept dicts keyed by landmark index.
        arr = [value.get(str(i), value.get(i, True)) for i in range(int(landmark_count))]
    else:
        arr = value
    if isinstance(arr, np.ndarray):
        arr = arr.tolist()
    if not isinstance(arr, (list, tuple)) or len(arr) != int(landmark_count):
        return None

    out = []
    for item in arr:
        if isinstance(item, str):
            label = _normalize_label(item)
            out.append(label not in {"", "0", "false", "none", "invalid", "missing", "self_occluded", "selfoccluded"})
        else:
            out.append(bool(item))
    return np.asarray(out, dtype=np.float32)


def _landmark_mask_from_entry(entry, metadata, landmark_count=68):
    # Priority matters. For MERL-RAV, coordinate-valid includes visible plus externally
    # occluded estimated points, and excludes only true no-coordinate self-occlusion.
    for key in (
        "landmark_mask",
        "landmark_coordinate_valid_mask",
        "landmark_source_valid_mask",
        "landmark_in_image_mask",
        "coordinate_valid_mask",
        "source_valid_mask",
        "valid_mask",
    ):
        mask = _as_bool_landmark_mask(entry.get(key), landmark_count)
        if mask is not None:
            return mask
        mask = _as_bool_landmark_mask(metadata.get(key), landmark_count)
        if mask is not None:
            return mask

    # Lower priority: visibility often means score-visible only, which would drop
    # externally occluded but coordinate-valid MERL-RAV points.
    for key in ("visibility", "landmark_score_visibility_mask", "score_visibility_mask"):
        mask = _as_bool_landmark_mask(entry.get(key), landmark_count)
        if mask is not None:
            return mask
        mask = _as_bool_landmark_mask(metadata.get(key), landmark_count)
        if mask is not None:
            return mask

    return np.ones((int(landmark_count),), dtype=np.float32)


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
            out.append(1 if bool(item) else 0)
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
            return target
        target = _as_visibility_target(metadata.get(key), landmark_count)
        if target is not None:
            return target
    return None


class LandmarkDataset(Dataset):
    """Faceswap-compatible 68-point landmark manifest dataset.

    Expected manifest schema:
      {"samples": [{"image": "...", "landmarks": "...", "metadata": {...}}]}

    Landmarks must be .npy arrays with shape (68, 2) in pixel coordinates. Arrays
    in [0, 1] are treated as normalized and scaled to the 256x256 CD-ViT crop.
    faceswap hard-negative metadata is preserved as a per-sample loss weight.
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
            raise ValueError("FS68Manifest does not support perturbation mode")
        if not manifest_path:
            raise ValueError("FS68Manifest requires --manifest, --train_manifest, or --test_manifest")

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
            raise ValueError(f"no 68-point samples found in {self.manifest_path} for{detail}")

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )
        self.aug_transform = GetAugTransform() if aug else None
        self.generateHM = GenerateHeatmap(self.heatmap_size) if self.heatmap_size > 0 else None
        self.data_list = self.loaditem_list() if preload else None

    def _load_manifest(self, manifest_path, split, eval_mode, heldout_datasets, split_policy):
        base_dir = manifest_path.parent
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = payload.get("samples", payload.get("scenarios", []))
        if not isinstance(entries, list):
            raise ValueError(f"manifest {manifest_path} must contain a samples or scenarios list")

        declared_splits = {_entry_split(entry) for entry in entries if isinstance(entry, dict) and _entry_split(entry)}
        use_split_filter = bool(declared_splits)

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

            metadata = entry.get("metadata", {}) if isinstance(entry.get("metadata"), dict) else {}
            source = entry.get("source", {}) if isinstance(entry.get("source"), dict) else {}
            landmarks_value = entry.get("landmarks") or entry.get("ground_truth")
            image_value = entry.get("image")
            if not landmarks_value or not image_value:
                continue

            landmarks_path = _resolve_path(base_dir, landmarks_value)
            try:
                landmarks = np.load(landmarks_path)
            except OSError:
                raise FileNotFoundError(f"could not read landmarks for manifest entry {index}: {landmarks_path}")
            try:
                raw_schema = str(metadata.get("source_schema") or entry.get("source_schema") or "")
                detected_schema = infer_schema(np.asarray(landmarks)[:, :2])
                declared_schema = canonicalize_schema(raw_schema) if raw_schema else detected_schema
                schema = declared_schema if landmarks.shape[:2] == (68, 2) and declared_schema == "2d_68" else detected_schema
                if declared_schema != detected_schema and detected_schema != "2d_68":
                    schema = declared_schema
                landmarks = normalize_landmark_array(landmarks[:, :2], schema=schema)
                head_name = head_name_for_schema(schema)
            except ValueError:
                skipped_non_trainable_schema += 1
                continue

            if not self.schema_aware_training and schema != "2d_68":
                skipped_non_trainable_schema += 1
                continue

            landmark_mask = _landmark_mask_from_entry(entry, metadata, landmarks.shape[0])
            visibility_target = _visibility_target_from_entry(entry, metadata, landmarks.shape[0])
            conditions = _coerce_conditions(entry, metadata)
            samples.append(
                {
                    "sample_id": str(entry.get("sample_id") or entry.get("id") or entry.get("name") or index),
                    "image": _resolve_path(base_dir, image_value),
                    "landmarks": landmarks_path,
                    "dataset": str(entry.get("dataset") or metadata.get("dataset") or ""),
                    "condition": str(entry.get("condition") or entry.get("scenario") or ""),
                    "conditions": conditions,
                    "source_schema": schema,
                    "head_name": head_name,
                    "source": source,
                    "metadata": metadata,
                    "face_bbox": entry.get("face_bbox", metadata.get("face_bbox", entry.get("bbox", metadata.get("bbox")))),
                    "bbox_format": entry.get("bbox_format", metadata.get("bbox_format", "")),
                    "visibility_target": visibility_target,
                    "sample_weight": _weight_from_entry(entry, metadata, conditions),
                    "landmark_mask": landmark_mask,
                }
            )

        if skipped_non_trainable_schema:
            reason = "non-trainable" if self.schema_aware_training else "non-68-point"
            print(f"FS68Manifest skipped {skipped_non_trainable_schema} {reason} sample(s) from {manifest_path}")
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_image_and_landmarks(self, sample):
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

        valid_lmk = lmk[valid]
        lt = np.min(valid_lmk, axis=0)
        rb = np.max(valid_lmk, axis=0)
        padding = 0
        margin = 5
        if lt[0] < margin:
            padding = margin - lt[0]
        if lt[1] < margin:
            padding = max(margin - lt[1], padding)
        if rb[0] > img.shape[1] - margin:
            padding = max(padding, rb[0] - img.shape[1] + margin)
        if rb[1] > img.shape[0] - margin:
            padding = max(padding, rb[1] - img.shape[0] + margin)
        if padding > 0:
            padding = int(round(padding))
            new_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
            lmk = lmk + padding
            lmk = lmk * img.shape[0] / new_img.shape[0]
            new_img = cv2.resize(new_img, (img.shape[0], img.shape[1]))
            return new_img, lmk
        return img, lmk

    def __getitem__(self, item):
        sample = self.samples[item]
        if self.data_list is None:
            img, lmk = self._load_image_and_landmarks(sample)
            landmark_mask = sample["landmark_mask"].copy()
        else:
            img, lmk, landmark_mask = self.data_list[item]
            img = img.copy()
            lmk = lmk.copy()
            landmark_mask = landmark_mask.copy()

        if self.aug_transform is not None:
            transformed = self.aug_transform(image=img, keypoints=lmk)
            img = transformed["image"]
            lmk = np.array(transformed["keypoints"], dtype=np.float32)

            if np.random.random() < 0.5:
                if self.schema_aware_training:
                    flip_index = flip_map_for_schema(sample["source_schema"])
                else:
                    flip_index = np.asarray(flip_points("300W"), dtype=np.int64)
                img = cv2.flip(img, 1)
                lmk = lmk[flip_index, :]
                landmark_mask = landmark_mask[flip_index]
                lmk[:, 0] = 255 - lmk[:, 0]

        img, lmk = self.MakeLMKInsideImage(img, lmk, landmark_mask)
        img = self.transform(img)
        lmk = torch.from_numpy(lmk / 255.0).float()
        landmark_mask_t = torch.from_numpy(np.asarray(landmark_mask, dtype=np.float32)).float()

        if self.generateHM is not None:
            heatmap = self.generateHM.Generate(lmk * (self.heatmap_size - 1))
            heatmap = torch.from_numpy(heatmap).float()
            heatmap = heatmap * landmark_mask_t.reshape(-1, 1, 1)
            denom = torch.sum(heatmap, dim=(1, 2), keepdim=True).clamp_min(1e-6)
            heatmap = torch.where(landmark_mask_t.reshape(-1, 1, 1) > 0.0, heatmap / denom, heatmap)
            if self.schema_aware_training:
                metadata = dict(sample.get("metadata", {}))
                metadata.update(
                    {
                        "sample_id": sample.get("sample_id", ""),
                        "dataset": sample.get("dataset", ""),
                        "condition": sample.get("condition", ""),
                        "conditions": list(sample.get("conditions", ())),
                        "source_schema": sample.get("source_schema", ""),
                        "head_name": sample.get("head_name", ""),
                        "face_bbox": sample.get("face_bbox"),
                        "bbox_format": sample.get("bbox_format", ""),
                        "visibility_target": sample.get("visibility_target").tolist()
                        if sample.get("visibility_target") is not None
                        else None,
                        "hard_negative_bucket": metadata.get("hard_negative_bucket", ""),
                    }
                )
                return {
                    "image": img,
                    "target": lmk,
                    "heatmap": heatmap,
                    "sample_weight": torch.tensor(sample["sample_weight"], dtype=torch.float32),
                    "landmark_mask": landmark_mask_t,
                    "schema": sample["source_schema"],
                    "head_name": sample["head_name"],
                    "metadata": metadata,
                }
            return img, lmk, heatmap, torch.tensor(sample["sample_weight"], dtype=torch.float32), landmark_mask_t

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
                    "source": sample.get("source", {}),
                    "face_bbox": sample.get("face_bbox"),
                    "bbox_format": sample.get("bbox_format", ""),
                    "visibility_target": sample.get("visibility_target").tolist()
                    if sample.get("visibility_target") is not None
                    else None,
                    "hard_negative_bucket": metadata.get("hard_negative_bucket", ""),
                }
            )
            return img, lmk, landmark_mask_t, metadata

        return img, lmk, landmark_mask_t
