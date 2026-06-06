import json
import os.path
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from DrawHeatmap import GenerateHeatmap
from ImageAugmentation import GetAugTransform
from RandomFlip import flip_points, random_flip


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
    raw = entry.get("split")
    if raw is None and isinstance(entry.get("metadata"), dict):
        raw = entry["metadata"].get("split")
    return _normalize_label(raw)


def _as_bool_landmark_mask(value):
    if value is None:
        return None
    if isinstance(value, dict):
        # Accept dicts keyed by landmark index.
        arr = [value.get(str(i), value.get(i, True)) for i in range(68)]
    else:
        arr = value
    if isinstance(arr, np.ndarray):
        arr = arr.tolist()
    if not isinstance(arr, (list, tuple)) or len(arr) != 68:
        return None

    out = []
    for item in arr:
        if isinstance(item, str):
            label = _normalize_label(item)
            out.append(label not in {"", "0", "false", "none", "invalid", "missing", "self_occluded", "selfoccluded"})
        else:
            out.append(bool(item))
    return np.asarray(out, dtype=np.float32)


def _landmark_mask_from_entry(entry, metadata):
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
        mask = _as_bool_landmark_mask(entry.get(key))
        if mask is not None:
            return mask
        mask = _as_bool_landmark_mask(metadata.get(key))
        if mask is not None:
            return mask

    # Lower priority: visibility often means score-visible only, which would drop
    # externally occluded but coordinate-valid MERL-RAV points.
    for key in ("visibility", "landmark_score_visibility_mask", "score_visibility_mask"):
        mask = _as_bool_landmark_mask(entry.get(key))
        if mask is not None:
            return mask
        mask = _as_bool_landmark_mask(metadata.get(key))
        if mask is not None:
            return mask

    return np.ones((68,), dtype=np.float32)


class LandmarkDataset(Dataset):
    """Faceswap-compatible 68-point landmark manifest dataset.

    Expected manifest schema:
      {"samples": [{"image": "...", "landmarks": "...", "metadata": {...}}]}

    Landmarks must be .npy arrays with shape (68, 2) in pixel coordinates. Arrays
    in [0, 1] are treated as normalized and scaled to the 256x256 CD-ViT crop.
    faceswap hard-negative metadata is preserved as a per-sample loss weight.
    """

    def __init__(self, manifest_path, split="train", preload=True, aug=True, heatmap_size=0, perturbation=0):
        super(LandmarkDataset, self).__init__()
        if perturbation:
            raise ValueError("FS68Manifest does not support perturbation mode")
        if not manifest_path:
            raise ValueError("FS68Manifest requires --manifest, --train_manifest, or --test_manifest")

        self.manifest_path = Path(manifest_path)
        self.split = split
        self.heatmap_size = int(heatmap_size or 0)
        self.samples = self._load_manifest(self.manifest_path, split)
        if not self.samples:
            raise ValueError(f"no 68-point samples found in {self.manifest_path} for split {split!r}")

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )
        self.aug_transform = GetAugTransform() if aug else None
        self.generateHM = GenerateHeatmap(self.heatmap_size) if self.heatmap_size > 0 else None
        self.data_list = self.loaditem_list() if preload else None

    def _load_manifest(self, manifest_path, split):
        base_dir = manifest_path.parent
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = payload.get("samples", payload.get("scenarios", []))
        if not isinstance(entries, list):
            raise ValueError(f"manifest {manifest_path} must contain a samples or scenarios list")

        split_label = _normalize_label(split)
        declared_splits = {_entry_split(entry) for entry in entries if isinstance(entry, dict) and _entry_split(entry)}
        use_split_filter = bool(declared_splits)

        samples = []
        skipped_non_68 = 0
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            entry_split = _entry_split(entry)
            if use_split_filter:
                if not entry_split:
                    continue
                if entry_split != split_label:
                    continue

            metadata = entry.get("metadata", {}) if isinstance(entry.get("metadata"), dict) else {}
            landmarks_value = entry.get("landmarks") or entry.get("ground_truth")
            image_value = entry.get("image")
            if not landmarks_value or not image_value:
                continue

            landmarks_path = _resolve_path(base_dir, landmarks_value)
            try:
                landmarks = np.load(landmarks_path)
            except OSError:
                raise FileNotFoundError(f"could not read landmarks for manifest entry {index}: {landmarks_path}")
            if getattr(landmarks, "ndim", 0) != 2 or int(landmarks.shape[0]) != 68 or int(landmarks.shape[1]) < 2:
                skipped_non_68 += 1
                continue

            landmark_mask = _landmark_mask_from_entry(entry, metadata)
            conditions = _coerce_conditions(entry, metadata)
            samples.append(
                {
                    "sample_id": str(entry.get("sample_id") or entry.get("id") or entry.get("name") or index),
                    "image": _resolve_path(base_dir, image_value),
                    "landmarks": landmarks_path,
                    "dataset": str(entry.get("dataset") or metadata.get("dataset") or ""),
                    "condition": str(entry.get("condition") or entry.get("scenario") or ""),
                    "conditions": conditions,
                    "metadata": metadata,
                    "sample_weight": _weight_from_entry(entry, metadata, conditions),
                    "landmark_mask": landmark_mask,
                }
            )

        if skipped_non_68:
            print(f"FS68Manifest skipped {skipped_non_68} non-68-point sample(s) from {manifest_path}")
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
            return img, lmk, heatmap, torch.tensor(sample["sample_weight"], dtype=torch.float32), landmark_mask_t

        return img, lmk, landmark_mask_t
