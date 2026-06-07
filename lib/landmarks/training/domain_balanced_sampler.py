"""Domain-balanced batch sampling for schema-aware landmark training."""

from __future__ import annotations

import math
import random
import typing as T

from torch.utils.data import Sampler


DEFAULT_BUCKET_TARGETS = {
    "anchor": 0.25,
    "occlusion": 0.25,
    "profile": 0.25,
    "profile_occlusion": 0.25,
}


def normalize_label(value: T.Any) -> str:
    label = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in label:
        label = label.replace("__", "_")
    return label.strip("_") or "unknown"


def parse_target_spec(value: str | None, defaults: dict[str, float] | None = None) -> dict[str, float]:
    if not value:
        return dict(defaults or {})
    out: dict[str, float] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"target item {item!r} must be key=value")
        key, raw_value = item.split("=", 1)
        amount = float(raw_value)
        if amount < 0:
            raise ValueError(f"target for {key!r} must be non-negative")
        out[normalize_label(key)] = amount
    return out


def _metadata(sample: T.Mapping[str, T.Any]) -> T.Mapping[str, T.Any]:
    metadata = sample.get("metadata")
    return metadata if isinstance(metadata, T.Mapping) else {}


def sample_bucket(sample: T.Mapping[str, T.Any]) -> str:
    metadata = _metadata(sample)
    bucket = normalize_label(metadata.get("hard_negative_bucket") or sample.get("condition"))
    conditions = sample.get("conditions") or metadata.get("conditions") or ()
    if isinstance(conditions, str):
        conditions = (conditions,)
    labels = {normalize_label(item) for item in conditions}
    if bucket != "unknown":
        return bucket
    if "profile" in labels and any("occlusion" in label or "occlud" in label for label in labels):
        return "profile_occlusion"
    if "profile" in labels or "large_yaw" in labels:
        return "profile"
    if any("occlusion" in label or "occlud" in label for label in labels):
        return "occlusion"
    return "anchor"


def sample_dataset(sample: T.Mapping[str, T.Any]) -> str:
    metadata = _metadata(sample)
    return normalize_label(sample.get("dataset") or metadata.get("dataset"))


def sample_schema(sample: T.Mapping[str, T.Any]) -> str:
    metadata = _metadata(sample)
    return normalize_label(sample.get("source_schema") or metadata.get("source_schema") or sample.get("head_name"))


def integer_targets(targets: dict[str, float], batch_size: int) -> list[str]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not targets:
        return ["*"] * batch_size
    total = sum(float(value) for value in targets.values())
    if total <= 0:
        return ["*"] * batch_size
    raw = {key: batch_size * float(value) / total for key, value in targets.items()}
    counts = {key: int(math.floor(value)) for key, value in raw.items()}
    remaining = batch_size - sum(counts.values())
    order = sorted(raw, key=lambda key: (raw[key] - counts[key], raw[key]), reverse=True)
    for key in order[:remaining]:
        counts[key] += 1
    labels: list[str] = []
    for key, count in counts.items():
        labels.extend([key] * count)
    return labels[:batch_size]


class DomainBalancedBatchSampler(Sampler[list[int]]):
    """Yield batches balanced by hard bucket, dataset, and schema."""

    def __init__(
        self,
        samples: T.Sequence[T.Mapping[str, T.Any]],
        *,
        bucket_targets: dict[str, float] | None = None,
        dataset_targets: dict[str, float] | None = None,
        schema_targets: dict[str, float] | None = None,
        batch_size: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.samples = samples
        self.bucket_targets = bucket_targets or dict(DEFAULT_BUCKET_TARGETS)
        self.dataset_targets = dataset_targets or {}
        self.schema_targets = schema_targets or {}
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = max(int(world_size), 1)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self._groups = self._build_groups(samples)

    def _build_groups(self, samples):
        groups: dict[tuple[str, str, str], list[int]] = {}
        for index, sample in enumerate(samples):
            key = (sample_bucket(sample), sample_dataset(sample), sample_schema(sample))
            groups.setdefault(key, []).append(index)
        if not groups:
            raise ValueError("DomainBalancedBatchSampler requires at least one sample")
        return groups

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        total_batches = len(self.samples) // self.batch_size if self.drop_last else math.ceil(len(self.samples) / self.batch_size)
        return math.ceil(total_batches / self.world_size)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        pools = {key: rng.sample(indices, len(indices)) for key, indices in self._groups.items()}
        global_batches = len(self.samples) // self.batch_size if self.drop_last else math.ceil(len(self.samples) / self.batch_size)
        bucket_plan = integer_targets(self.bucket_targets, self.batch_size)
        dataset_plan = integer_targets(self.dataset_targets, self.batch_size) if self.dataset_targets else []
        schema_plan = integer_targets(self.schema_targets, self.batch_size) if self.schema_targets else []

        for batch_index in range(global_batches):
            batch = []
            rng.shuffle(bucket_plan)
            if dataset_plan:
                rng.shuffle(dataset_plan)
            if schema_plan:
                rng.shuffle(schema_plan)
            for slot, bucket in enumerate(bucket_plan):
                dataset = dataset_plan[slot % len(dataset_plan)] if dataset_plan else None
                schema = schema_plan[slot % len(schema_plan)] if schema_plan else None
                batch.append(self._draw_index(pools, rng, bucket=bucket, dataset=dataset, schema=schema))
            if batch_index % self.world_size == self.rank:
                yield batch

    def _draw_index(self, pools, rng, *, bucket: str, dataset: str | None, schema: str | None) -> int:
        candidates = [
            key
            for key in pools
            if (bucket == "*" or key[0] == bucket)
            and (dataset is None or dataset == "*" or key[1] == dataset)
            and (schema is None or schema == "*" or key[2] == schema)
        ]
        if not candidates:
            candidates = [key for key in pools if bucket == "*" or key[0] == bucket]
        if not candidates:
            candidates = list(pools)
        key = rng.choice(candidates)
        if not pools[key]:
            pools[key] = rng.sample(self._groups[key], len(self._groups[key]))
        return pools[key].pop()


__all__ = [
    "DEFAULT_BUCKET_TARGETS",
    "DomainBalancedBatchSampler",
    "integer_targets",
    "parse_target_spec",
    "sample_bucket",
    "sample_dataset",
    "sample_schema",
]
