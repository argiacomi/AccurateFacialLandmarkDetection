"""Domain-balanced batch sampling for schema-aware landmark training."""

from __future__ import annotations

import math
import random
import typing as T

from torch.utils.data import Sampler

from lib.landmarks.evaluation.split_safe import normalize_dataset


DEFAULT_BUCKET_TARGETS = {
    "anchor": 0.25,
    "occlusion": 0.25,
    "profile": 0.25,
    "profile_occlusion": 0.25,
}


BUCKET_ALIASES = {
    "normal": "anchor",
    "clean": "anchor",
    "frontal": "anchor",
    "large_yaw": "profile",
    "large_yaw_pose": "profile",
    "profile_pose": "profile",
    "rolled_profile_occlusion": "profile_occlusion",
    "large_yaw_occlusion": "profile_occlusion",
    "occluded": "occlusion",
    "single_eye_visible": "occlusion",
    "mouth_or_jaw_occluded": "occlusion",
}


def normalize_label(value: T.Any) -> str:
    label = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in label:
        label = label.replace("__", "_")
    return label.strip("_") or "unknown"


def canonical_bucket(value: T.Any) -> str:
    label = normalize_label(value)
    if label in BUCKET_ALIASES:
        return BUCKET_ALIASES[label]
    if label.startswith("yaw_"):
        return "profile"
    is_profile = "profile" in label or "large_yaw" in label
    is_occlusion = "occlusion" in label or "occluded" in label or "occlud" in label
    if is_profile and is_occlusion:
        return "profile_occlusion"
    if is_profile:
        return "profile"
    if is_occlusion:
        return "occlusion"
    if label in DEFAULT_BUCKET_TARGETS:
        return label
    return label


def normalize_target_key(value: T.Any, *, kind: str) -> str:
    if kind == "bucket":
        return canonical_bucket(value)
    if kind == "dataset":
        return normalize_dataset(value) or "unknown"
    return normalize_label(value)


def parse_target_spec(
    value: str | None, defaults: dict[str, float] | None = None
) -> dict[str, float]:
    return parse_target_spec_for_kind(value, defaults, kind="bucket")


def parse_target_spec_for_kind(
    value: str | None,
    defaults: dict[str, float] | None = None,
    *,
    kind: str,
) -> dict[str, float]:
    if not value:
        return dict(defaults or {})
    out: dict[str, float] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"target item {item!r} must be key=value")
        key, raw_value = item.split("=", 1)
        target_key = normalize_target_key(key, kind=kind)
        try:
            amount = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"target for {target_key!r} has invalid value {raw_value!r}"
            ) from exc
        if not math.isfinite(amount) or amount < 0:
            raise ValueError(
                f"target for {target_key!r} must be a finite non-negative value, got {raw_value!r}"
            )
        out[target_key] = amount
    return out


def _metadata(sample: T.Mapping[str, T.Any]) -> T.Mapping[str, T.Any]:
    metadata = sample.get("metadata")
    return metadata if isinstance(metadata, T.Mapping) else {}


def sample_bucket(sample: T.Mapping[str, T.Any]) -> str:
    metadata = _metadata(sample)
    bucket = canonical_bucket(
        metadata.get("hard_negative_bucket") or sample.get("condition")
    )
    conditions = sample.get("conditions") or metadata.get("conditions") or ()
    if isinstance(conditions, str):
        conditions = (conditions,)
    labels = {canonical_bucket(item) for item in conditions}
    if bucket != "unknown":
        return bucket
    if "profile" in labels and "occlusion" in labels:
        return "profile_occlusion"
    if "profile" in labels:
        return "profile"
    if "occlusion" in labels:
        return "occlusion"
    return "anchor"


def sample_dataset(sample: T.Mapping[str, T.Any]) -> str:
    metadata = _metadata(sample)
    return (
        normalize_dataset(sample.get("dataset") or metadata.get("dataset")) or "unknown"
    )


def sample_schema(sample: T.Mapping[str, T.Any]) -> str:
    metadata = _metadata(sample)
    return normalize_label(
        sample.get("source_schema")
        or metadata.get("source_schema")
        or sample.get("head_name")
    )


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
    order = sorted(
        raw, key=lambda key: (raw[key] - counts[key], raw[key]), reverse=True
    )
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
        auto_balance_datasets: bool = True,
        auto_balance_schemas: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.samples = samples
        self.bucket_targets = _normalize_targets(
            bucket_targets or DEFAULT_BUCKET_TARGETS, kind="bucket"
        )
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = max(int(world_size), 1)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self._groups = self._build_groups(samples)
        self.dataset_targets = _resolved_targets(
            dataset_targets,
            observed={key[1] for key in self._groups},
            kind="dataset",
            auto_balance=auto_balance_datasets,
        )
        self.schema_targets = _resolved_targets(
            schema_targets,
            observed={key[2] for key in self._groups},
            kind="schema",
            auto_balance=auto_balance_schemas,
        )
        self.last_epoch_diagnostics = self._empty_diagnostics()

    def _build_groups(
        self, samples: T.Sequence[T.Mapping[str, T.Any]]
    ) -> dict[tuple[str, str, str], list[int]]:
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
        return self._per_rank_batches()

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        pools = {
            key: rng.sample(indices, len(indices))
            for key, indices in self._groups.items()
        }
        per_rank_batches = self._per_rank_batches()
        global_batches = per_rank_batches * self.world_size
        bucket_plan = integer_targets(self.bucket_targets, self.batch_size)
        dataset_plan = (
            integer_targets(self.dataset_targets, self.batch_size)
            if self.dataset_targets
            else []
        )
        schema_plan = (
            integer_targets(self.schema_targets, self.batch_size)
            if self.schema_targets
            else []
        )
        diagnostics = self._empty_diagnostics()

        for batch_index in range(global_batches):
            batch = []
            rng.shuffle(bucket_plan)
            if dataset_plan:
                rng.shuffle(dataset_plan)
            if schema_plan:
                rng.shuffle(schema_plan)
            for slot, bucket in enumerate(bucket_plan):
                dataset = (
                    dataset_plan[slot % len(dataset_plan)] if dataset_plan else None
                )
                schema = schema_plan[slot % len(schema_plan)] if schema_plan else None
                index, fallback = self._draw_index(
                    pools, rng, bucket=bucket, dataset=dataset, schema=schema
                )
                batch.append(index)
                if batch_index % self.world_size == self.rank:
                    diagnostics["fallback_counts"][fallback] = (
                        int(diagnostics["fallback_counts"].get(fallback, 0)) + 1
                    )
                    self._increment_mix(diagnostics, self.samples[index])
            if batch_index % self.world_size == self.rank:
                self.last_epoch_diagnostics = diagnostics
                yield batch
        self.last_epoch_diagnostics = diagnostics

    def _per_rank_batches(self) -> int:
        total_batches = (
            len(self.samples) // self.batch_size
            if self.drop_last
            else math.ceil(len(self.samples) / self.batch_size)
        )
        if self.drop_last:
            return total_batches // self.world_size
        return math.ceil(total_batches / self.world_size)

    def resolved_targets(self) -> dict[str, dict[str, float]]:
        return {
            "bucket": dict(self.bucket_targets),
            "dataset": dict(self.dataset_targets),
            "schema": dict(self.schema_targets),
        }

    def _empty_diagnostics(self) -> dict[str, T.Any]:
        return {
            "requested_targets": self.resolved_targets(),
            "actual_mix": {"bucket": {}, "dataset": {}, "schema": {}},
            "fallback_counts": {"exact": 0, "exact_to_bucket": 0, "bucket_to_any": 0},
            "missing_targets": {
                "bucket": sorted(
                    _missing_targets(
                        self.bucket_targets, {key[0] for key in self._groups}
                    )
                ),
                "dataset": sorted(
                    _missing_targets(
                        self.dataset_targets, {key[1] for key in self._groups}
                    )
                ),
                "schema": sorted(
                    _missing_targets(
                        self.schema_targets, {key[2] for key in self._groups}
                    )
                ),
            },
            "rank": int(self.rank),
            "world_size": int(self.world_size),
            "batches_per_rank": int(self._per_rank_batches()),
        }

    def _increment_mix(
        self, diagnostics: dict[str, T.Any], sample: T.Mapping[str, T.Any]
    ) -> None:
        for key, label in (
            ("bucket", sample_bucket(sample)),
            ("dataset", sample_dataset(sample)),
            ("schema", sample_schema(sample)),
        ):
            mix = diagnostics["actual_mix"][key]
            mix[label] = int(mix.get(label, 0)) + 1

    def _draw_index(
        self,
        pools: dict[tuple[str, str, str], list[int]],
        rng: random.Random,
        *,
        bucket: str,
        dataset: str | None,
        schema: str | None,
    ) -> tuple[int, str]:
        candidates = [
            key
            for key in pools
            if (bucket == "*" or key[0] == bucket)
            and (dataset is None or dataset == "*" or key[1] == dataset)
            and (schema is None or schema == "*" or key[2] == schema)
        ]
        fallback = "exact"
        if not candidates:
            candidates = [key for key in pools if bucket == "*" or key[0] == bucket]
            fallback = "exact_to_bucket"
        if not candidates:
            candidates = list(pools)
            fallback = "bucket_to_any"
        key = rng.choice(candidates)
        if not pools[key]:
            pools[key] = rng.sample(self._groups[key], len(self._groups[key]))
        return pools[key].pop(), fallback


def _normalize_targets(
    targets: T.Mapping[str, float], *, kind: str
) -> dict[str, float]:
    return {
        normalize_target_key(key, kind=kind): float(value)
        for key, value in targets.items()
    }


def _resolved_targets(
    targets: dict[str, float] | None,
    *,
    observed: set[str],
    kind: str,
    auto_balance: bool,
) -> dict[str, float]:
    if targets:
        return _normalize_targets(targets, kind=kind)
    if not auto_balance:
        return {}
    labels = sorted(label for label in observed if label and label != "unknown")
    return {label: 1.0 for label in labels}


def _missing_targets(targets: T.Mapping[str, float], observed: set[str]) -> set[str]:
    return {
        key
        for key, value in targets.items()
        if key != "*" and float(value) > 0.0 and key not in observed
    }


__all__ = [
    "DEFAULT_BUCKET_TARGETS",
    "DomainBalancedBatchSampler",
    "canonical_bucket",
    "integer_targets",
    "parse_target_spec",
    "parse_target_spec_for_kind",
    "sample_bucket",
    "sample_dataset",
    "sample_schema",
]
