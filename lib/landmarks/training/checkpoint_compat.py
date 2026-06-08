"""Shared checkpoint compatibility logic for trainer and pipeline."""

from __future__ import annotations

import hashlib
import json
import typing as T
from pathlib import Path


INT_COMPAT_KEYS = {
    "batch_size",
    "heatmap_size",
    "lmk_num",
    "sched_step",
    "nstack",
    "max_depth",
    "seed",
}

FLOAT_COMPAT_KEYS = {
    "lr",
    "hw",
    "locw",
    "mul",
    "schema_consistency_weight",
    "star_loss_weight",
    "auxiliary_loss_weight",
}

BOOL_COMPAT_KEYS = {
    "auto_dataset_balancing",
    "auto_schema_balancing",
    "schema_aware_training",
    "domain_balanced_sampling",
    "auxiliary_heads",
}

STRING_COMPAT_KEYS = {
    "data_name",
    "eval_mode",
    "split_policy",
    "bucket_targets",
    "dataset_targets",
    "schema_targets",
    "schema_head_loss_weighting",
    "schema_head_loss_weights",
}


def normalize_path_for_compat(value: T.Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return str(Path(value).expanduser().resolve())
    except OSError:
        return str(Path(value).expanduser())


def file_sha256_or_none(value: T.Any) -> str | None:
    if value in (None, ""):
        return None
    path = Path(value)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_manifest_path_for_compat(args: T.Any) -> T.Any:
    return (
        getattr(args, "train_manifest", "")
        or getattr(args, "manifest", "")
        or getattr(args, "root_folder", "")
    )


def build_training_compat_config_from_args(args: T.Any) -> dict[str, T.Any]:
    config: dict[str, T.Any] = {
        "manifest_sha256": file_sha256_or_none(training_manifest_path_for_compat(args)),
        "train_manifest_sha256": file_sha256_or_none(
            getattr(args, "train_manifest", "")
        ),
        "test_manifest_sha256": file_sha256_or_none(getattr(args, "test_manifest", "")),
    }

    for key in sorted(INT_COMPAT_KEYS):
        try:
            config[key] = int(getattr(args, key, None))
        except (TypeError, ValueError):
            config[key] = getattr(args, key, None)

    for key in sorted(FLOAT_COMPAT_KEYS):
        try:
            config[key] = float(getattr(args, key, None))
        except (TypeError, ValueError):
            config[key] = getattr(args, key, None)

    for key in sorted(BOOL_COMPAT_KEYS):
        config[key] = bool(getattr(args, key, False))

    for key in sorted(STRING_COMPAT_KEYS):
        config[key] = str(getattr(args, key, ""))

    config["heldout_dataset"] = [
        str(item) for item in list(getattr(args, "heldout_dataset", []) or [])
    ]
    return config


def build_pipeline_training_compat_config(
    args: T.Any,
    paths: T.Any,
    *,
    train_arg_option: T.Callable[..., T.Any],
    train_bool_arg: T.Callable[..., bool],
    train_arg_values: T.Callable[..., list[T.Any]],
    effective_training_manifest_for_compat: T.Callable[[T.Any, T.Any], T.Any],
    safe_sha256_file: T.Callable[[Path], str | None],
) -> dict[str, T.Any]:
    """Build the same checkpoint contract for pipeline invocations.

    The pipeline appends --train-arg values after generated trainer arguments, so
    the helper receives pipeline parsing callbacks and applies those override
    semantics here, in the shared compatibility module.
    """

    def int_opt(default: int, *names: str) -> int:
        return int(train_arg_option(args, *names, default=default))

    def float_opt(default: float, *names: str) -> float:
        return float(train_arg_option(args, *names, default=default))

    def str_opt(default: str, *names: str) -> str:
        return str(train_arg_option(args, *names, default=default))

    data_name = str_opt(str(args.train_data_name), "--data_name", "--data-name")
    split_policy = str_opt("declared_or_random_hash", "--split-policy")
    if train_bool_arg(args, "--respect-declared-splits", default=False):
        split_policy = "declared"
    if train_bool_arg(args, "--ignore-declared-splits", default=False):
        split_policy = "random_hash"

    return {
        "manifest_sha256": safe_sha256_file(
            Path(effective_training_manifest_for_compat(args, paths))
        ),
        "train_manifest_sha256": safe_sha256_file(
            Path(
                train_arg_option(
                    args, "--train_manifest", "--train-manifest", default=""
                )
            )
        ),
        "test_manifest_sha256": safe_sha256_file(
            Path(
                train_arg_option(args, "--test_manifest", "--test-manifest", default="")
            )
        ),
        "batch_size": int_opt(int(args.batch_size), "--batch_size", "--batch-size"),
        "heatmap_size": int_opt(
            int(args.heatmap_size), "--heatmap_size", "--heatmap-size"
        ),
        "lmk_num": int_opt(int(args.lmk_num), "--lmk_num", "--lmk-num"),
        "sched_step": int_opt(200, "--sched_step", "--sched-step"),
        "nstack": int_opt(8, "--nstack"),
        "max_depth": int_opt(256, "--max_depth", "--max-depth"),
        "seed": int_opt(0, "--seed"),
        "lr": float_opt(float(args.lr), "--lr"),
        "hw": float_opt(10.0, "--hw"),
        "locw": float_opt(1.0, "--locw"),
        "mul": float_opt(1.2, "--mul"),
        "schema_consistency_weight": float_opt(
            0.05,
            "--schema-consistency-weight",
            "--schema_consistency_weight",
        ),
        "star_loss_weight": float_opt(
            0.0,
            "--star-loss-weight",
            "--star_loss_weight",
        ),
        "auxiliary_loss_weight": float_opt(
            0.1,
            "--auxiliary-loss-weight",
            "--auxiliary_loss_weight",
        ),
        "schema_aware_training": train_bool_arg(
            args,
            "--schema-aware-training",
            "--no-schema-aware-training",
            default=True,
        ),
        "domain_balanced_sampling": train_bool_arg(
            args,
            "--domain-balanced-sampling",
            default=False,
        ),
        "auto_dataset_balancing": train_bool_arg(
            args,
            "--auto-dataset-balancing",
            "--no-auto-dataset-balancing",
            default=True,
        ),
        "auto_schema_balancing": train_bool_arg(
            args,
            "--auto-schema-balancing",
            "--no-auto-schema-balancing",
            default=True,
        ),
        "auxiliary_heads": train_bool_arg(
            args,
            "--auxiliary-heads",
            "--no-auxiliary-heads",
            default=True,
        ),
        "data_name": data_name,
        "eval_mode": str_opt("random_hash", "--eval-mode"),
        "split_policy": split_policy,
        "bucket_targets": str_opt(
            "anchor=0.25,occlusion=0.25,profile=0.25,profile_occlusion=0.25",
            "--bucket-targets",
        ),
        "dataset_targets": str_opt("", "--dataset-targets"),
        "schema_targets": str_opt("", "--schema-targets"),
        "schema_head_loss_weighting": str_opt(
            "sample_count",
            "--schema-head-loss-weighting",
            "--schema_head_loss_weighting",
        ),
        "schema_head_loss_weights": str_opt(
            "",
            "--schema-head-loss-weights",
            "--schema_head_loss_weights",
        ),
        "heldout_dataset": [
            str(item) for item in train_arg_values(args, "--heldout-dataset")
        ],
    }


def training_compat_digest_from_config(config: T.Mapping[str, T.Any]) -> str:
    payload = json.dumps(
        dict(config),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def training_compat_digest_from_args(args: T.Any) -> str:
    return training_compat_digest_from_config(
        build_training_compat_config_from_args(args)
    )


def checkpoint_compat_errors_for_config(
    checkpoint: T.Any,
    expected_config: T.Mapping[str, T.Any],
    *,
    current_manifest_sha: str | None = None,
    fallback_expected_args: T.Mapping[str, T.Any] | None = None,
    message_prefix: str = "checkpoint",
) -> list[str]:
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("format") != "cdvit-training-checkpoint-v1"
    ):
        return []

    errors: list[str] = []
    expected_config = dict(expected_config)
    actual_config = checkpoint.get("compat_config")

    if isinstance(actual_config, dict):
        comparable_actual = {key: actual_config.get(key) for key in expected_config}
        if comparable_actual != expected_config:
            errors.append(
                f"{message_prefix} training contract differs from the current invocation"
            )
    else:
        actual_digest = checkpoint.get("compat_config_digest")
        if actual_digest:
            if actual_digest != training_compat_digest_from_config(expected_config):
                errors.append(
                    f"{message_prefix} training contract digest differs from the current invocation"
                )
        else:
            saved_args = (
                checkpoint.get("args")
                if isinstance(checkpoint.get("args"), dict)
                else {}
            )
            fallback = dict(fallback_expected_args or {})
            critical_keys = {
                "data_name": str(
                    fallback.get("data_name", expected_config.get("data_name", ""))
                ),
                "batch_size": int(
                    fallback.get("batch_size", expected_config.get("batch_size", 0))
                ),
                "heatmap_size": int(
                    fallback.get("heatmap_size", expected_config.get("heatmap_size", 0))
                ),
                "lmk_num": int(
                    fallback.get("lmk_num", expected_config.get("lmk_num", 0))
                ),
                "lr": float(fallback.get("lr", expected_config.get("lr", 0.0))),
                "schema_aware_training": bool(
                    fallback.get(
                        "schema_aware_training",
                        expected_config.get("schema_aware_training", False),
                    )
                ),
                "domain_balanced_sampling": bool(
                    fallback.get(
                        "domain_balanced_sampling",
                        expected_config.get("domain_balanced_sampling", False),
                    )
                ),
                "auxiliary_heads": bool(
                    fallback.get(
                        "auxiliary_heads", expected_config.get("auxiliary_heads", False)
                    )
                ),
            }
            for key, expected in critical_keys.items():
                if key not in saved_args:
                    continue
                actual = saved_args[key]
                try:
                    if isinstance(expected, bool):
                        matches = bool(actual) == expected
                    elif isinstance(expected, int):
                        matches = int(actual) == expected
                    elif isinstance(expected, float):
                        matches = float(actual) == expected
                    else:
                        matches = str(actual) == str(expected)
                except (TypeError, ValueError):
                    matches = False
                if not matches:
                    errors.append(
                        f"checkpoint arg {key!r} differs: checkpoint={actual!r}, current={expected!r}"
                    )

    checkpoint_manifest_sha = checkpoint.get("manifest_sha256")
    if (
        current_manifest_sha
        and checkpoint_manifest_sha
        and current_manifest_sha != checkpoint_manifest_sha
    ):
        errors.append("checkpoint manifest SHA differs from the current manifest")

    return errors


def checkpoint_compat_errors_from_args(checkpoint: T.Any, args: T.Any) -> list[str]:
    return checkpoint_compat_errors_for_config(
        checkpoint,
        build_training_compat_config_from_args(args),
        current_manifest_sha=file_sha256_or_none(
            training_manifest_path_for_compat(args)
        ),
        fallback_expected_args=vars(args) if hasattr(args, "__dict__") else {},
    )
