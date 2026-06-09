"""Compatibility entrypoint for CD-ViT heatmap-stage training.

The trainer implementation lives in :mod:`lib.landmarks.training.heatmap_stage`.
Keep this file as the stable script path used by torchrun and as a legacy import
surface for tests/tools that still access private helper names from
``TrainHeatmapStageFP16`` directly.
"""

from __future__ import annotations

# BEGIN LEGACY_PRIVATE_HELPER_EXPORTS
# Prefer public imports from lib.landmarks.training.* for new code. These
# aliases preserve the historical TrainHeatmapStageFP16.py private-helper
# surface for older tests/tools and object-identity checks such as
# train._dataloader_kwargs(...)["worker_init_fn"] is train._seed_worker.
# END LEGACY_PRIVATE_HELPER_EXPORTS
from lib.landmarks.training import checkpoint_compat as _checkpoint_compat
from lib.landmarks.training import heatmap_stage as _heatmap_stage_impl
from lib.landmarks.training import runtime as _runtime

# BEGIN LEGACY_PRIVATE_HELPER_EXPORTS
# Preserve the historical TrainHeatmapStageFP16.py private-helper surface.
_dataloader_kwargs = _runtime.dataloader_kwargs
_maybe_limit_eval_dataset = _runtime.maybe_limit_eval_dataset
_normalize_runtime_args = _runtime.normalize_runtime_args
_seed_worker = _runtime.seed_worker
_set_dataset_runtime_epoch = _runtime.set_dataset_runtime_epoch
_should_run_interval = _runtime.should_run_interval

_training_compat_config = _checkpoint_compat.build_training_compat_config_from_args
_checkpoint_compat_errors = _checkpoint_compat.checkpoint_compat_errors_from_args
_file_sha256_or_none = _checkpoint_compat.file_sha256_or_none
_normalize_path_for_compat = _checkpoint_compat.normalize_path_for_compat
_training_compat_digest = _checkpoint_compat.training_compat_digest_from_args
_training_manifest_path_for_compat = (
    _checkpoint_compat.training_manifest_path_for_compat
)
# END LEGACY_PRIVATE_HELPER_EXPORTS

main = _heatmap_stage_impl.main


def __getattr__(name: str):
    """Delegate legacy module attributes to the modularized trainer.

    This preserves imports such as ``TrainHeatmapStageFP16._dataloader_kwargs``
    without duplicating helper imports in this wrapper.
    """

    try:
        return getattr(_heatmap_stage_impl, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_heatmap_stage_impl)))


if __name__ == "__main__":
    main()
