"""Compatibility entrypoint for CD-ViT heatmap-stage training.

The trainer implementation lives in :mod:`lib.landmarks.training.heatmap_stage`.
Keep this file as the stable script path used by torchrun and as a legacy import
surface for tests/tools that still access private helper names from
``TrainHeatmapStageFP16`` directly.
"""

from __future__ import annotations

from lib.landmarks.training import heatmap_stage as _heatmap_stage_impl

# BEGIN LEGACY_PRIVATE_HELPER_EXPORTS
# Keep legacy private imports available from TrainHeatmapStageFP16.py after
# moving the implementation to lib.landmarks.training.heatmap_stage.
#
# Tests and older scripts import these names directly from the historical entry
# point. Defining them here also preserves object identity checks such as
# train._dataloader_kwargs(...)["worker_init_fn"] is train._seed_worker.
from lib.landmarks.training.runtime import (
    dataloader_kwargs as _dataloader_kwargs,
    maybe_limit_eval_dataset as _maybe_limit_eval_dataset,
    normalize_runtime_args as _normalize_runtime_args,
    seed_worker as _seed_worker,
    set_dataset_runtime_epoch as _set_dataset_runtime_epoch,
    should_run_interval as _should_run_interval,
)
from lib.landmarks.training.checkpoint_compat import (
    build_training_compat_config_from_args as _training_compat_config,
    checkpoint_compat_errors_from_args as _checkpoint_compat_errors,
    file_sha256_or_none as _file_sha256_or_none,
    normalize_path_for_compat as _normalize_path_for_compat,
    training_compat_digest_from_args as _training_compat_digest,
    training_manifest_path_for_compat as _training_manifest_path_for_compat,
)
from lib.landmarks.training.checkpointing import (
    _checkpoint_metadata_path,
    _checkpoint_metadata_payload,
    _checkpoint_rng_state_for_payload,
    _collect_rng_state_by_rank,
    _current_rank_string,
    _json_safe_checkpoint_value,
    _local_rng_state_for_checkpoint,
    _model_state,
    _restore_training_checkpoint,
    _rng_state_for_current_rank,
    _save_training_checkpoint,
    _set_checkpoint_rng_state_by_rank,
    _torch_load_training_checkpoint,
    _write_checkpoint_metadata,
    _write_training_complete_sentinel,
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
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_heatmap_stage_impl)))


if __name__ == "__main__":
    main()
