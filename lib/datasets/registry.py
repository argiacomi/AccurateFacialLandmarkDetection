from lib.core.manifest_aliases import (
    LEGACY_MANIFEST_DATA_NAME,
    MANIFEST_DATA_NAME_ALIASES,
    IsSchemaAwareManifestDataset,
)
from lib.datasets.cofw import LandmarkDataset as Dcofw68
from lib.datasets.multischema_manifest import (
    LandmarkDataset as DMultiSchemaLandmarkManifest,
)
from lib.datasets.w300 import LandmarkDataset as D300W
from lib.datasets.wflw import LandmarkDataset as DWFLW

FS68_DATASET_NAME = LEGACY_MANIFEST_DATA_NAME
SCHEMA_AWARE_MANIFEST_ALIASES = MANIFEST_DATA_NAME_ALIASES


def GetDataset(
    name,
    data_root,
    split,
    preload=True,
    aug=True,
    perturbation=False,
    heatmap_size=0,
    manifest_path="",
    eval_mode="random_hash",
    heldout_datasets=None,
    include_metadata=False,
    schema_aware_training=False,
    split_policy="declared_or_random_hash",
):
    if name == "WFLW":
        return DWFLW(
            data_root=data_root,
            split=split,
            preload=preload,
            aug=aug,
            perturbation=perturbation,
            heatmap_size=heatmap_size,
        )
    if name == "cofw68":
        return Dcofw68(
            data_root=data_root,
            split=split,
            preload=preload,
            aug=aug,
            perturbation=perturbation,
            heatmap_size=heatmap_size,
        )
    if name == "300W":
        return D300W(
            data_root=data_root,
            split=split,
            preload=preload,
            aug=aug,
            perturbation=perturbation,
            heatmap_size=heatmap_size,
        )
    if IsSchemaAwareManifestDataset(name):
        return DMultiSchemaLandmarkManifest(
            manifest_path=manifest_path or data_root,
            split=split,
            preload=preload,
            aug=aug,
            perturbation=perturbation,
            heatmap_size=heatmap_size,
            eval_mode=eval_mode,
            heldout_datasets=heldout_datasets,
            include_metadata=include_metadata,
            schema_aware_training=schema_aware_training,
            split_policy=split_policy,
        )
    raise ValueError(f"unknown dataset name: {name}")


__all__ = [
    "FS68_DATASET_NAME",
    "GetDataset",
    "IsSchemaAwareManifestDataset",
    "SCHEMA_AWARE_MANIFEST_ALIASES",
]
