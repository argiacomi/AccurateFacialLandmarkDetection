from DatasetCOFW import LandmarkDataset as DCOFW
from Dataset300W import LandmarkDataset as D300W
from Dataset import LandmarkDataset as DWFLW
from DatasetFS68Manifest import LandmarkDataset as DFS68Manifest


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
    if name == "COFW":
        return DCOFW(
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
    if name == "FS68Manifest":
        return DFS68Manifest(
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
