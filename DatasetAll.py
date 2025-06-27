from DatasetCOFW import LandmarkDataset as DCOFW
from Dataset300W import LandmarkDataset as D300W
from Dataset import LandmarkDataset as DWFLW


def GetDataset(name, data_root, split, preload=True, aug=True, perturbation=False, heatmap_size=0):
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
