# Legacy Module Map

This file tracks the migration from original top-level CD-ViT modules to
`lib.landmarks.*` package paths. New code should import through the package
paths below. Some top-level modules still exist temporarily, but no new
top-level shims should be added.

Adapters that were no longer imported by the live implementation have been
removed (the old AAM/LHM/MicroWFLW/Tangent/WFLW-v2/MultiResHM/300W-angle-weight
dataset adapters, the unused Regression/ccnet/UNet3/UNetHM models, the
standalone Evaluation/EvaluationAll/EvaluationVideo scripts, AUGHIH, the legacy
`core.config` shim, and the `ensemble` package). Their mapping rows are dropped
below.

## Datasets

- `Dataset.py` -> `lib.landmarks.datasets.wflw`
- `DatasetCOFW.py` -> `lib.landmarks.datasets.cofw`
- `Dataset300W.py` -> `lib.landmarks.datasets.w300`
- `DatasetFS68Manifest.py` -> `lib.landmarks.datasets.manifest`
- `DatasetMultiSchemaLandmarkManifest.py` -> `lib.landmarks.datasets.multischema_manifest`
- `DatasetAll.py` -> `lib.landmarks.datasets.registry`

## Models

- `Attention.py` -> `lib.landmarks.models.attention`
- `Net.py` CD-ViT classes -> `lib.landmarks.models.cdvit`
- `Heatmap.py` -> `lib.landmarks.models.heatmap`
- `UNet2.py` -> `lib.landmarks.models.unet`
- `Vit.py` -> `lib.landmarks.models.vit`
- `coord_conv.py` -> `lib.landmarks.models.coord_conv`
- `helpers.py` -> `lib.landmarks.models.blocks`

## Training And Transforms

- `EMA.py` -> `lib.landmarks.training.ema`
- `ImageAugmentation.py` -> `lib.landmarks.training.augmentation`
- `DrawHeatmap.py` -> `lib.landmarks.training.heatmap_targets`
- `loss_function.py` -> `lib.landmarks.training.loss_function`
- `RandomFlip.py` -> `lib.landmarks.transforms.flip`

`TrainHeatmapStageFP16.py` should stay as a temporary compatibility wrapper
around `lib.landmarks.training.heatmap_stage`.
