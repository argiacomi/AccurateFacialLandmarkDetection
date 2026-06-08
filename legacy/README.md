# Legacy Module Map

This file tracks the migration from original top-level CD-ViT modules to
`lib.landmarks.*` package paths. New code should import through the package
paths below. Some top-level modules still exist temporarily, but no new
top-level shims should be added.

## Datasets

- `Data300WAngleWeights.py` -> `lib.landmarks.datasets.w300_angle_weights`
- `DataMutliResHM.py` -> `lib.landmarks.datasets.multi_res_heatmap`
- `Dataset2.py` -> `lib.landmarks.datasets.wflw_v2`
- `DatasetAAM.py` -> `lib.landmarks.datasets.aam`
- `DatasetLHM.py` -> `lib.landmarks.datasets.lhm`
- `DatasetMicroWFLW.py` -> `lib.landmarks.datasets.micro_wflw`
- `DatasetTagent.py` -> `lib.landmarks.datasets.tangent`
- `Dataset.py` -> `lib.landmarks.datasets.wflw`
- `DatasetCOFW.py` -> `lib.landmarks.datasets.cofw`
- `Dataset300W.py` -> `lib.landmarks.datasets.w300`
- `DatasetFS68Manifest.py` -> `lib.landmarks.datasets.manifest`
- `DatasetMultiSchemaLandmarkManifest.py` -> `lib.landmarks.datasets.multischema_manifest`
- `DatasetAll.py` -> `lib.landmarks.datasets.registry`

## Models

- `Regression.py` -> `lib.landmarks.models.regression`
- `Attention.py` -> `lib.landmarks.models.attention`
- `Net.py` CD-ViT classes -> `lib.landmarks.models.cdvit`
- `Heatmap.py` -> `lib.landmarks.models.heatmap`
- `UNet3.py` -> `lib.landmarks.models.unet3`
- `UNetHM.py` -> `lib.landmarks.models.unet_hm`
- `UNet2.py` -> `lib.landmarks.models.unet`
- `Vit.py` -> `lib.landmarks.models.vit`
- `ccnet.py` -> `lib.landmarks.models.ccnet`
- `coord_conv.py` -> `lib.landmarks.models.coord_conv`
- `helpers.py` -> `lib.landmarks.models.blocks`

## Training And Transforms

- `AUGHIH.py` -> `lib.landmarks.training.aughih`
- `Config.py` -> `lib.landmarks.core.config`
- `EMA.py` -> `lib.landmarks.training.ema`
- `ImageAugmentation.py` -> `lib.landmarks.training.augmentation`
- `DrawHeatmap.py` -> `lib.landmarks.training.heatmap_targets`
- `loss_function.py` -> `lib.landmarks.training.loss_function`
- `RandomFlip.py` -> `lib.landmarks.transforms.flip`
- `RandomFlip.py` -> `lib.landmarks.transforms.random_flip`

## Evaluation

- `Evaluation.py` -> `lib.landmarks.evaluation.legacy`
- `EvaluationAll.py` -> `lib.landmarks.evaluation.all`
- `EvaluationVideo.py` -> `lib.landmarks.evaluation.video`

`TrainHeatmapStageFP16.py` should stay as a temporary compatibility wrapper
around `lib.landmarks.training.heatmap_stage`.
