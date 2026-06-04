# the code for our WACV2025 paper "Cascaded Dual Vision Transformer for Accurate Facial Landmark Detection"


# train
Download the pre-cropped WFLW images from https://github.com/starhiking/HeatmapInHeatmap?tab=readme-ov-file, unzip the WFLW.zip under root folder and run the following commands:

        torchrun --nproc_per_node=2 TrainHeatmapStageFP16.py


# faceswap 68-point hard-negative training pipeline

This fork also supports training CD-ViT on faceswap-compatible 68-point manifests with hard-negative weighting for profile, occlusion, and profile+occlusion samples.

The CD-ViT repository consumes the final manifest. The dataset normalization and manifest mining are still delegated to `argiacomi/faceswap` via `tools/landmarks/run_cdvit_manifest_training_pipeline.py`.

Example full pipeline:

```bash
python tools/landmarks/run_cdvit_manifest_training_pipeline.py \
  --faceswap-root /path/to/faceswap \
  --output-root runs/landmarks \
  --run-name cdvit_fs68_hard \
  --dataset wflw,cofw,merl-rav,aflw2000-3d,300w,menpo2d,multipie \
  --dataset-source wflw=/data/raw/wflw \
  --dataset-source cofw=/data/raw/cofw \
  --dataset-source merl-rav=/data/raw/merl-rav \
  --dataset-source aflw2000-3d=/data/raw/aflw2000-3d \
  --dataset-source 300w=/data/raw/300w \
  --dataset-source menpo2d=/data/raw/menpo2d \
  --dataset-source multipie=/data/raw/multipie \
  --max-profile-occlusion 50000 \
  --max-profile 50000 \
  --max-occlusion 50000 \
  --max-anchors 50000 \
  --nproc-per-node 2 \
  --batch-size 16 \
  --epoch 500 \
  --heatmap-size 32
```

The script runs these stages:

1. `build_dataset_manifests`: calls faceswap `build_quality_dataset.py` once per dataset.
2. `build_hard_negative_manifest`: calls faceswap `build_hard_negative_manifest.py` to create the biased hard-negative mix.
3. `validate_cdvit_manifest`: verifies that the final manifest has readable `(68, 2)` landmark `.npy` files.
4. `train_cdvit`: launches `TrainHeatmapStageFP16.py --data_name FS68Manifest` with the mined manifest.

Useful controls:

```bash
# Preview commands without executing them.
python tools/landmarks/run_cdvit_manifest_training_pipeline.py --dry-run ...

# Only build manifests and stop before training.
python tools/landmarks/run_cdvit_manifest_training_pipeline.py --stop-after validate_cdvit_manifest ...

# Resume from an already-built hard-negative manifest.
python tools/landmarks/run_cdvit_manifest_training_pipeline.py \
  --manifest runs/landmarks/cdvit_fs68_hard/hard_negative_mix/manifest.json \
  --start-at validate_cdvit_manifest
```

By default, Menpo2D and MultiPIE 39-point profile samples are excluded because CD-ViT currently trains on exactly 68 landmarks. Pass `--include-39pt-profile --allow-non68` only if you intentionally want mixed manifests and are comfortable with `DatasetFS68Manifest` skipping non-68 samples.
