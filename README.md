# Cascaded Dual Vision Transformer for Accurate Facial Landmark Detection

Code for the WACV 2025 paper **"Cascaded Dual Vision Transformer for Accurate Facial Landmark Detection"**.

This fork also includes a local 68-point landmark manifest workflow for training CD-ViT from WFLW, COFW, 300W, AFLW2000-3D, MERL-RAV, Menpo2D, and MultiPIE-style sources.

## Setup

Create and activate a Python environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install PyTorch with the CUDA or CPU wheel appropriate for your machine if the default `pip install -r requirements.txt` does not select the right build. See the PyTorch install selector for the correct command for your CUDA version.

Google Drive-backed dataset downloads require `gdown`, which is included in `requirements.txt`.

## Original WFLW training path

Download the pre-cropped WFLW images from the HeatmapInHeatmap release, unzip `WFLW.zip` under the repository root, and run:

```bash
torchrun --nproc_per_node=2 TrainHeatmapStageFP16.py
```

## Local 68-point landmark workflow

The local workflow has three phases:

1. Download and extract dataset sources into `data/landmarks`.
2. Prepare any dataset-specific bridges, especially MERL-RAV labels matched to native AFLW images.
3. Build canonical 68-point manifest sources and then run the CD-ViT hard-negative training pipeline.

The convenience script performs the first three prep steps and writes copy/pasteable dataset-source arguments for the pipeline:

```bash
bash tools/landmarks/download_prepare_build_quality_datasets.sh
```

For a small smoke run that keeps going when a dataset is unavailable:

```bash
SAMPLES_PER_SCENARIO=50 CONTINUE_ON_ERROR=1 \
  bash tools/landmarks/download_prepare_build_quality_datasets.sh
```

The script writes:

```text
data/landmarks/<dataset>/archives/*
data/landmarks/<dataset>/extracted/*
data/landmarks/merl-rav/organized/manifest.json
runs/landmarks/quality_datasets/<dataset>/manifest.json
runs/landmarks/quality_datasets/dataset_source_args.txt
```

### Download datasets only

```bash
python tools/landmarks/download_landmark_datasets.py \
  --output-root data/landmarks \
  --dataset all \
  --extract \
  --include-google-drive \
  --keep-going
```

List the configured source URLs and Google Drive ids:

```bash
python tools/landmarks/download_landmark_datasets.py --list
```

Optional 300W iBUG split archive parts can be downloaded with:

```bash
python tools/landmarks/download_landmark_datasets.py \
  --dataset 300w \
  --include-alternates \
  --output-root data/landmarks
```

### Prepare MERL-RAV + AFLW

MERL-RAV labels are annotations over AFLW images. After downloading/extracting both sources, organize them with:

```bash
python tools/landmarks/prepare_merl_rav_aflw.py \
  --merl-rav-root data/landmarks/merl-rav/extracted/MERL-RAV_dataset-master.zip \
  --aflw-root data/landmarks/aflw/extracted/AFLW.zip \
  --output-dir data/landmarks/merl-rav/organized
```

The helper matches labels to AFLW `flickr/` images by `imageNNNNN`, preserves occlusion/visibility metadata, and writes a manifest consumable by `build_quality_dataset.py`.

### Build one dataset manifest manually

```bash
python tools/landmarks/build_quality_dataset.py \
  --dataset 300w \
  --source-dir data/landmarks/300w/extracted \
  --output-dir runs/landmarks/quality_datasets/300w
```

All emitted landmark files are materialized as canonical `(68, 2)` `.npy` files. Non-68/non-98 labels are skipped and reported in the dataset audit.

### Run the CD-ViT hard-negative pipeline

After running the setup script, launch the pipeline using the generated source arguments:

```bash
python tools/landmarks/run_cdvit_manifest_training_pipeline.py \
  --dataset wflw,cofw,300w,aflw2000-3d,merl-rav,menpo2d,multipie \
  $(tr "\n" " " < runs/landmarks/quality_datasets/dataset_source_args.txt) \
  --nproc-per-node 2 \
  --batch-size 16 \
  --epoch 500 \
  --heatmap-size 32
```

Pipeline stages:

1. `build_dataset_manifests`: calls local `tools/landmarks/build_quality_dataset.py` once per dataset.
2. `build_hard_negative_manifest`: calls local `tools/landmarks/build_hard_negative_manifest.py` to create a ratio-aware hard-negative mix.
3. `validate_cdvit_manifest`: verifies that the final manifest has readable `(68, 2)` landmark `.npy` files.
4. `train_cdvit`: launches `TrainHeatmapStageFP16.py --data_name FS68Manifest` with the mined manifest.

### Hard-negative mix defaults

`build_hard_negative_manifest.py` samples by ratio/percentage rather than by fixed bucket caps. By default, `--total-samples 0` uses every feasible classified sample after deduping, while preserving this hard-negative-heavy ordering/target ratio as much as the available buckets allow:

```text
profile_occlusion = 3
profile           = 2
occlusion         = 2
anchor            = 1
```

That corresponds to target percentages of 37.5%, 25%, 25%, and 12.5%. If one bucket has too few samples, the remaining capacity is redistributed to the other buckets with available samples. The resulting `hard_negative_mix.json` reports available counts, target counts, actual percentages, and any optional ceilings.

For a bounded experiment, pass an explicit total sample target through the pipeline:

```bash
python tools/landmarks/run_cdvit_manifest_training_pipeline.py \
  --hard-negative-arg "--total-samples 10000" \
  --hard-negative-arg "--bucket-percentages profile_occlusion=30,profile=30,occlusion=25,anchor=15" \
  ...
```

The old `--max-profile-occlusion`, `--max-profile`, `--max-occlusion`, and `--max-anchors` flags still exist as optional hard ceilings, but they are no longer the primary way to define the mix.

Useful controls:

```bash
# Preview commands without executing them.
python tools/landmarks/run_cdvit_manifest_training_pipeline.py --dry-run ...

# Only build manifests and stop before training.
python tools/landmarks/run_cdvit_manifest_training_pipeline.py \
  --stop-after validate_cdvit_manifest ...

# Resume from an already-built hard-negative manifest.
python tools/landmarks/run_cdvit_manifest_training_pipeline.py \
  --manifest runs/landmarks/cdvit_fs68_hard/hard_negative_mix/manifest.json \
  --start-at validate_cdvit_manifest
```

By default, Menpo2D and MultiPIE non-68 profile samples are excluded because CD-ViT currently trains on exactly 68 landmarks. Use `--allow-non68` only if you intentionally want mixed manifests and are comfortable with `DatasetFS68Manifest` training on the exact-68 subset.
