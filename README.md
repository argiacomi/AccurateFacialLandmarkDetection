# Cascaded Dual Vision Transformer for Accurate Facial Landmark Detection

Code for the WACV 2025 paper **"Cascaded Dual Vision Transformer for Accurate Facial Landmark Detection"**.

This fork also includes a local schema-aware landmark manifest workflow for training CD-ViT from WFLW, COFW, 300W, AFLW2000-3D, MERL-RAV, Menpo2D, MultiPIE-style sources, and Faceswap production alignment directories.

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

## Local schema-aware landmark workflow

The local workflow has three phases:

1. Download and extract dataset sources into `data/landmarks`.
2. Prepare any dataset-specific bridges, especially MERL-RAV labels matched to native AFLW images.
3. Build schema-aware manifest sources and then run the CD-ViT hard-negative training pipeline.

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

Dataset manifests preserve native trainable schemas when available, including 29, 39, 68, 98, 106, and 194 point layouts. Each emitted landmark file is materialized as a `.npy` array whose shape must match its `target_schema`; unsupported or unaudited labels are skipped and reported in the dataset audit.

### Faceswap production alignment directories or zip archives

Use `--prod-dir` when you have either a directory or a `.zip` archive containing production images and exactly one Faceswap `.fsa` alignments file. The local helper reads the `.fsa`, writes one canonical `(68, 2)` `.npy` landmark file per face, and emits a `production_validated` manifest.

```bash
python tools/landmarks/build_production_validated_manifest.py \
  --prod-dir /path/to/production_dir_or_zip \
  --output-dir data/landmarks/production_validated
```

When `--prod-dir` points at a `.zip`, the archive is safely extracted under the output directory and manifest image paths point at that stable extracted copy.

The output includes:

```text
data/landmarks/production_validated/manifest.json
data/landmarks/production_validated/landmarks/*.npy
```

You can also let the CD-ViT pipeline build and include it automatically:

```bash
python tools/landmarks/run_cdvit_manifest_training_pipeline.py \
  --dataset wflw,cofw,300w,aflw2000-3d,merl-rav,menpo2d,multipie \
  $(tr "\n" " " < runs/landmarks/quality_datasets/dataset_source_args.txt) \
  --prod-dir /path/to/production_dir_or_zip \
  --nproc-per-node 2 \
  --batch-size 16 \
  --epoch 500 \
  --heatmap-size 32
```

Production runtime buckets such as `frontal`, `intermediate`, `large_yaw_left`, `profile_right`, `large_roll`, `extreme_roll`, and rolled profile/yaw buckets are recognized during hard-negative classification. Review `runs/.../hard_negative_mix/hard_negative_mix.json` and `dataset_audit.json` to confirm how many production samples land in each bucket.

Because `.fsa` files are compressed pickle files, only use `--prod-dir` with trusted local production sources.

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

1. `build_dataset_manifests`: calls local `tools/landmarks/build_quality_dataset.py` once per dataset and, when `--prod-dir` is supplied, builds `production_validated` from the production source.
2. `build_hard_negative_manifest`: calls local `tools/landmarks/build_hard_negative_manifest.py` to create a ratio-aware hard-negative mix.
3. `validate_cdvit_manifest`: compatibility stage name for schema-aware training-manifest validation. It verifies readable landmark `.npy` files, `source_schema` / `target_schema` consistency, `head_name`, `landmark_count`, split-safe identifiers, projection-audit metadata, and schema/count reports.
4. `train_cdvit`: launches `TrainHeatmapStageFP16.py` with the mined manifest. `FS68Manifest` remains the backward-compatible default `--data_name`; `MultiSchemaLandmarkManifest` is the canonical schema-aware alias.

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

Schema-aware CD-ViT training and evaluation use native heads for supported manifest schemas instead of forcing every sample through the 68-point output. The default head registry covers `landmarks_68`, `landmarks_98`, `landmarks_106`, `landmarks_194`, `profile39`, and `landmarks_29`; production inference still consumes `landmarks_68`. Schemas without audited flip maps are not randomly flipped during augmentation until a schema-specific flip map is registered.

For standalone checkpoint evaluation, pass `--schema-aware-model` to construct the native heads. `--schema-aware-eval` defaults to the same value and makes the evaluator load non-68 samples with their native schema/head metadata:

```bash
python tools/landmarks/evaluate_cdvit_manifest.py \
  --checkpoint runs/landmarks/cdvit/best_model \
  --manifest runs/landmarks/cdvit_fs68_hard/hard_negative_mix/manifest.json \
  --schema-aware-model \
  --eval-report-json runs/landmarks/eval/report.json
```

Visible/occluded landmark metrics are emitted when per-point visibility targets are present. JSON and CSV summaries include `NME_all`, `NME_visible`, `NME_occluded`, visibility AP/F1@0.5/ROC-AUC when visibility scores are predicted, visible/occluded landmark counts, and skipped-label counts. Per-sample records include visible/occluded counts, per-sample visible/occluded NME, the native evaluation head, and `visibility_target_source` so fields such as `visibility_mask` remain auditable. `NME_visible` and `NME_occluded` are means of per-sample visible/occluded NME values; use the reported landmark counts when interpreting slices where samples have very different numbers of labeled visible or occluded points.

### Training runtime controls

The pipeline forwards runtime flags to `TrainHeatmapStageFP16.py` so long runs can be resumed and profiled safely:

- Data loading: `--preload 0`, `--pin-memory`, `--persistent-workers`, and `--prefetch-factor`.
- Evaluation cadence: `--eval-every`, `--full-eval-every`, `--eval-ema-every`, and `--eval-max-samples`.
- Checkpoints: `last_checkpoint.pt`, `best_checkpoint.pt`, metadata sidecars, manifest/config compatibility checks, and optional `--restore-rng`.
- Metrics: `runtime_metrics.jsonl` records epoch throughput, CUDA memory, data wait, transfer, forward/backward/update, eval, checkpoint, and total epoch timing where available.

