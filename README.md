# Cascaded Dual Vision Transformer for Accurate Facial Landmark Detection

Code for the WACV 2025 paper **"Cascaded Dual Vision Transformer for Accurate Facial Landmark Detection"**.

This fork also includes a local schema-aware landmark manifest workflow for training CD-ViT from WFLW, cofw68, 300W, AFLW2000-3D, MERL-RAV, Menpo2D, MultiPIE-style sources, and Faceswap production alignment directories.

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
bash tools/download_prepare_build_quality_datasets.sh
```

For a small smoke run that keeps going when a dataset is unavailable:

```bash
SAMPLES_PER_SCENARIO=50 CONTINUE_ON_ERROR=1 \
  bash tools/download_prepare_build_quality_datasets.sh
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
python tools/download_landmark_datasets.py \
  --output-root data/landmarks \
  --dataset all \
  --extract \
  --keep-going
```

List the configured source URLs and Google Drive ids:

```bash
python tools/download_landmark_datasets.py --list
```

Optional 300W iBUG split archive parts can be downloaded with:

```bash
python tools/download_landmark_datasets.py \
  --dataset 300w \
  --include-alternates \
  --output-root data/landmarks
```

### Prepare MERL-RAV + AFLW

MERL-RAV labels are annotations over AFLW images. After downloading/extracting both sources, organize them with:

```bash
python tools/prepare_merl_rav_aflw.py \
  --merl-rav-root data/landmarks/merl-rav/extracted/MERL-RAV_dataset-master.zip \
  --aflw-root data/landmarks/aflw/extracted/AFLW.zip \
  --output-dir data/landmarks/merl-rav/organized
```

The helper matches labels to AFLW `flickr/` images by `imageNNNNN`, preserves occlusion/visibility metadata, and writes a manifest consumable by `build_quality_dataset.py`.

### Build one dataset manifest manually

```bash
python tools/build_quality_dataset.py \
  --dataset 300w \
  --source-dir data/landmarks/300w/extracted \
  --output-dir runs/landmarks/quality_datasets/300w
```

Dataset manifests preserve native trainable schemas when available, including 29, 39, 68, 98, 106, and 194 point layouts. Each emitted landmark file is materialized as a `.npy` array whose shape must match its `target_schema`; `source_schema` records the original annotation source and `target_schema` records the saved training target. Unsupported or unaudited labels are skipped and reported in the dataset audit.

### Faceswap production alignment directories or zip archives

By default, the production helper downloads the `production_validated` source from Google Drive file id `1XFW3_xx9t6gnyAIRY6g71keDzzHFRWRg`. Use `--prod-dir` only when you have a local directory or `.zip` archive containing production images and exactly one Faceswap `.fsa` alignments file. The local helper reads the `.fsa`, writes one canonical `(68, 2)` `.npy` landmark file per face, and emits a `production_validated` manifest.

```bash
python tools/build_production_validated_manifest.py \
  --output-dir data/landmarks/production_validated
```

To override the default download with a local source:

```bash
python tools/build_production_validated_manifest.py \
  --prod-dir /path/to/production_dir_or_zip \
  --output-dir data/landmarks/production_validated
```

When `--prod-dir` points at a `.zip`, the archive is safely extracted under the output directory and manifest image paths point at that stable extracted copy. When `--prod-dir` is omitted, the downloaded archive is cached under `<output-dir>/_downloads` unless `--download-root` is supplied.

The output includes:

```text
data/landmarks/production_validated/manifest.json
data/landmarks/production_validated/landmarks/*.npy
```

You can also let the CD-ViT pipeline build and include it automatically:

```bash
python tools/run_cdvit_manifest_training_pipeline.py \
  --dataset wflw,cofw68,300w,aflw2000-3d,merl-rav,menpo2d,multipie,prod \
  $(tr "\n" " " < runs/landmarks/quality_datasets/dataset_source_args.txt) \
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
python tools/run_cdvit_manifest_training_pipeline.py \
  --dataset wflw,cofw68,300w,aflw2000-3d,merl-rav,menpo2d,multipie \
  $(tr "\n" " " < runs/landmarks/quality_datasets/dataset_source_args.txt) \
  --nproc-per-node 2 \
  --batch-size 16 \
  --epoch 500 \
  --heatmap-size 32
```

Pipeline stages:

1. `build_dataset_manifests`: calls local `tools/build_quality_dataset.py` once per dataset and, when `--prod-dir` is supplied, builds `production_validated` from the production source.
2. `build_hard_negative_manifest`: calls local `tools/build_hard_negative_manifest.py` to create a ratio-aware hard-negative mix.
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
python tools/run_cdvit_manifest_training_pipeline.py \
  --hard-negative-arg "--total-samples 10000" \
  --hard-negative-arg "--bucket-percentages profile_occlusion=30,profile=30,occlusion=25,anchor=15" \
  ...
```

The old `--max-profile-occlusion`, `--max-profile`, `--max-occlusion`, and `--max-anchors` flags still exist as optional hard ceilings, but they are no longer the primary way to define the mix.

Useful controls:

```bash
# Preview commands without executing them.
python tools/run_cdvit_manifest_training_pipeline.py --dry-run ...

# Only build manifests and stop before training.
python tools/run_cdvit_manifest_training_pipeline.py \
  --stop-after validate_cdvit_manifest ...

# Resume from an already-built hard-negative manifest.
python tools/run_cdvit_manifest_training_pipeline.py \
  --manifest runs/landmarks/cdvit_fs68_hard/hard_negative_mix/manifest.json \
  --start-at validate_cdvit_manifest
```

Schema-aware CD-ViT training and evaluation use native heads for supported manifest schemas instead of forcing every sample through the 68-point output. The default head registry covers `landmarks_68`, `landmarks_98`, `landmarks_106`, `landmarks_194`, `profile39`, and `landmarks_29`; production inference still consumes `landmarks_68`. Schemas without audited flip maps are not randomly flipped during augmentation until a schema-specific flip map is registered. That currently disables horizontal flipping for `2d_39`; `2d_68` and WFLW-style `2d_98` have registered flip maps.

Mixed-schema training defaults to sample-count-weighted head losses (`--schema-head-loss-weighting sample_count`) so a head with one sample does not contribute the same as a head with many samples. Optional per-head multipliers can be supplied with `--schema-head-loss-weights`, for example `landmarks_98=1.0,profile39=0.5`. True 98-point samples train `landmarks_98` directly and can also regularize the production `landmarks_68` head through `--schema-consistency-weight`; the consistency projection uses `MAP_98_TO_68` from the 98-head prediction and detaches that projection, so the consistency term updates the 68 head without replacing native 98 supervision.

`STARLoss_v2` can be added as a small regularizer for active supervised schema heads with `--star-loss-weight`. It is disabled by default; start with small values such as `0.005`, `0.01`, `0.02`, or `0.05` and evaluate hard-case slices. STAR loss is applied after sample weights and landmark masks, and training logs report coordinate, heatmap, consistency, STAR, auxiliary, per-head counts, and per-head contributions separately.

Schema-aware checkpoints resume strictly by default. To intentionally resume an older 68-only checkpoint into a schema-aware model, pass `--allow-missing-schema-heads`; only missing schema/auxiliary head parameters are tolerated and they are initialized from the current model. Use `--no-schema-aware-training` for a legacy 68-only run, and keep strict resume behavior for schema-aware checkpoints unless the missing-head migration is intentional.

For standalone checkpoint evaluation, pass `--schema-aware-model` to construct the native heads. `--schema-aware-eval` defaults to the same value and makes the evaluator load non-68 samples with their native schema/head metadata:

```bash
python tools/evaluate_cdvit_manifest.py \
  --checkpoint runs/landmarks/cdvit/best.weights.pt \
  --manifest runs/landmarks/cdvit_fs68_hard/hard_negative_mix/manifest.json \
  --schema-aware-model \
  --eval-report-json runs/landmarks/eval/report.json
```

Visible/occluded landmark metrics are emitted when per-point visibility targets are present. JSON and CSV summaries include `NME_all`, `NME_visible`, `NME_occluded`, visibility AP/F1@0.5/ROC-AUC when visibility scores are predicted, visible/occluded landmark counts, and skipped-label counts. Per-sample records include visible/occluded counts, per-sample visible/occluded NME, the native evaluation head, and `visibility_target_source` so fields such as `visibility_mask` remain auditable. `NME_visible` and `NME_occluded` are means of per-sample visible/occluded NME values; use the reported landmark counts when interpreting slices where samples have very different numbers of labeled visible or occluded points.

### Training runtime controls

`FS68Manifest` remains a backward-compatible alias for the schema-aware manifest loader. New docs should describe the manifest path as schema-aware or multi-schema.

The pipeline forwards runtime flags to `TrainHeatmapStageFP16.py` so long runs can be resumed and profiled safely:

- Roll augmentation: `--roll-quarter-turn-prob 0.4` splits 40% of training samples evenly across exact `-90`/`+90` rotations, while `--roll-diagonal-prob 0.1` splits 10% across `-45`/`+45`. The remaining 50% receives no coarse rotation. The existing affine jitter then adds up to `±20` degrees, scale, translation, and shear.
- Data loading: `--preload 0`, `--pin-memory`, `--persistent-workers`, and `--prefetch-factor`.
- Evaluation cadence: `--eval-every`, `--full-eval-every`, `--eval-ema-every`, and `--eval-max-samples`.
- Checkpoints: `last_checkpoint.pt`, `last_checkpoint.weights.pt`, `best_checkpoint.pt`, `best.weights.pt`, metadata sidecars, manifest/config compatibility checks, and optional `--restore-rng`.
- Metrics: `runtime_metrics.jsonl` records epoch throughput, CUDA memory, data wait, device transfer, forward/loss, backward, optimizer step, scaler update, eval, EMA eval, checkpoint, aggregate compute, unattributed, and total epoch timing. CUDA transfer/compute/eval sections use CUDA event timing by default through `--synchronize-runtime-timing`; pass `--no-synchronize-runtime-timing` for low-overhead CPU wall-clock timing. Checkpoint timing includes periodic, legacy, best, EMA-best, and last-checkpoint writes.

Evaluation slice reports include `by_roll_bucket`: `upright` for `|roll| < 30°`, `diagonal` for `30° <= |roll| < 70°`, `horizontal` for `|roll| >= 70°`, and `unknown` when roll metadata is unavailable.

### Console output vs metrics files

The console is human-first and intentionally terse: it summarizes a run with short
`[tag]` lines (`[data]`, `[train]`, `[epoch]`, `[eval]`, `[sampler]`, `[pipeline]`,
`[manifest]`) and points at the structured files for detail. The machine-readable
record of a run always lives in those files regardless of console settings:
`runtime_metrics.jsonl` (trainer), `pipeline_progress.jsonl` (pipeline),
`eval_report.json` / eval record CSV/JSONL, and `dataset_audit.json` (dataset builders).

The trainer, pipeline, and dataset tools share two console flags:

- `--log-level quiet|info|verbose|debug` (default `info`):
  - `quiet` — only epoch/eval/stage summaries and errors.
  - `info` — adds the live training progress bar (or compact periodic `[train]` lines in non-TTY/JSON output).
  - `verbose` — adds schema-head diagnostics, the domain-balanced sampler mix, and checkpoint writes.
  - `debug` — adds full structures (e.g. raw sampler diagnostics, per-stage JSON) and stdlib `logger` output.
- `--log-format human|json`: `human` prints tagged lines; `json` emits one JSON object per event for CI/log parsing. The structured metrics files above are written either way.

```text
[data]  train 124,000 samples | test 6,200 samples | device cuda:0
[epoch] 12 done | 8m41s | 124,000 samples | 238.0 samples/s | lr 1.00e-04 | peak 11,842.0 MB
[eval]  test sampled | NME 3.84% | FR@0.10 1.25% | AUC@0.10 0.0741 | n=2,048
```

Progress bars use [Rich](https://github.com/Textualize/rich) when stderr is an interactive
terminal and automatically degrade to plain output under captured logs, `--log-format json`,
or when Rich is not installed (it is an optional dependency). Dataset tools also expose
`--progress/--no-progress` (and honor non-TTY detection) to control progress bars directly.
