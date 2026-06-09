# Landmark Dataset Staging

This document records the staging contract for the schema-aware manifest builder.
The builder keeps native point schemas and routes them to native training heads:

- `2d_194` -> `landmarks_194`
- `2d_106` -> `landmarks_106`
- `2d_98` -> `landmarks_98`
- `2d_68` -> `landmarks_68`
- `2d_29` -> `landmarks_29`
- `2d_39`, `menpo2d_profile_39`, `multipie_profile_39` -> `profile39`

`2d_98` (`MAP_98_TO_68`) and `2d_106` (`MAP_106_TO_68`) have audited projections
to canonical 68. The 106-point map is a semantic subsampling of the standard
106-point markup shared by LaPa and JD-landmark (and the FLL2/FLL3 derivatives
that use the same `2d_106` layout). The 29-, 39-, and 194-point schemas are
trainable native schemas and are marked `not_projectable` for 68-point
projection until an audited overlap map is added.

## Generic Staged Layout

For HELEN, LaPa, JD-landmark, fll2, FLL3, cofw68 original, XM2VTS, and FRGC, the
preferred local staging layout is:

```text
<dataset-root>/
  images/.../<sample>.jpg
  annotations/.../<sample>.pts
```

Each issue #8 dataset now enters through a dataset-specific parser function
before any generic fallback. Those parsers accept `.txt`, `.npy`, `.mat`, `.pts`,
and JSON files with `samples` or `entries`, but they also validate the expected
native point count for HELEN, LaPa, JD-landmark, fll2, FLL3, and cofw68 original.
Image and landmark files can be same-stem siblings, or the image can be found
recursively under `--image-root`.

JSON entries should use:

```json
{
    "sample_id": "subject/session/sample",
    "dataset": "lapa",
    "image": "images/sample.jpg",
    "landmarks": "annotations/sample.pts",
    "source_schema": "2d_106",
    "split": "train",
    "visibility": [1, 1, 0],
    "metadata": {
        "subject_id": "subject",
        "session_id": "session"
    }
}
```

If `split` is absent, a deterministic train/test split is assigned from
`split_safe_id`, `video_id`, `session_id`, `subject_id`, or `sample_id` in that
order. XM2VTS and FRGC directory layouts should keep subject/session/capture
folders because those identifiers are preserved for leakage checks.

## Dataset Sources

| Dataset         | Schema            | Source                                                                                      | Builder command                                                                                                                                                                       |
| --------------- | ----------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| HELEN dense     | `2d_194`          | `https://github.com/argiacomi/faceswap/issues/99`                                           | `python tools/build_quality_dataset.py --dataset helen --source-dir <annotations-root> --image-root <300w-cache>/data/300w/300w --output-dir runs/landmarks/build_helen`    |
| LaPa            | `2d_106`          | Google Drive file id `1XOBoRGSraP50_pS1YPB8_i8Wmw_5L-NG`                                    | `python tools/build_quality_dataset.py --dataset lapa --source-dir <root> --output-dir runs/landmarks/build_lapa`                                                           |
| JD-landmark     | `2d_106`          | `https://github.com/argiacomi/faceswap/issues/98`                                           | `python tools/build_quality_dataset.py --dataset jd-landmark --source-dir <jd-root> --image-root <300w-cache>/data/300w/300w --output-dir runs/landmarks/build_jd_landmark` |
| fll2            | `2d_106`          | Google Drive file id `16fiVoBaTtOevQa4mH34rWggfkNKNEL2A`                                    | `python tools/build_quality_dataset.py --dataset fll2 --source-dir <root> --output-dir runs/landmarks/build_fll2`                                                           |
| FLL3            | `2d_106`          | Google Drive file id `1F_UnmpRnUnNS3Wk3V6CkJiIUYmG5Wjdr`                                    | `python tools/build_quality_dataset.py --dataset fll3 --source-dir <root> --output-dir runs/landmarks/build_fll3`                                                           |
| cofw68 original | `2d_29`           | `https://data.caltech.edu/records/bc0bf-nc666/files/COFW_color.zip?download=1`              | `python tools/build_quality_dataset.py --dataset cofw29 --source-dir <root> --output-dir runs/landmarks/build_cofw68_original`                                              |
| XM2VTS          | staged schema     | Google Drive file id `1qdBlQhq9YEt5lzX1OGy5_AyjFL3vWxRs`                                    | `python tools/build_quality_dataset.py --dataset xm2vts --source-dir <root> --output-dir runs/landmarks/build_xm2vts`                                                       |
| FRGC            | staged schema     | Google Drive file id `1T2Ux0tjd5CxI9PWZb5sXThuGvWH-oM5p`                                    | `python tools/build_quality_dataset.py --dataset frgc --source-dir <root> --output-dir runs/landmarks/build_frgc`                                                           |
| 300VW           | frame annotations | `https://ibug.doc.ic.ac.uk/download/300VW_Dataset_2015_12_14.zip/`                          | `python tools/build_quality_dataset.py --dataset 300vw --source-dir <root> --output-dir runs/landmarks/build_300vw --frame-stride 5`                                        |
| WFLW-V          | frame annotations | Google Drive file id `1YSJdgIb-vToJIAV04PGh_U7nX6dxVSjt`                                    | `python tools/build_quality_dataset.py --dataset wflw-v --source-dir <root> --output-dir runs/landmarks/build_wflw_v --frame-stride 5`                                      |

Use `tools/download_landmark_datasets.py --list --dataset all` to
inspect configured sources.

The downloader reuses existing archives before downloading. It checks the
dataset's archive directory for the configured filename and known alternates
(for example `WFLW_images.zip`, `WFLW_images.tar.gz`, and `WFLW_images.tgz`),
and reuses a shared image archive across variants (`cofw68` and `cofw29` share
`COFW_color.zip`). Use `--force` to re-download. Files saved with an archive
extension that are not valid zip/tar archives (such as HTML error/login pages)
are rejected before extraction and removed so later runs do not reuse them.
The `registry.json` records whether each asset was `downloaded`, `reused`,
`reused_shared`, or manually staged.

When preparing multiple datasets, `--audit-overlay-limit` applies per dataset,
and overlays are written under `visual_audit/overlays/<dataset>/<schema>/`.

HELEN dense and JD-landmark are annotation layers over the 300W image cache.
HELEN expects `annotations.json` from the linked S3 source and resolves images
under `<300w-cache>/data/300w/300w/helen/{trainset,testset}`. JD-landmark
resolves corrected/test annotation names back to 300W subsets, for example
`AFW_134212_1_0.jpg.txt` resolves to `afw/134212_1.jpg`; ambiguous or missing
300W matches are reported as staging errors.

## Video Layout

Video datasets can be staged with videos and per-frame annotations:

```text
<dataset-root>/
  videos/<video_id>.avi
  annotations/<video_id>/000001.pts
  annotations/<video_id>/000002.pts
```

The builder searches `annotations/`, `landmarks/`, `labels/`, and
`<dataset-root>/<video_id>/` for zero-based and one-based frame names. Extracted
frames are written under `<output-dir>/frames/<dataset>/...` unless
`--frame-output-dir` is provided.

All frames from a video use `split_safe_id=<video_id>` and share the same split.
Each manifest entry includes `video_id`, `frame_id`, `frame_index`, and source
video/landmark paths for leakage-safe evaluation and auditing.

## Head Pose Metadata

Each sample carries head-pose fields in its `metadata` when they can be resolved.
Signed yaw values emit `pose_yaw_deg`, `pose_abs_yaw_deg`, `pose_side`, and a
side-specific `pose_bucket`. Magnitude-only yaw labels emit `pose_abs_yaw_deg`,
`pose_side="unknown"`, and a side-agnostic `pose_bucket` rather than guessing
left or right. Missing yaw or pitch evidence is recorded as `unknown`, not as
`frontal` or `neutral`.

Pose fields:

- `pose_yaw_deg` -- signed yaw angle when side is known. Yaw > 0 turns toward
  image-right.
- `pose_abs_yaw_deg` -- absolute yaw magnitude when signed yaw or label-only yaw
  magnitude is known.
- `pose_side` -- `unknown`, `frontal`, `left`, or `right`.
- `pose_bucket` -- one of `unknown`, `frontal`, `left_slight`/`right_slight`,
  `left_profile`/`right_profile`, `left_extreme`/`right_extreme`, or the
  side-agnostic magnitude tiers `slight`, `profile`, and `extreme`.
- `pose_pitch_deg` -- signed pitch angle when available. Pitch > 0 is up.
- `pitch_bucket` -- one of `unknown`, `neutral`, `up`/`down`, or
  `up_extreme`/`down_extreme`.
- `pose_roll_deg` -- roll angle when available. Roll > 0 is clockwise.
- `pose_source` -- source used to resolve pose metadata.

`pose_source` is one of:

- `annotation` -- dataset-provided angles (e.g. AFLW2000-3D `Pose_Para`).
- `dataset_label` -- MultiPIE profile/semifrontal capture labels.
- `landmark_geometry` -- approximate estimate from the 68-point geometry, used
  for any schema with an audited 68-point projection (300W, WFLW, Menpo2D,
  300VW, cofw68, LaPa, JD-landmark). Yaw and roll are reliable; pitch is a
  coarse 2D proxy. Sparse schemas without a 68 projection (e.g. cofw29) get no
  pose fields.

## Manifest Reports

Every build writes:

- `manifest.json`
- `dataset_audit.json`

Reports include per-dataset counts, split counts, source/target schema counts,
head counts, skipped sample examples, and projection status counts. Unsupported
or unaudited projection status is reported as `not_projectable` instead of
silently collapsing labels to 68 points.

Pass `--write-overlays` to emit `visual_audit/visual_audit.json` and schema
native point overlays under `visual_audit/overlays/`.
