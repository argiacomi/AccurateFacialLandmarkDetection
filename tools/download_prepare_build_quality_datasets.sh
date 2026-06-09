#!/usr/bin/env bash
# Download landmark datasets, prepare MERL-RAV/AFLW, and build quality manifests.
#
# This script performs the data-prep stages that should exist before running
# tools/run_cdvit_manifest_training_pipeline.py.
#
# Required for Google Drive-backed sources:
#   pip install gdown
#
# Optional env vars:
#   PYTHON=python3
#   DATA_ROOT=data/landmarks
#   QUALITY_ROOT=runs/landmarks/quality_datasets
#   INCLUDE_300W_ALTERNATES=0       # default: 0, set 1 to also fetch official iBUG split parts
#   CONTINUE_ON_ERROR=0             # default: 0, set 1 for partial bring-up
#   MERL_RAV_IMAGE_MODE=absolute    # absolute|symlink|copy
#   MERL_RAV_SPLITS=train,test
#   MERL_RAV_SKIP_IMAGE_VALIDATION=0
#   MANIFEST_MODE=replace           # replace|merge
#   ALLOW_OVERLAP=0
#   SAMPLES_PER_SCENARIO=           # e.g. 100 for a quick smoke build
#
# Example:
#   bash tools/download_prepare_build_quality_datasets.sh
#
# Partial/smoke run:
#   SAMPLES_PER_SCENARIO=50 CONTINUE_ON_ERROR=1 bash tools/download_prepare_build_quality_datasets.sh

set -Eeuo pipefail

PYTHON="${PYTHON:-python3}"
DATA_ROOT="${DATA_ROOT:-data/landmarks}"
QUALITY_ROOT="${QUALITY_ROOT:-runs/landmarks/quality_datasets}"
INCLUDE_300W_ALTERNATES="${INCLUDE_300W_ALTERNATES:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
MERL_RAV_IMAGE_MODE="${MERL_RAV_IMAGE_MODE:-absolute}"
MERL_RAV_SPLITS="${MERL_RAV_SPLITS:-train,test}"
MERL_RAV_SKIP_IMAGE_VALIDATION="${MERL_RAV_SKIP_IMAGE_VALIDATION:-0}"
MANIFEST_MODE="${MANIFEST_MODE:-replace}"
ALLOW_OVERLAP="${ALLOW_OVERLAP:-0}"
SAMPLES_PER_SCENARIO="${SAMPLES_PER_SCENARIO:-}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

warn() {
  printf '\nWARNING: %s\n' "$*" >&2
}

fail() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

run_step() {
  local name="$1"
  shift
  log "${name}"
  if "$@"; then
    return 0
  fi
  local status=$?
  if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
    warn "Step failed with status ${status}: ${name}"
    return 0
  fi
  return "${status}"
}

need_file() {
  [[ -f "$1" ]] || fail "Missing required file: $1"
}

maybe_args=()
if [[ -n "${SAMPLES_PER_SCENARIO}" ]]; then
  maybe_args+=(--samples-per-scenario "${SAMPLES_PER_SCENARIO}")
fi
if [[ "${ALLOW_OVERLAP}" == "1" ]]; then
  maybe_args+=(--allow-overlap)
fi

need_file "tools/download_landmark_datasets.py"
need_file "tools/prepare_merl_rav_aflw.py"
need_file "tools/build_quality_dataset.py"

mkdir -p "${DATA_ROOT}" "${QUALITY_ROOT}"

# 1. Download all configured sources into DATA_ROOT.
download_args=(
  "tools/download_landmark_datasets.py"
  --output-root "${DATA_ROOT}"
  --dataset all
  --extract
)

if [[ "${INCLUDE_300W_ALTERNATES}" == "1" ]]; then
  download_args+=(--include-alternates)
fi
if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
  download_args+=(--keep-going)
fi

run_step "Download landmark datasets into ${DATA_ROOT}" "${PYTHON}" "${download_args[@]}"

# 2. Prepare MERL-RAV + native AFLW into an organized source directory.
MERL_RAV_ROOT="${MERL_RAV_ROOT:-${DATA_ROOT}/merl-rav/extracted/MERL-RAV_dataset-master.zip}"
AFLW_ROOT="${AFLW_ROOT:-${DATA_ROOT}/aflw/extracted/AFLW.zip}"
MERL_RAV_ORGANIZED="${MERL_RAV_ORGANIZED:-${DATA_ROOT}/merl-rav/organized}"

prepare_merl_args=(
  "tools/prepare_merl_rav_aflw.py"
  --merl-rav-root "${MERL_RAV_ROOT}"
  --aflw-root "${AFLW_ROOT}"
  --output-dir "${MERL_RAV_ORGANIZED}"
  --splits "${MERL_RAV_SPLITS}"
  --image-mode "${MERL_RAV_IMAGE_MODE}"
)
if [[ "${MERL_RAV_SKIP_IMAGE_VALIDATION}" == "1" ]]; then
  prepare_merl_args+=(--skip-image-validation)
fi

run_step "Prepare MERL-RAV annotations against AFLW images" "${PYTHON}" "${prepare_merl_args[@]}"

# 3. Build all per-dataset quality manifests as a fast validation/preflight of
#    each prepared source directory. The downstream training pipeline can also
#    rebuild per-run manifests from the source directories written below.
build_dataset() {
  local dataset="$1"
  local source_dir="$2"
  local output_dir="$3"
  shift 3

  if [[ ! -d "${source_dir}" ]]; then
    if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
      warn "Skipping ${dataset}: source directory does not exist: ${source_dir}"
      return 0
    fi
    fail "Source directory for ${dataset} does not exist: ${source_dir}"
  fi

  run_step "Build ${dataset} quality manifest" \
    "${PYTHON}" tools/build_quality_dataset.py \
      --dataset "${dataset}" \
      --source-dir "${source_dir}" \
      --output-dir "${output_dir}" \
      --manifest-mode "${MANIFEST_MODE}" \
      "${maybe_args[@]}" \
      "$@"
}

WFLW_SOURCE="${WFLW_SOURCE:-${DATA_ROOT}/wflw/extracted}"
cofw68_SOURCE="${cofw68_SOURCE:-${DATA_ROOT}/cofw68/extracted}"
W300_SOURCE="${W300_SOURCE:-${DATA_ROOT}/300w/extracted}"
AFLW2000_SOURCE="${AFLW2000_SOURCE:-${DATA_ROOT}/aflw2000-3d/extracted}"
MENPO2D_SOURCE="${MENPO2D_SOURCE:-${DATA_ROOT}/menpo2d/extracted}"
MULTIPIE_SOURCE="${MULTIPIE_SOURCE:-${DATA_ROOT}/multipie/extracted}"

build_dataset wflw "${WFLW_SOURCE}" "${QUALITY_ROOT}/wflw"
build_dataset cofw68 "${cofw68_SOURCE}" "${QUALITY_ROOT}/cofw68"
build_dataset 300w "${W300_SOURCE}" "${QUALITY_ROOT}/300w"
build_dataset aflw2000-3d "${AFLW2000_SOURCE}" "${QUALITY_ROOT}/aflw2000-3d"
build_dataset merl-rav "${MERL_RAV_ORGANIZED}" "${QUALITY_ROOT}/merl-rav"
build_dataset menpo2d "${MENPO2D_SOURCE}" "${QUALITY_ROOT}/menpo2d"
build_dataset multipie "${MULTIPIE_SOURCE}" "${QUALITY_ROOT}/multipie"

# Write source-directory arguments for run_cdvit_manifest_training_pipeline.py.
# The pipeline expects source roots, not manifest.json files, because it creates
# per-run manifests before mining the hard-negative mix.
DATASET_SOURCE_ARGS="${QUALITY_ROOT}/dataset_source_args.txt"
cat > "${DATASET_SOURCE_ARGS}" <<EOF
--dataset-source wflw=${WFLW_SOURCE}
--dataset-source cofw68=${cofw68_SOURCE}
--dataset-source 300w=${W300_SOURCE}
--dataset-source aflw2000-3d=${AFLW2000_SOURCE}
--dataset-source merl-rav=${MERL_RAV_ORGANIZED}
--dataset-source menpo2d=${MENPO2D_SOURCE}
--dataset-source multipie=${MULTIPIE_SOURCE}
EOF

log "Finished quality dataset setup"
printf 'Data root: %s\n' "${DATA_ROOT}"
printf 'Quality manifests root: %s\n' "${QUALITY_ROOT}"
printf 'Dataset-source args: %s\n' "${DATASET_SOURCE_ARGS}"
printf '\nNext step example:\n'
printf '  %s tools/run_cdvit_manifest_training_pipeline.py --dataset wflw,cofw68,300w,aflw2000-3d,merl-rav,menpo2d,multipie $(tr "\\n" " " < %s)\n' "${PYTHON}" "${DATASET_SOURCE_ARGS}"
