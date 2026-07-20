#!/usr/bin/env bash
# run_sam3d_mhr_smpl.sh
#
# SAM3D-Body + MHR->SMPL pipeline.
#
# Steps:
#   0a  extract frames from video (ffmpeg)
#   0b  SAM3D-Body detection + MoGe-2 focal length estimation
#   1   select target-athlete detection per frame
#   2   MHR -> SMPL conversion  (pixi / MHR env)
#   3   add SMPL 2D/3D keypoints
#
# Output per athlete:
#   $SAM3D_DIR/outputs/smpl_sequences/<Athlete>/smpl_sequence.pkl
#
# Required env vars (set once, or add to ~/.bashrc):
#   SAM3D_DIR   path to sam-3d-body repo
#   MHR_DIR     path to MHR repo
#   SMPL_MODEL  path to SMPL_N_model_generate_from_npz.pkl
#   DATASET_DIR folder containing <Athlete>.mp4 files
#
# Usage:
#   export SAM3D_DIR=... MHR_DIR=... SMPL_MODEL=... DATASET_DIR=...
#
#   bash scripts/run_sam3d_mhr_smpl.sh Goree Bishop   # specific athletes
#   bash scripts/run_sam3d_mhr_smpl.sh                # all .mp4 in DATASET_DIR
#   DEVICE=cuda SKIP_EXISTING=1 bash scripts/run_sam3d_mhr_smpl.sh Goree
#   FOV_NAME="" bash scripts/run_sam3d_mhr_smpl.sh    # disable MoGe-2
# -----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODES_DIR="$(dirname "${SCRIPT_DIR}")"

# --- validate required env vars ----------------------------------------------
for var in SAM3D_DIR MHR_DIR SMPL_MODEL DATASET_DIR; do
    if [ -z "${!var:-}" ]; then
        echo "[ERROR] \$$var is not set. Export it before running."
        exit 1
    fi
done

if [ ! -f "${SMPL_MODEL}" ]; then
    echo "[ERROR] SMPL model not found: ${SMPL_MODEL}"
    echo "  Download from https://smpl.is.tue.mpg.de/"
    echo "  Then convert: python tools/convert_smpl_npz_to_pkl.py --input SMPL_NEUTRAL.npz --output ${SMPL_MODEL}"
    exit 1
fi

CHECKPOINT="${SAM3D_DIR}/checkpoints/sam-3d-body-dinov3/model.ckpt"
if [ ! -f "${CHECKPOINT}" ]; then
    echo "[ERROR] SAM3D checkpoint not found: ${CHECKPOINT}"
    echo "  Request access + download from https://huggingface.co/facebook/sam-3d-body-dinov3"
    exit 1
fi

# --- paths -------------------------------------------------------------------
IMAGES_DIR="${SAM3D_DIR}/datasets/images"
PKL_OUTPUTS_DIR="${SAM3D_DIR}/outputs/pkl_outputs"
TEMP_EXTRACT_DIR="${SAM3D_DIR}/outputs/_tmp_extracted"
MHR_SEQ_DIR="${SAM3D_DIR}/outputs/sequences"
OUT_DIR="${SAM3D_DIR}/outputs/smpl_sequences"
MHR_MODEL="${SAM3D_DIR}/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
MHR_CONV_DIR="${MHR_DIR}/tools/mhr_smpl_conversion"
ATHLETE_CONFIG="${SAM3D_DIR}/outputs/athlete_target_config.json"

CONDA_ENV="${CONDA_ENV:-sam_3d_body}"
DEVICE="${DEVICE:-cpu}"
VIDEO_FPS="${VIDEO_FPS:-30}"
BBOX_THRESH="${BBOX_THRESH:-0.8}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FOV_NAME="${FOV_NAME:-moge2}"
FOV_ONCE="${FOV_ONCE:-1}"

# --- athlete list ------------------------------------------------------------
if [ $# -ge 1 ]; then
    ATHLETES=("$@")
else
    mapfile -t ATHLETES < <(
        for f in "${DATASET_DIR}"/*.mp4; do
            [ -f "$f" ] && basename "$f" .mp4
        done | sort
    )
fi

if [ ${#ATHLETES[@]} -eq 0 ]; then
    echo "[ERROR] No athletes found. Pass names as args or set DATASET_DIR to a folder with .mp4 files."
    exit 1
fi

mkdir -p "${PKL_OUTPUTS_DIR}" "${OUT_DIR}"

echo "===================================================="
echo "SAM3D + MHR->SMPL  |  athletes: ${ATHLETES[*]}"
echo "  SAM3D_DIR : ${SAM3D_DIR}"
echo "  out_dir   : ${OUT_DIR}"
echo "  device    : ${DEVICE}  |  fov: ${FOV_NAME:-off}  |  skip_existing: ${SKIP_EXISTING}"
echo "===================================================="

# --- helpers -----------------------------------------------------------------
run_conda() { conda run --no-capture-output -n "${CONDA_ENV}" python "$@"; }

count_files() {
    local dir="$1" pattern="$2"
    [ -d "${dir}" ] && find "${dir}" -maxdepth 1 -name "${pattern}" | wc -l || echo 0
}

get_strategy() {
    local ath="$1"
    [ -f "${ATHLETE_CONFIG}" ] && python3 -c "
import json
try:
    print(json.load(open('${ATHLETE_CONFIG}')).get('${ath}', 'largest_bbox'))
except Exception:
    print('largest_bbox')
" || echo "largest_bbox"
}

FAILED=()
total=${#ATHLETES[@]}
count=0

for ATH in "${ATHLETES[@]}"; do
    count=$((count + 1))
    echo ""
    echo "---- [$count/$total] ${ATH} ----------------------------------------"

    ATH_IMAGES_DIR="${IMAGES_DIR}/${ATH}"
    PKL_DIR="${PKL_OUTPUTS_DIR}/${ATH}/pkl_files"
    EXTRACT_DIR="${TEMP_EXTRACT_DIR}/${ATH}"
    ATH_OUT_DIR="${OUT_DIR}/${ATH}"
    RAW_PKL="${ATH_OUT_DIR}/smpl_raw.pkl"
    FINAL_PKL="${ATH_OUT_DIR}/smpl_sequence.pkl"
    MHR_SEQ_PKL="${MHR_SEQ_DIR}/${ATH}/sequence_mhr.pkl"
    VIDEO_FILE="${DATASET_DIR}/${ATH}.mp4"

    if [ ! -f "${VIDEO_FILE}" ]; then
        echo "  [WARN] video not found: ${VIDEO_FILE} -- skipping"
        FAILED+=("${ATH}:no_video"); continue
    fi

    # step 0a: extract frames
    N_FRAMES=$(count_files "${ATH_IMAGES_DIR}" "*.jpg")
    if [ "${SKIP_EXISTING}" = "1" ] && [ "${N_FRAMES}" -gt 0 ]; then
        echo "  [skip 0a] ${N_FRAMES} frames already extracted"
    else
        echo "  [step 0a] extracting frames -> ${ATH_IMAGES_DIR}"
        mkdir -p "${ATH_IMAGES_DIR}"
        ffmpeg -y -i "${VIDEO_FILE}" -vf "fps=${VIDEO_FPS}" -q:v 2 -start_number 0 \
            "${ATH_IMAGES_DIR}/%05d.jpg" -loglevel warning \
            || { echo "  [ERROR] frame extraction failed"; FAILED+=("${ATH}:frames"); continue; }
        echo "  [step 0a] done -- $(count_files "${ATH_IMAGES_DIR}" "*.jpg") frames"
    fi

    # step 0b: SAM3D detection + MoGe-2
    N_PKLS=$(count_files "${PKL_DIR}" "*.pkl")
    if [ "${SKIP_EXISTING}" = "1" ] && [ "${N_PKLS}" -gt 0 ]; then
        echo "  [skip 0b] ${N_PKLS} detection pkls already exist"
    else
        echo "  [step 0b] SAM3D detection (MoGe-2=${FOV_NAME:-off})"
        FOV_ARGS=()
        [ -n "${FOV_NAME}" ] && FOV_ARGS+=(--fov_name "${FOV_NAME}")
        [ "${FOV_ONCE}" = "1" ] && [ -n "${FOV_NAME}" ] && FOV_ARGS+=(--fov_once)
        conda run --no-capture-output -n "${CONDA_ENV}" \
            python "${SAM3D_DIR}/demo.py" \
            --image_folder    "${ATH_IMAGES_DIR}" \
            --output_folder   "${PKL_OUTPUTS_DIR}/${ATH}" \
            --checkpoint_path "${CHECKPOINT}" \
            --mhr_path        "${MHR_MODEL}" \
            --bbox_thresh     "${BBOX_THRESH}" \
            "${FOV_ARGS[@]}" \
            || { echo "  [ERROR] SAM3D detection failed"; FAILED+=("${ATH}:detection"); continue; }
        echo "  [step 0b] done -- $(count_files "${PKL_DIR}" "*.pkl") pkls"
    fi

    [ ! -d "${PKL_DIR}" ] && { echo "  [ERROR] pkl_files dir missing after detection"; FAILED+=("${ATH}:no_pkl_dir"); continue; }

    mkdir -p "${ATH_OUT_DIR}"
    if [ "${SKIP_EXISTING}" = "1" ] && [ -f "${FINAL_PKL}" ]; then
        echo "  [skip] smpl_sequence.pkl already exists"
        continue
    fi

    # step 1: select target-athlete detection per frame
    echo "  [step 1] selecting target-athlete frames (strategy: $(get_strategy "${ATH}"))"
    mkdir -p "${EXTRACT_DIR}"
    run_conda "${CODES_DIR}/tools/extract_frames_for_conversion.py" \
        --in_dir "${PKL_DIR}" --out_dir "${EXTRACT_DIR}" \
        --strategy "$(get_strategy "${ATH}")" --skip_existing "${SKIP_EXISTING}" \
        || { echo "  [ERROR] step 1 failed"; FAILED+=("${ATH}:step1"); continue; }
    echo "  [step 1] done -- $(count_files "${EXTRACT_DIR}" "*.pkl") frames"

    # step 2: MHR -> SMPL (pixi)
    echo "  [step 2] MHR -> SMPL conversion (pixi)"
    if [ "${SKIP_EXISTING}" = "1" ] && [ -f "${RAW_PKL}" ]; then
        echo "  [skip] smpl_raw.pkl already exists"
    else
        ( cd "${MHR_CONV_DIR}" && \
          pixi run python "${CODES_DIR}/tools/convert_mhr_to_smpl_and_rots_nrdf.py" \
              --in_dir "${EXTRACT_DIR}" --out_pkl "${RAW_PKL}" \
              --smpl_model_path "${SMPL_MODEL}" \
              --mhr_assets "${MHR_DIR}/assets" \
              --gender neutral --device "${DEVICE}" \
              --method pytorch --single_identity 1 --keep_src 0 --return_errors 1 \
        ) || { echo "  [ERROR] step 2 failed"; FAILED+=("${ATH}:step2"); continue; }
        echo "  [step 2] done"
    fi

    # step 3: add SMPL 2D/3D keypoints
    echo "  [step 3] adding SMPL keypoints"
    MHR_SEQ_ARG=""
    [ -f "${MHR_SEQ_PKL}" ] && MHR_SEQ_ARG="--mhr_seq_pkl ${MHR_SEQ_PKL}"
    run_conda "${CODES_DIR}/tools/add_smpl_kpts_to_sequence.py" \
        --smpl_pkl "${RAW_PKL}" --smpl_model_path "${SMPL_MODEL}" \
        --out_pkl "${FINAL_PKL}" --device cpu --gender neutral \
        ${MHR_SEQ_ARG} \
        || { echo "  [ERROR] step 3 failed"; FAILED+=("${ATH}:step3"); continue; }
    echo "  [step 3] done -- ${FINAL_PKL}"

done

# --- summary -----------------------------------------------------------------
echo ""
echo "===================================================="
for ATH in "${ATHLETES[@]}"; do
    FINAL_PKL="${OUT_DIR}/${ATH}/smpl_sequence.pkl"
    if [ -f "${FINAL_PKL}" ]; then
        echo "  [OK]     ${ATH} -> $(du -h "${FINAL_PKL}" | cut -f1)"
    else
        echo "  [FAILED] ${ATH}"
    fi
done
[ ${#FAILED[@]} -gt 0 ] && echo "  failed steps: ${FAILED[*]}"
echo "===================================================="
echo ""
echo "Next: bash scripts/run_rollout_all_athletes.sh ${ATHLETES[*]}"
