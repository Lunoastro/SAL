#!/usr/bin/bash -l
#SBATCH --job-name=sal-visualize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=A100:1
#SBATCH --constraint=GPUMEM80GB
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-visualize-%j.out
#SBATCH --error=slurm-visualize-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-}"
if [[ -z "${REPO_ROOT}" || ! -f "${REPO_ROOT}/src/train.py" ]]; then
    echo "ERROR: Submit from the SAL repository root: sbatch cluster/visualize.sbatch [MODE] [SAMPLE_ID]" >&2
    exit 1
fi
cd "${REPO_ROOT}"
export PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}}"

# Check .env
if [[ ! -f ".env" ]]; then
    echo "ERROR: Root '.env' file not found at ${REPO_ROOT}/.env!" >&2
    exit 1
fi

# Load the remote environment configuration
set -a
# shellcheck disable=SC1091
source .env
set +a
export PROJECT_ROOT="${REPO_ROOT}"

# Load Apptainer module
module load apptainer
if ! command -v apptainer &>/dev/null; then
    echo "ERROR: Apptainer is unavailable after loading its cluster module." >&2
    exit 1
fi

SAL_IMAGE="${SAL_IMAGE:-${REPO_ROOT}/cluster/sal.sif}"
if [[ ! -r "${SAL_IMAGE}" ]]; then
    echo "ERROR: SAL container image not found: ${SAL_IMAGE}" >&2
    exit 1
fi

# Ensure output directory exists
OUT_DIR="${REPO_ROOT}/eval/visualize"
mkdir -p "${OUT_DIR}"

# Determine mode and sample ID from arguments
# Usage: sbatch cluster/visualize.sbatch [gradcam|part_error|all] [SAMPLE_ID]
MODE="${1:-gradcam}"
SAMPLE_ID="${2:-CON_E_0000096}"
case "$MODE" in gradcam|part_error|all) ;; *) echo "ERROR: Invalid mode: $MODE" >&2; exit 1;; esac

# Historical S7 (<-- default )
MODEL_CKPT="${MODEL_CKPT:-${SAL_LOG_DIR}/S7/run4/train/checkpoints/epoch_012-step_010322-eer_2.4693.ckpt}"
if [[ "$MODE" != part_error && ! -r "$MODEL_CKPT" ]]; then
    echo "ERROR: Model checkpoint not readable: $MODEL_CKPT" >&2
    exit 1
fi

echo "============================================================"
echo "SAL Visualization Job started on $(hostname) at $(date)"
echo "Mode: ${MODE}"
echo "Model Checkpoint: ${MODEL_CKPT}"
echo "============================================================"

run_in_sal() {
    srun apptainer exec \
        --nv \
        --cleanenv \
        --pwd "${REPO_ROOT}" \
        --env "PROJECT_ROOT=${PROJECT_ROOT}" \
        "${SAL_IMAGE}" \
        python -u "$@"
}

# ---------------------------------------------------------
# Step 1a: Positional Error Rates (Start, Middle, End, Unit)
# ---------------------------------------------------------
if [[ "${MODE}" == "part_error" || "${MODE}" == "all" ]]; then
    echo ""
    echo "--- Step 1a: Generating Positional Error Rate Plot (part_error.py) ---"
    : "${RESULT_FILE:?Set RESULT_FILE to verified positional statistics (indices 0..7), not S7 aggregate metrics}"

    if [[ -n "${RESULT_FILE}" && -f "${RESULT_FILE}" ]]; then
        echo "Found training result file: ${RESULT_FILE}"
        run_in_sal scripts/visualize/part_error.py \
            --file "${RESULT_FILE}" \
            --out "${OUT_DIR}/positional_error_rates.png"
        echo "Positional error plot saved to ${OUT_DIR}/positional_error_rates.png"
    else
        echo "ERROR: Result file not found: $RESULT_FILE" >&2
        exit 1
    fi
fi

# ---------------------------------------------------------
# Step 1b & 2: Grad-CAM and Direct Prediction Timeline
# ---------------------------------------------------------
if [[ "${MODE}" == "gradcam" || "${MODE}" == "all" ]]; then
    echo ""
    echo "--- Step 1b / 2: Generating Grad-CAM and Prediction Timeline (gradcam.py) ---"
    
    # Keep audio and labels on the same split and never substitute another split.
    SPLIT="${SPLIT:-eval}"
    case "$SPLIT" in eval|dev) ;; *) echo "ERROR: Invalid split: $SPLIT" >&2; exit 1;; esac
    LABELS_PATH="${PARTIALSPOOF_ROOT}/segment_labels/${SPLIT}_seglab_0.16.npy"
    AUDIO_PATH="${PARTIALSPOOF_ROOT}/${SPLIT}/${SAMPLE_ID}.wav"
    for file in "$LABELS_PATH" "$AUDIO_PATH" "$XLSR_CHECKPOINT"; do
        [[ -r "$file" ]] || { echo "ERROR: File not readable: $file" >&2; exit 1; }
    done

    echo "Audio: ${AUDIO_PATH}"
    echo "Labels: ${LABELS_PATH}"
    echo "Sample: ${SAMPLE_ID}"
    echo "SSL Encoder: ${XLSR_CHECKPOINT}"

    run_in_sal scripts/visualize/gradcam.py \
        --audio_path "${AUDIO_PATH}" \
        --labels_path "${LABELS_PATH}" \
        --sample_id "${SAMPLE_ID}" \
        --ckpt "${XLSR_CHECKPOINT}" \
        --model_ckpt "${MODEL_CKPT}" \
        --model_type ssl_seq_8labels_2loss \
        --ssl_mode s3prl_weighted \
        --device cuda \
        --output_path "${OUT_DIR}/gradcam_${SAMPLE_ID}.png"

    echo "Grad-CAM & timeline plot saved to: ${OUT_DIR}/gradcam_${SAMPLE_ID}.png"
fi

echo ""
echo "=== SAL Visualization Completed successfully at $(date) ==="
