#!/usr/bin/bash -l
#SBATCH --job-name=sal-visualize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=A100:1
#SBATCH --constraint=GPUMEM80GB
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=eval/visualize/slurm-%j.out
#SBATCH --error=eval/visualize/slurm-%j.err

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

# Load environment variables without overriding exported ones
set -a
# shellcheck disable=SC1091
source .env
set +a

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
MODE="${1:-all}"
SAMPLE_ID="${2:-}"

# Paths for S7_fixed checkpoint
CKPT_DIR="${SAL_LOG_DIR}/S7_fixed/run1/train/checkpoints"
MODEL_CKPT="${CKPT_DIR}/epoch_022-step_018262-eer_0.9787.ckpt"
if [[ ! -f "${MODEL_CKPT}" ]]; then
    # Fallback to historical S7 run4 if S7_fixed is not found
    echo "Notice: S7_fixed checkpoint not found at ${MODEL_CKPT}. Checking historical S7..."
    MODEL_CKPT="${SAL_LOG_DIR}/S7/run4/train/checkpoints/epoch_012-step_010322-eer_2.4693.ckpt"
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
    RESULT_FILE="${SAL_LOG_DIR}/S7/run4/train/result.txt"
    if [[ ! -f "${RESULT_FILE}" ]]; then
        # Try finding result.txt in S5 or S7_fixed
        RESULT_FILE="$(find "${SAL_LOG_DIR}" -name "result.txt" | head -n 1 || true)"
    fi

    if [[ -n "${RESULT_FILE}" && -f "${RESULT_FILE}" ]]; then
        echo "Found training result file: ${RESULT_FILE}"
        run_in_sal scripts/visualize/part_error.py \
            --file "${RESULT_FILE}" \
            --out "${OUT_DIR}/positional_error_rates.png"
        echo "Positional error plot saved to ${OUT_DIR}/positional_error_rates.png"
    else
        echo "Warning: result.txt not found in ${SAL_LOG_DIR}. Skipping part_error plot."
    fi
fi

# ---------------------------------------------------------
# Step 1b & 2: Grad-CAM and Direct Prediction Timeline
# ---------------------------------------------------------
if [[ "${MODE}" == "gradcam" || "${MODE}" == "all" ]]; then
    echo ""
    echo "--- Step 1b / 2: Generating Grad-CAM and Prediction Timeline (gradcam.py) ---"
    
    LABELS_PATH="${PARTIALSPOOF_ROOT}/segment_labels/eval_seglab_0.16.npy"
    if [[ ! -f "${LABELS_PATH}" ]]; then
        # Fallback to dev split if eval labels are elsewhere
        LABELS_PATH="${PARTIALSPOOF_ROOT}/segment_labels/dev_seglab_0.16.npy"
    fi

    # If no sample ID is provided, query the first sample from labels dict
    if [[ -z "${SAMPLE_ID}" ]]; then
        echo "No SAMPLE_ID provided. Auto-selecting first sample from ${LABELS_PATH}..."
        SAMPLE_ID=$(run_in_sal -c "import numpy as np; labels = np.load('${LABELS_PATH}', allow_pickle=True).item(); print(list(labels.keys())[0])")
        echo "Selected SAMPLE_ID: ${SAMPLE_ID}"
    fi

    # Find audio path for the sample
    AUDIO_PATH="${PARTIALSPOOF_ROOT}/eval/${SAMPLE_ID}.wav"
    if [[ ! -f "${AUDIO_PATH}" ]]; then
        AUDIO_PATH="${PARTIALSPOOF_ROOT}/dev/${SAMPLE_ID}.wav"
    fi

    if [[ ! -f "${AUDIO_PATH}" ]]; then
        echo "ERROR: Audio file not found for sample ${SAMPLE_ID} in eval or dev directory." >&2
        exit 1
    fi

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
