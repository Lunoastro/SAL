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
    echo "ERROR: Submit from the SAL repository root: sbatch cluster/visualize.sbatch [localise] [SAMPLE_ID] [AUDIO_PATH]" >&2
    exit 1
fi
cd "${REPO_ROOT}"
export PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}}"

if [[ ! -f ".env" ]]; then
    echo "ERROR: Root '.env' file not found at ${REPO_ROOT}/.env!" >&2
    exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a
export PROJECT_ROOT="${REPO_ROOT}"

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

MODE="${1:-localise}"
SAMPLE_ID="${2:-CON_E_0000096}"
if [[ "$MODE" != localise ]]; then
    echo "ERROR: Invalid mode: $MODE (expected: localise)" >&2
    exit 1
fi

# Historical S7 (<-- default )
MODEL_CKPT="${MODEL_CKPT:-${SAL_LOG_DIR}/S7/run4/train/checkpoints/epoch_012-step_010322-eer_2.4693.ckpt}"
MODEL_TYPE="${MODEL_TYPE:-ssl_seq_8labels_2loss}"
SPLIT="${SPLIT:-eval}"
RESOLUTION="${RESOLUTION:-0.16}"
if [[ ! -r "$MODEL_CKPT" ]]; then
    echo "ERROR: Model checkpoint not readable: $MODEL_CKPT" >&2
    exit 1
fi

OUT_DIR="${REPO_ROOT}/eval/visualize/${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"

AUDIO_PATH="${3:-$PARTIALSPOOF_ROOT/$SPLIT/con_wav/$SAMPLE_ID.wav}"
LABEL_ARGS=()
if [[ -n "${LABELS_PATH:-}" ]]; then
    LABEL_ARGS=(--labels_path "$LABELS_PATH")
elif [[ $# -lt 3 ]]; then
    LABEL_ARGS=(--labels_path "$PARTIALSPOOF_ROOT/segment_labels/${SPLIT}_seglab_${RESOLUTION}.npy")
fi

echo "============================================================"
echo "SAL Visualization Job started on $(hostname) at $(date)"
echo "Mode: ${MODE} | Sample: ${SAMPLE_ID} | Resolution: ${RESOLUTION}s"
echo "Model Checkpoint: ${MODEL_CKPT} (${MODEL_TYPE})"
echo "============================================================"

srun apptainer exec \
    --nv \
    --cleanenv \
    --pwd "${REPO_ROOT}" \
    --env "PROJECT_ROOT=${PROJECT_ROOT}" \
    --env "MPLBACKEND=Agg" \
    "${SAL_IMAGE}" \
    python -u scripts/visualize/gradcam.py --localisation \
        --audio_path "$AUDIO_PATH" "${LABEL_ARGS[@]}" --sample_id "$SAMPLE_ID" \
        --ckpt "$XLSR_CHECKPOINT" --model_ckpt "$MODEL_CKPT" \
        --model_type "$MODEL_TYPE" --ssl_encoder xlsr --ssl_mode s3prl_weighted \
        --resolution "$RESOLUTION" --device cuda \
        --output_path "$OUT_DIR/localisation_${SAMPLE_ID}.png"
