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

SPLIT="${SPLIT:-eval}"
RESOLUTION="${RESOLUTION:-0.16}"
OUT_DIR="${REPO_ROOT}/eval/visualize/${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"

# Frozen per-model thresholds on the logit[1]-logit[0] scale, taken from that model's own
# PS-dev EER point (thresholds.json). Set THRESHOLD_S3 / THRESHOLD_S7 (THRESHOLD for MODEL_CKPT).
# Missing -> job stops; ALLOW_ARGMAX_THRESHOLD=1 falls back to 0.5.
threshold_args() {
    local tag="$1" value="$2"
    if [[ -n "$value" ]]; then
        echo "Threshold (${tag}): logit-diff ${value} (frozen from ${tag}'s PS dev set)" >&2
        printf '%s\n' --logit_threshold "$value"
    elif [[ "${ALLOW_ARGMAX_THRESHOLD:-0}" == 1 ]]; then
        echo "WARNING (${tag}): no frozen threshold set; falling back to P(real) 0.5 (argmax)." >&2
        echo "WARNING (${tag}): 0.5 confounds calibration shift with localisation quality; do not use these plots for PS-vs-LPS claims." >&2
        printf '%s\n' --threshold 0.5
    else
        echo "ERROR (${tag}): no frozen threshold. Set THRESHOLD_${tag} (or THRESHOLD for MODEL_CKPT)" >&2
        echo "       to that model's PS-dev EER threshold from thresholds.json, or ALLOW_ARGMAX_THRESHOLD=1." >&2
        exit 1
    fi
}

# SAMPLE_LIST=<make_sample_list.py output> plots every listed utterance (audio from AUDIO_DIR);
# otherwise one utterance (SAMPLE_ID / AUDIO_PATH).
SAMPLE_LIST="${SAMPLE_LIST:-}"
AUDIO_DIR="${AUDIO_DIR:-$PARTIALSPOOF_ROOT/$SPLIT/con_wav}"
if [[ -n "$SAMPLE_LIST" ]]; then
    [[ -r "$SAMPLE_LIST" ]] || { echo "ERROR: SAMPLE_LIST not readable: $SAMPLE_LIST" >&2; exit 1; }
    if [[ $# -ge 3 ]]; then
        echo "ERROR: AUDIO_PATH argument cannot be combined with SAMPLE_LIST (use AUDIO_DIR)" >&2
        exit 1
    fi
    cp "$SAMPLE_LIST" "${OUT_DIR}/sample_list.tsv"
fi

AUDIO_PATH="${3:-$PARTIALSPOOF_ROOT/$SPLIT/con_wav/$SAMPLE_ID.wav}"
LABEL_ARGS=()
if [[ -n "${LABELS_PATH:-}" ]]; then
    LABEL_ARGS=(--labels_path "$LABELS_PATH")
elif [[ $# -lt 3 ]]; then
    LABEL_ARGS=(--labels_path "$PARTIALSPOOF_ROOT/segment_labels/${SPLIT}_seglab_${RESOLUTION}.npy")
fi

run_localise() {
    local tag="$1" ckpt="$2" model_type="$3" threshold="$4"
    [[ -r "$ckpt" ]] || { echo "ERROR: Model checkpoint not readable: $ckpt" >&2; exit 1; }
    local thr_args sample_args
    mapfile -t thr_args < <(threshold_args "$tag" "$threshold")
    # process substitution hides exit 1 from threshold_args, so check the result here
    [[ ${#thr_args[@]} -eq 2 ]] || exit 1
    if [[ -n "$SAMPLE_LIST" ]]; then
        sample_args=(--sample_list "$SAMPLE_LIST" --audio_dir "$AUDIO_DIR"
                     --output_path "$OUT_DIR/localisation_${tag}_{sample_id}.png")
    else
        sample_args=(--audio_path "$AUDIO_PATH" --sample_id "$SAMPLE_ID"
                     --output_path "$OUT_DIR/localisation_${tag}_${SAMPLE_ID}.png")
    fi
    echo "Running localise (${tag}): ${ckpt} (${model_type})"
    srun apptainer exec \
        --nv \
        --cleanenv \
        --pwd "${REPO_ROOT}" \
        --env "PROJECT_ROOT=${PROJECT_ROOT}" \
        --env "MPLBACKEND=Agg" \
        "${SAL_IMAGE}" \
        python -u scripts/visualize/gradcam.py --localisation \
            "${sample_args[@]}" "${LABEL_ARGS[@]}" "${thr_args[@]}" \
            --ckpt "$XLSR_CHECKPOINT" --model_ckpt "$ckpt" \
            --model_type "$model_type" --ssl_encoder xlsr --ssl_mode s3prl_weighted \
            --resolution "$RESOLUTION" --device cuda
}

# Check all thresholds before the first GPU run (direct call, so exit 1 stops the job).
if [[ -n "${MODEL_CKPT:-}" ]]; then
    [[ -n "${THRESHOLD:-}" || "${ALLOW_ARGMAX_THRESHOLD:-0}" == 1 ]] || threshold_args "${MODEL_TAG:-custom}" ""
else
    [[ -n "${THRESHOLD_S3:-}" || "${ALLOW_ARGMAX_THRESHOLD:-0}" == 1 ]] || threshold_args S3 ""
    [[ -n "${THRESHOLD_S7:-}" || "${ALLOW_ARGMAX_THRESHOLD:-0}" == 1 ]] || threshold_args S7 ""
fi

if [[ -n "${MODEL_CKPT:-}" ]]; then
    run_localise "${MODEL_TAG:-custom}" "$MODEL_CKPT" "${MODEL_TYPE:-ssl_seq_8labels_2loss}" "${THRESHOLD:-}"
else
    run_localise S3 "${SAL_LOG_DIR}/S3/run1/train/checkpoints/epoch_010-step_008734-eer_2.6435.ckpt" ssl_seq "${THRESHOLD_S3:-}"
    run_localise S7 "${SAL_LOG_DIR}/S7/run4/train/checkpoints/epoch_012-step_010322-eer_2.4693.ckpt" ssl_seq_8labels_2loss "${THRESHOLD_S7:-}"
fi
