#!/usr/bin/bash -l
#SBATCH --job-name=sal-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=A100:1
#SBATCH --constraint=GPUMEM80GB
#SBATCH --mem=32G
#SBATCH --time=23:50:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-}"
if [[ -z "${REPO_ROOT}" || ! -f "${REPO_ROOT}/src/train.py" ]]; then
    echo "ERROR: submit from the SAL repository root: sbatch cluster/train.sbatch ..."
    exit 1
fi
cd "${REPO_ROOT}"
export PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}}"

echo "=== SAL Training Job Started on $(hostname) at $(date) ==="
echo "Working directory: ${REPO_ROOT}"

# Verify root .env exists
if [[ ! -f ".env" ]]; then
    echo "ERROR: Root '.env' file not found at ${REPO_ROOT}/.env!"
    echo "Please copy cluster/.env.example to .env and configure remote paths."
    exit 1
fi

module load apptainer

if ! command -v apptainer &>/dev/null; then
    echo "ERROR: Apptainer is unavailable after loading its cluster module."
    exit 1
fi

SAL_IMAGE="${SAL_IMAGE:-${REPO_ROOT}/cluster/sal.sif}"
if [[ ! -r "${SAL_IMAGE}" ]]; then
    echo "ERROR: SAL container image not found: ${SAL_IMAGE}"
    echo "Build cluster/sal.def into cluster/sal.sif before submitting this job."
    exit 1
fi

run_in_sal() {
    srun apptainer exec \
        --nv \
        --cleanenv \
        --pwd "${REPO_ROOT}" \
        --env "PROJECT_ROOT=${PROJECT_ROOT}" \
        "${SAL_IMAGE}" \
        python -u "$@"
}

# Parse the script-only flag before forwarding the remaining arguments to Hydra.
PREFLIGHT_ONLY=false
if [[ "${1:-}" == "--preflight-only" ]]; then
    PREFLIGHT_ONLY=true
    shift
fi

ARGS=("$@")
if [[ ${#ARGS[@]} -eq 0 ]]; then
    ARGS=("experiment=S0")
fi

# Run preflight environment check with GPU
echo "Running GPU environment check..."
run_in_sal scripts/env_check.py --mode preflight

echo "Checking Hydra configuration: ${ARGS[*]}"
run_in_sal src/train.py "${ARGS[@]}" --cfg job --resolve >/dev/null

if [[ "${PREFLIGHT_ONLY}" == true ]]; then
    echo "=== SAL Preflight Completed at $(date) ==="
    exit 0
fi

echo "Launching training with arguments: ${ARGS[*]}"
run_in_sal src/train.py "${ARGS[@]}"

echo "=== SAL Training Job Completed at $(date) ==="
