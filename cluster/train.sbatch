#!/usr/bin/env bash
#SBATCH --job-name=sal-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=23:50:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# Submit this script from the repository root. Slurm may execute a spooled copy, (<-- Need to find that out somehow)
# so the script's own location is not a reliable way to find the checkout.
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

if ! command -v conda &>/dev/null; then
    echo "ERROR: conda is unavailable. Load it before submitting this job."
    exit 1
fi
eval "$(conda shell.bash hook)"
conda activate sal

# Run preflight environment check with GPU
echo "Running GPU environment check..."
python -u scripts/env_check.py --mode preflight

# Launch training with passed CLI arguments (should default to experiment=S0 if not specified)
ARGS=("$@")
if [[ ${#ARGS[@]} -eq 0 ]]; then
    ARGS=("experiment=S0")
fi

echo "Launching training with arguments: ${ARGS[*]}"
python -u src/train.py "${ARGS[@]}"

echo "=== SAL Training Job Completed at $(date) ==="
