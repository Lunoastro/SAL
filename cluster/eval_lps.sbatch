#!/usr/bin/bash -l
#SBATCH --job-name=sal-lps-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=A100:1
#SBATCH --constraint=GPUMEM80GB
#SBATCH --mem=32G
#SBATCH --time=23:50:00
#SBATCH --output=eval/slurm-%j.out
#SBATCH --error=eval/slurm-%j.err

set -euo pipefail

[[ $# -eq 2 ]] || { echo "Usage: sbatch cluster/eval_lps.sbatch RUN PART" >&2; exit 2; }
run="$1"
part="$2"
case "$part" in
    cf|cp|oa) ;;
    *) echo "Part must be cf, cp, or oa" >&2; exit 2 ;;
esac

case "$run" in
    S0) run_dir="S0/run1"; model="baseline/pool_xlsr"
        file="epoch_010-step_008734-eer_1.0729.ckpt" ;;
    S1) run_dir="S1/run1"; model="baseline/pool_xlsr_weighted"
        file="epoch_010-step_008734-eer_1.0663.ckpt" ;;
    S2) run_dir="S2/run1"; model="seq/seq_cf_weighted"
        file="epoch_010-step_008734-eer_1.0879.ckpt" ;;
    S3) run_dir="S3/run1"; model="seq/seq_cf_weighted"
        file="epoch_010-step_008734-eer_2.6435.ckpt" ;;
    S4) run_dir="S4/run1"; model="transition/seq_cf_2labels_2loss"
        file="epoch_024-step_019850-eer_2.4853.ckpt" ;;
    S5) run_dir="S5/run1"; model="spl/seq_cf_8labels_2loss"
        file="epoch_013-step_011116-eer_2.4391.ckpt" ;;
    S6) run_dir="S6/run1"; model="sal/seq_cf_8labels_2loss"
        file="epoch_013-step_011116-eer_2.4567.ckpt" ;;
    S7) run_dir="S7/run4"; model="sal/seq_cf_8labels_2loss"
        file="epoch_012-step_010322-eer_2.4693.ckpt" ;;
    S8) run_dir="S8/run1"; model="sal/seq_cf_8labels_2loss"
        file="epoch_010-step_008734-eer_2.4932.ckpt" ;;
    WavLM) run_dir="wavlm-cf/run4"; model="sal/wavlm_cf"
        file="epoch_025-step_020644-eer_2.5456.ckpt" ;;
    *) echo "Unknown run: $run" >&2; exit 2 ;;
esac

repo="${SLURM_SUBMIT_DIR:?Submit from the SAL repository root}"
cd "$repo"
[[ -f src/eval.py && -f configs/eval_lps.yaml ]] ||
    { echo "Submit from the SAL repository root" >&2; exit 1; }
[[ -f .env ]] || { echo "Missing .env" >&2; exit 1; }
[[ -r cluster/sal.sif ]] || { echo "Missing cluster/sal.sif" >&2; exit 1; }

base="${SAL_LOG_DIR:?Export SAL_LOG_DIR before submitting this job}"
checkpoint="$base/$run_dir/train/checkpoints/$file"
output_dir="$base/$run_dir/eval_lps/$part"
[[ -r "$checkpoint" ]] || { echo "Checkpoint missing for $run" >&2; exit 1; }

module load apptainer
# Reserve atomically
mkdir -p "$(dirname "$output_dir")"
mkdir "$output_dir" || { echo "Cannot create evaluation directory; check for an earlier run" >&2; exit 1; }
mkdir "$output_dir/checkpoints"
ln -s "$checkpoint" "$output_dir/checkpoints/$file"

echo "Evaluating $run on LlamaPartialSpoof $part"
srun apptainer exec \
    --nv --cleanenv --pwd "$repo" \
    --env "PROJECT_ROOT=$repo" \
    --env "SAL_EVAL_CHECKPOINT=$output_dir/checkpoints/$file" \
    "$repo/cluster/sal.sif" \
    python -u src/eval.py --config-name eval_lps \
    "model=$model" \
    "data.part=$part" \
    "hydra.run.dir=$output_dir" \
    "+trainer.enable_progress_bar=false"
