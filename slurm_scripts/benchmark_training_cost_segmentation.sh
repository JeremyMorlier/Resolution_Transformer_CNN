#!/bin/bash
# Measured training cost for dense prediction (RegSeg on Cityscapes geometry).
#
# Segmentation is the regime where activation memory dominates most: inputs are R x 2R and the
# decoder keeps high-resolution feature maps, so the resolution lever moves a much larger share of
# total training memory than it does for classification.
#SBATCH --time=04:00:00
#SBATCH --job-name=RTCNN_Cost_SEG
#SBATCH --output=logs/%j/output_seg.out
#SBATCH --error=logs/%j/error_seg.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=Brain_GPU
#SBATCH --gres=gpu:a100:1

set -euo pipefail

cd "${PROJECT_ROOT:-/SCRATCH/j20morli/Resolution_Transformer_CNN}"
source "${VENV_PATH:-.venv/bin/activate}"

export WANDB_DIR="${WANDB_DIR:-${WORK:-$HOME}/wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SLURM_JOB_LABEL="${SLURM_JOB_ID:-manual}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/SCRATCH/j20morli/results_resolution/training_cost/segmentation/${SLURM_JOB_LABEL}}"
mkdir -p "$OUTPUT_ROOT"

# Batch 8 matches the RegSeg recipe in the README. Cityscapes has 2975 training images, so the
# epoch extrapolation uses that rather than the ImageNet count.
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-1000}"

srun python3 benchmark_training_step.py \
    --families regseg \
    --segmentation-resolutions "256 384 512 768 1024" \
    --device cuda \
    --precision "${PRECISION:-fp32}" \
    --batch-size "$BATCH_SIZE" \
    --warmup-steps 10 \
    --measure-steps 20 \
    --repeats "${REPEATS:-3}" \
    --epochs "$EPOCHS" \
    --lr 0.05 \
    --weight-decay 0.0001 \
    --optimizer sgd_momentum \
    --no-clip-grad-norm \
    --tf32 on \
    --cudnn-benchmark on \
    --shuffle-configs \
    --settle-seconds 2 \
    --run-id "${SLURM_JOB_LABEL}_seg" \
    --output "${OUTPUT_ROOT}/training_cost_segmentation.csv"

echo "wrote ${OUTPUT_ROOT}/training_cost_segmentation.csv"
