#!/bin/bash
# CPU training-cost baseline: the edge-training proxy.
#
# The chapter argues (Figure fig:flops_flops) that FLOP counts predict latency differently on CPU
# than on accelerators. These rows let that claim be made with this thesis's own models instead of
# a citation. A server CPU is a weak proxy for an actual edge SoC or NPU -- state that in the
# methods section rather than implying it models a Jetson.
#
# NOTE: confirm the partition name before the first submission; it is the one line here that was
# not verifiable from the development machine.
#SBATCH --time=12:00:00
#SBATCH --job-name=RTCNN_Cost_CPU
#SBATCH --output=logs/%j/output_cpu.out
#SBATCH --error=logs/%j/error_cpu.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=Brain_CPU

set -euo pipefail

cd "${PROJECT_ROOT:-/SCRATCH/j20morli/Resolution_Transformer_CNN}"
source "${VENV_PATH:-.venv/bin/activate}"

export WANDB_DIR="${WANDB_DIR:-${WORK:-$HOME}/wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-10}}"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

SLURM_JOB_LABEL="${SLURM_JOB_ID:-manual}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/SCRATCH/j20morli/results_resolution/training_cost/cpu/${SLURM_JOB_LABEL}}"
mkdir -p "$OUTPUT_ROOT"

# A CPU ResNet-50 step at 224px is roughly a second, so the CPU grid uses a small batch and few
# steps. Memory columns stay empty on CPU (there is no allocator peak counter); these rows
# contribute timing only, plus cpu_max_rss_bytes as a low-confidence upper bound.
BATCH_SIZE="${BATCH_SIZE:-8}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
MEASURE_STEPS="${MEASURE_STEPS:-10}"
REPEATS="${REPEATS:-3}"

srun python3 benchmark_training_step.py \
    --families resnet vit \
    --include-resnet-depth \
    --resolutions "112 144 176 224 256 288 320 384" \
    --device cpu \
    --precision fp32 \
    --batch-size "$BATCH_SIZE" \
    --warmup-steps "$WARMUP_STEPS" \
    --measure-steps "$MEASURE_STEPS" \
    --repeats "$REPEATS" \
    --threads "$THREADS" \
    --shuffle-configs \
    --run-id "${SLURM_JOB_LABEL}_cpu" \
    --output "${OUTPUT_ROOT}/training_cost_cpu.csv"

echo "wrote ${OUTPUT_ROOT}/training_cost_cpu.csv"
