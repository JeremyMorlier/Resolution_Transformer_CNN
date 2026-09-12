#!/bin/bash
# Measured training step time and training memory across the classification scaling grid.
#
# One array task per precision. The whole 41-point grid takes minutes per precision, not the
# 48 h that measure_scaling_training_time.sh needs for the same coverage, because no dataset is
# read: the benchmark feeds synthetic batches and therefore measures the model, not the input
# pipeline.
#
# Set ISOLATE=1 to run one process per grid point instead (slower, but cuDNN autotune caches and
# allocator fragmentation cannot leak between configs). Use that for the numbers that go in the
# thesis; the single-process sweep is for quick iteration.
#SBATCH --time=06:00:00
#SBATCH --job-name=RTCNN_Cost_GPU
#SBATCH --output=logs/%j/output_%a.out
#SBATCH --error=logs/%j/error_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=Brain_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-1

set -euo pipefail

cd "${PROJECT_ROOT:-/SCRATCH/j20morli/Resolution_Transformer_CNN}"
source "${VENV_PATH:-.venv/bin/activate}"

export WANDB_DIR="${WANDB_DIR:-${WORK:-$HOME}/wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"
# Reduces fragmentation-induced OOM at the large ViT points. Recorded in the CSV, because peak
# reserved memory is not comparable across different allocator configurations.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PRECISIONS=("fp32" "amp_bf16")
PRECISION="${PRECISIONS[${SLURM_ARRAY_TASK_ID:-0}]}"

SLURM_JOB_LABEL="${SLURM_ARRAY_JOB_ID:-manual}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/SCRATCH/j20morli/results_resolution/training_cost/gpu/${SLURM_JOB_LABEL}}"
mkdir -p "$OUTPUT_ROOT"

# Batch size is fixed across the entire grid on purpose: points measured at different batch
# sizes are not comparable, and the head-to-head between resolution and model scaling is the
# whole point. 64 is chosen so vit_hidden_1280 and vit_b_16@384 still fit on a 40 GB A100.
BATCH_SIZE="${BATCH_SIZE:-64}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
MEASURE_STEPS="${MEASURE_STEPS:-30}"
REPEATS="${REPEATS:-3}"
EPOCHS="${EPOCHS:-120}"

# These grid flags MUST match the ones passed to scaling_resources.py and to the other benchmark
# scripts: --config-index is an index into this enumeration, and the CSV join depends on the two
# grids covering the same points.
GRID_ARGS=(
    --families resnet vit
    --include-resnet-depth
    --resolutions "112 144 176 224 256 288 320 384"
)

COMMON_ARGS=(
    "${GRID_ARGS[@]}"
    --device cuda
    --precision "$PRECISION"
    --batch-size "$BATCH_SIZE"
    --warmup-steps "$WARMUP_STEPS"
    --measure-steps "$MEASURE_STEPS"
    --epochs "$EPOCHS"
    --tf32 on
    --cudnn-benchmark on
    --shuffle-configs
    --settle-seconds 2
    --run-id "${SLURM_JOB_LABEL}_${PRECISION}"
    --dump-raw-steps "${OUTPUT_ROOT}/raw_steps_${PRECISION}.jsonl"
)

OUTPUT_CSV="${OUTPUT_ROOT}/training_cost_${PRECISION}.csv"

if [ "${ISOLATE:-0}" = "1" ]; then
    NUM_CONFIGS=$(srun python3 benchmark_training_step.py "${GRID_ARGS[@]}" --list-configs | tail -n1 | awk '{print $1}')
    echo "Running ${NUM_CONFIGS} configs, one process each, precision ${PRECISION}"
    for CONFIG_INDEX in $(seq 0 $((NUM_CONFIGS - 1))); do
        for REPEAT in $(seq 1 "$REPEATS"); do
            srun python3 benchmark_training_step.py "${COMMON_ARGS[@]}" \
                --config-index "$CONFIG_INDEX" \
                --seed "$REPEAT" \
                --output "$OUTPUT_CSV" \
                --append
        done
    done
else
    srun python3 benchmark_training_step.py "${COMMON_ARGS[@]}" \
        --repeats "$REPEATS" \
        --output "$OUTPUT_CSV"
fi

echo "wrote ${OUTPUT_CSV}"
