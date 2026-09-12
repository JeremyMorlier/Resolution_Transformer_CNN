#!/bin/bash
# Validation subset for the synthetic training-cost benchmark.
#
# Trains ONE real ImageNet epoch at a handful of points and compares the measured epoch time
# against the epoch time predicted from benchmark_training_step.py. Two things come out of it:
#   1. confirmation that the synthetic step time predicts real training in the GPU-bound regime,
#      which is what legitimises using the synthetic grid everywhere else;
#   2. a measurement of the dataloader-bound regime at low resolution -- tasks 6 and 7 repeat the
#      112px point with fewer and with more workers, so the boundary can be shown to move with the
#      CPU budget rather than with the model.
#
# This replaces the 36-task measure_scaling_training_time.sh as the epoch-level experiment: that
# script spent 36 ImageNet epochs measuring what the synthetic benchmark covers in minutes, and its
# low-resolution points were dataloader bound anyway.
#
# The batch size MUST match the one used by benchmark_training_cost_gpu.sh, or the predicted and
# measured epoch times are not comparable.
#SBATCH --time=48:00:00
#SBATCH --job-name=RTCNN_Epoch_Valid
#SBATCH --output=logs/%j/output_%a.out
#SBATCH --error=logs/%j/error_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=Brain_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-7%1

set -euo pipefail

cd "${PROJECT_ROOT:-/SCRATCH/j20morli/Resolution_Transformer_CNN}"
source "${VENV_PATH:-.venv/bin/activate}"

export WANDB_DIR="${WANDB_DIR:-${WORK:-$HOME}/wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"

MEASURED_EPOCHS="${MEASURED_EPOCHS:-1}"
ESTIMATED_EPOCHS="${ESTIMATED_EPOCHS:-120}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-1281167}"
BACKWARD_MULTIPLIER="${BACKWARD_MULTIPLIER:-3.0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DATA_PATH="${DATA_PATH:-/SCRATCH/datasets/imagenet}"
SLURM_JOB_LABEL="${SLURM_ARRAY_JOB_ID:-manual}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
WORLD_SIZE="${SLURM_NTASKS:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/SCRATCH/j20morli/results_resolution/training_time/validation/$SLURM_JOB_LABEL}"

# Fields: family axis name image_size workers model-specific-values
#   ResNet values: channels_csv depths_csv
#   ViT values:    patch_size num_layers num_heads hidden_dim mlp_dim
CONFIGS=(
    "resnet resolution resnet50_c1.00_r112 112 10 64,64,128,256,512 3,4,6,3"
    "resnet resolution resnet50_c1.00_r224 224 10 64,64,128,256,512 3,4,6,3"
    "resnet resolution resnet50_c1.00_r384 384 10 64,64,128,256,512 3,4,6,3"
    "vit resolution vit_b_16_r112 112 10 16 12 12 768 3072"
    "vit resolution vit_b_16_r224 224 10 16 12 12 768 3072"
    "vit resolution vit_b_16_r384 384 10 16 12 12 768 3072"
    "resnet workers resnet50_c1.00_r112_w4 112 4 64,64,128,256,512 3,4,6,3"
    "resnet workers resnet50_c1.00_r112_w20 112 20 64,64,128,256,512 3,4,6,3"
)

CONFIG="${CONFIGS[$TASK_ID]}"
read -r FAMILY AXIS CONFIG_NAME IMAGE_SIZE WORKERS ARG1 ARG2 ARG3 ARG4 ARG5 <<< "$CONFIG"

VAL_RESIZE_SIZE=$((IMAGE_SIZE + 8))
TASK_OUTPUT_ROOT="${OUTPUT_ROOT}/${TASK_ID}_${AXIS}_${CONFIG_NAME}"
mkdir -p "$TASK_OUTPUT_ROOT"

echo "Task ${TASK_ID}: ${FAMILY} ${AXIS} ${CONFIG_NAME}, image size ${IMAGE_SIZE}, batch ${BATCH_SIZE}, workers ${WORKERS}"

if [ "$FAMILY" = "resnet" ]; then
    CHANNELS=(${ARG1//,/ })
    DEPTHS=(${ARG2//,/ })
    RUN_NAME="resnet50_resize_${IMAGE_SIZE}_${IMAGE_SIZE}_${VAL_RESIZE_SIZE}_0"
    for CHANNEL in "${CHANNELS[@]}"; do
        RUN_NAME="${RUN_NAME}_${CHANNEL}"
    done
    RUN_DIR="${TASK_OUTPUT_ROOT}/${RUN_NAME}"
    LOG_FILE="${RUN_DIR}/resolution_CNN_ViT_${RUN_NAME}.log"

    srun python3 train_classification.py \
        --model resnet50_resize \
        --epochs "$MEASURED_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --workers "$WORKERS" \
        --opt adamw \
        --lr 0.003 \
        --wd 0.3 \
        --lr_scheduler cosineannealinglr \
        --lr_warmup_epochs 0 \
        --label_smoothing 0.11 \
        --mixup_alpha 0.2 \
        --auto_augment ra \
        --clip_grad_norm 1 \
        --ra_sampler \
        --cutmix_alpha 1.0 \
        --model_ema \
        --train_crop_size "$IMAGE_SIZE" \
        --val_resize_size "$VAL_RESIZE_SIZE" \
        --val_crop_size "$IMAGE_SIZE" \
        --channels "${CHANNELS[@]}" \
        --depths "${DEPTHS[@]}" \
        --output_dir "$TASK_OUTPUT_ROOT" \
        --data_path "$DATA_PATH" \
        --world_size "$WORLD_SIZE" \
        --logger txt \
        --skip_resolution_evaluation
elif [ "$FAMILY" = "vit" ]; then
    PATCH_SIZE="$ARG1"
    NUM_LAYERS="$ARG2"
    NUM_HEADS="$ARG3"
    HIDDEN_DIM="$ARG4"
    MLP_DIM="$ARG5"
    RUN_NAME="vit_custom_${PATCH_SIZE}_${NUM_LAYERS}_${NUM_HEADS}_${HIDDEN_DIM}_${MLP_DIM}_${IMAGE_SIZE}"
    RUN_DIR="${TASK_OUTPUT_ROOT}/${RUN_NAME}"
    LOG_FILE="${RUN_DIR}/resolution_CNN_ViT_${RUN_NAME}.log"

    srun python3 train_classification.py \
        --model vit_custom \
        --epochs "$MEASURED_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --workers "$WORKERS" \
        --opt adamw \
        --lr 0.003 \
        --wd 0.3 \
        --lr_scheduler cosineannealinglr \
        --lr_warmup_epochs 0 \
        --label_smoothing 0.11 \
        --mixup_alpha 0.2 \
        --auto_augment ra \
        --clip_grad_norm 1 \
        --ra_sampler \
        --cutmix_alpha 1.0 \
        --model_ema \
        --train_crop_size "$IMAGE_SIZE" \
        --val_resize_size "$VAL_RESIZE_SIZE" \
        --val_crop_size "$IMAGE_SIZE" \
        --patch_size "$PATCH_SIZE" \
        --num_layers "$NUM_LAYERS" \
        --num_heads "$NUM_HEADS" \
        --hidden_dim "$HIDDEN_DIM" \
        --mlp_dim "$MLP_DIM" \
        --img_size "$IMAGE_SIZE" \
        --output_dir "$TASK_OUTPUT_ROOT" \
        --data_path "$DATA_PATH" \
        --world_size "$WORLD_SIZE" \
        --logger txt \
        --skip_resolution_evaluation
else
    echo "Unsupported family: ${FAMILY}" >&2
    exit 1
fi

python3 measure_training_time.py "$LOG_FILE" --epoch 0 --epochs "$ESTIMATED_EPOCHS" \
    | tee "${RUN_DIR}/training_time_${ESTIMATED_EPOCHS}_epochs.txt"

python3 measure_training_resources.py "$LOG_FILE" \
    --epochs "$ESTIMATED_EPOCHS" \
    --train-samples "$TRAIN_SAMPLES" \
    --backward-multiplier "$BACKWARD_MULTIPLIER" \
    | tee "${RUN_DIR}/training_resources_${ESTIMATED_EPOCHS}_epochs.txt"

# Keys consumed verbatim by collect_scaling_training_times.py.
{
    echo "family: ${FAMILY}"
    echo "scaling_axis: ${AXIS}"
    echo "config_name: ${CONFIG_NAME}"
    echo "image_size: ${IMAGE_SIZE}"
    echo "batch_size: ${BATCH_SIZE}"
    echo "workers: ${WORKERS}"
    echo "measured_epochs: ${MEASURED_EPOCHS}"
    echo "estimated_epochs: ${ESTIMATED_EPOCHS}"
    echo "log_file: ${LOG_FILE}"
} > "${RUN_DIR}/scaling_config.txt"
