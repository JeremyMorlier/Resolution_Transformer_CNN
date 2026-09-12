#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --job-name=RTCNN_Scaling_TIME
#SBATCH --output=logs/%j/output_%a.out
#SBATCH --error=logs/%j/error_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=Brain_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-35%1

set -euo pipefail

cd "${PROJECT_ROOT:-/SCRATCH/j20morli/Resolution_Transformer_CNN}"
source "${VENV_PATH:-.venv/bin/activate}"

export WANDB_DIR="${WANDB_DIR:-${WORK:-$HOME}/wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"

MEASURED_EPOCHS="${MEASURED_EPOCHS:-1}"
ESTIMATED_EPOCHS="${ESTIMATED_EPOCHS:-120}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-1281167}"
BACKWARD_MULTIPLIER="${BACKWARD_MULTIPLIER:-3.0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DATA_PATH="${DATA_PATH:-/SCRATCH/datasets/imagenet}"
SLURM_JOB_LABEL="${SLURM_ARRAY_JOB_ID:-manual}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
WORLD_SIZE="${SLURM_NTASKS:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/SCRATCH/j20morli/results_resolution/training_time/scaling/$SLURM_JOB_LABEL}"

# Fields:
# family axis name image_size model-specific-values
#
# ResNet values: channels_csv depths_csv
# ViT values: patch_size num_layers num_heads hidden_dim mlp_dim
CONFIGS=(
    "resnet model resnet50_c0.50 224 32,32,64,128,256 3,4,6,3"
    "resnet model resnet50_c0.75 224 48,48,96,192,384 3,4,6,3"
    "resnet model resnet50_c1.00 224 64,64,128,256,512 3,4,6,3"
    "resnet model resnet50_c1.50 224 96,96,192,384,768 3,4,6,3"
    "resnet model resnet50_c2.00 224 128,128,256,512,1024 3,4,6,3"
    "resnet resolution resnet50_c1.00_r112 112 64,64,128,256,512 3,4,6,3"
    "resnet resolution resnet50_c1.00_r144 144 64,64,128,256,512 3,4,6,3"
    "resnet resolution resnet50_c1.00_r176 176 64,64,128,256,512 3,4,6,3"
    "resnet resolution resnet50_c1.00_r224 224 64,64,128,256,512 3,4,6,3"
    "resnet resolution resnet50_c1.00_r256 256 64,64,128,256,512 3,4,6,3"
    "resnet resolution resnet50_c1.00_r288 288 64,64,128,256,512 3,4,6,3"
    "resnet resolution resnet50_c1.00_r320 320 64,64,128,256,512 3,4,6,3"
    "resnet resolution resnet50_c1.00_r384 384 64,64,128,256,512 3,4,6,3"
    "vit model_layers vit_layers_6 224 16 6 12 768 3072"
    "vit model_layers vit_layers_9 224 16 9 12 768 3072"
    "vit model_layers vit_layers_12 224 16 12 12 768 3072"
    "vit model_layers vit_layers_18 224 16 18 12 768 3072"
    "vit model_layers vit_layers_24 224 16 24 12 768 3072"
    "vit model_hidden_dim vit_hidden_384 224 16 12 6 384 3072"
    "vit model_hidden_dim vit_hidden_576 224 16 12 9 576 3072"
    "vit model_hidden_dim vit_hidden_768 224 16 12 12 768 3072"
    "vit model_hidden_dim vit_hidden_1024 224 16 12 16 1024 3072"
    "vit model_hidden_dim vit_hidden_1280 224 16 12 20 1280 3072"
    "vit model_mlp_dim vit_mlp_1536 224 16 12 12 768 1536"
    "vit model_mlp_dim vit_mlp_2304 224 16 12 12 768 2304"
    "vit model_mlp_dim vit_mlp_3072 224 16 12 12 768 3072"
    "vit model_mlp_dim vit_mlp_4096 224 16 12 12 768 4096"
    "vit model_mlp_dim vit_mlp_5120 224 16 12 12 768 5120"
    "vit resolution vit_b_16_r112 112 16 12 12 768 3072"
    "vit resolution vit_b_16_r144 144 16 12 12 768 3072"
    "vit resolution vit_b_16_r176 176 16 12 12 768 3072"
    "vit resolution vit_b_16_r224 224 16 12 12 768 3072"
    "vit resolution vit_b_16_r256 256 16 12 12 768 3072"
    "vit resolution vit_b_16_r288 288 16 12 12 768 3072"
    "vit resolution vit_b_16_r320 320 16 12 12 768 3072"
    "vit resolution vit_b_16_r384 384 16 12 12 768 3072"
)

CONFIG="${CONFIGS[$TASK_ID]}"
read -r FAMILY AXIS CONFIG_NAME IMAGE_SIZE ARG1 ARG2 ARG3 ARG4 ARG5 <<< "$CONFIG"

VAL_RESIZE_SIZE=$((IMAGE_SIZE + 8))
TASK_OUTPUT_ROOT="${OUTPUT_ROOT}/${TASK_ID}_${AXIS}_${CONFIG_NAME}"
mkdir -p "$TASK_OUTPUT_ROOT"

echo "Task ${TASK_ID}: ${FAMILY} ${AXIS} ${CONFIG_NAME}, image size ${IMAGE_SIZE}, batch size ${BATCH_SIZE}"

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

{
    echo "family: ${FAMILY}"
    echo "scaling_axis: ${AXIS}"
    echo "config_name: ${CONFIG_NAME}"
    echo "image_size: ${IMAGE_SIZE}"
    echo "batch_size: ${BATCH_SIZE}"
    echo "measured_epochs: ${MEASURED_EPOCHS}"
    echo "estimated_epochs: ${ESTIMATED_EPOCHS}"
    echo "log_file: ${LOG_FILE}"
} > "${RUN_DIR}/scaling_config.txt"
