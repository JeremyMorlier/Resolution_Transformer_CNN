#!/bin/bash
#SBATCH --job-name=vit_s_time
#SBATCH --output=slurm-vit-s-time-%A_%a.out
#SBATCH --error=slurm-vit-s-time-%A_%a.err
#SBATCH --array=0-5
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --hint=nomultithread
#SBATCH --account=sxq@a100
#SBATCH --signal=USR1@40

set -euo pipefail
set -x

module purge
conda deactivate || true
module load anaconda-py3/2023.09
conda activate "$WORK/venvs/venvResolution"

export WANDB_DIR="$WORK/wandb"
export WANDB_MODE=offline

# Sequence lengths exclude the class token. With 16x16 patches, an N x N
# patch grid uses an image size of N * 16.
SEQUENCE_LENGTHS=(64 121 144 169 196 225)
IMAGE_SIZES=(128 176 192 208 224 240)

SEQUENCE_LENGTH="${SEQUENCE_LENGTHS[$SLURM_ARRAY_TASK_ID]}"
IMAGE_SIZE="${IMAGE_SIZES[$SLURM_ARRAY_TASK_ID]}"
VAL_RESIZE_SIZE=$((IMAGE_SIZE + 8))

MEASURED_EPOCHS=1
ESTIMATED_EPOCHS="${ESTIMATED_EPOCHS:-300}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$WORK/results_resolution/training_time/vit_s_16/$SLURM_ARRAY_JOB_ID}"
DATA_PATH="${DATA_PATH:-$DSDIR/imagenet}"
RUN_NAME="vit_custom_16_12_6_384_1536_${IMAGE_SIZE}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_FILE="${RUN_DIR}/resolution_CNN_ViT_${RUN_NAME}.log"

echo "Measuring ViT-S/16 with sequence length ${SEQUENCE_LENGTH} and image size ${IMAGE_SIZE}"

srun python3 train_classification.py \
    --model vit_custom \
    --epochs "$MEASURED_EPOCHS" \
    --batch_size 512 \
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
    --patch_size 16 \
    --num_layers 12 \
    --num_heads 6 \
    --hidden_dim 384 \
    --mlp_dim 1536 \
    --img_size "$IMAGE_SIZE" \
    --output_dir "$OUTPUT_ROOT" \
    --data_path "$DATA_PATH" \
    --world_size "$SLURM_NTASKS" \
    --logger txt \
    --skip_resolution_evaluation

python3 measure_training_time.py "$LOG_FILE" --epoch 0 --epochs "$ESTIMATED_EPOCHS" \
    | tee "${RUN_DIR}/training_time_${ESTIMATED_EPOCHS}_epochs.txt"
