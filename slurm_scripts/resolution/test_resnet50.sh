#!/bin/bash
#SBATCH --job-name=ResNet # nom du job
#SBATCH --output=log/ResNet/%j/logs.out # fichier de sortie (%j = job ID)
#SBATCH --error=log/ResNet/%j/errors.err # fichier d’erreur (%j = job ID)
#SBATCH --constraint=v100-16g # demander des GPU a 16 Go de RAM
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=4 # reserver 4 taches (ou processus)
#SBATCH --gres=gpu:4 # reserver 4 GPU
#SBATCH --cpus-per-task=10 # reserver 10 CPU par tache (et memoire associee)
#SBATCH --time=100:00:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --qos=qos_gpu-t4 # QoS
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
#SBATCH --account=sxq@v100 # comptabilite V100

module purge # nettoyer les modules herites par defaut
conda deactivate # desactiver les environnements herites par defaut
module load anaconda-py3/2023.09
conda activate $WORK/venvs/venvResolution
set -x # activer l’echo des commandes
export CUDA_VISIBLE_DEVICES=0,1,2,3 
export WANDB_DIR=$WORK/wandb/
export WANDB_MODE=offline

CUDA_VISIBLE_DEVICES=0 srun torchrun --standalone --nnodes=1 --nproc-per-node=1 train_classification.py --model resnet50_resize --batch-size 256 --lr 0.1 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --auto-augment ta_wide --epochs 120 --random-erase 0.1 --weight-decay 0.00002 --norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 --train-crop-size 176 --model-ema --val-resize-size 232 --val-crop-size 224 --channels  16 32 32 512 --output-dir $WORK/results_resolution/ --data-path $DSDIR/imagenet &
P1=$!
CUDA_VISIBLE_DEVICES=1 srun torchrun --standalone --nnodes=1 --nproc-per-node=1 train_classification.py --model resnet50_resize --batch-size 256 --lr 0.1 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --auto-augment ta_wide --epochs 120 --random-erase 0.1 --weight-decay 0.00002 --norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 --train-crop-size 176 --model-ema --val-resize-size 232 --val-crop-size 224 --channels  16 64 128 256 --output-dir $WORK/results_resolution/  --data-path $DSDIR/imagenet &
P2=$!
CUDA_VISIBLE_DEVICES=2 srun torchrun --standalone --nnodes=1 --nproc-per-node=1 train_classification.py --model resnet50_resize --batch-size 256 --lr 0.1 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --auto-augment ta_wide --epochs 120 --random-erase 0.1 --weight-decay 0.00002 --norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 --train-crop-size 176 --model-ema --val-resize-size 232 --val-crop-size 224 --channels  32 64 256 256 --output-dir $WORK/results_resolution/  --data-path $DSDIR/imagenet &
P3=$!
CUDA_VISIBLE_DEVICES=3 srun torchrun --standalone --nnodes=1 --nproc-per-node=1 train_classification.py --model resnet50_resize --batch-size 256 --lr 0.1 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --auto-augment ta_wide --epochs 120 --random-erase 0.1 --weight-decay 0.00002 --norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 --train-crop-size 224 --model-ema --val-resize-size 232 --val-crop-size 224 --first-conv-resize 64 --output-dir $WORK/results_resolution/  --data-path $DSDIR/imagenet &
P4=$!
wait $P1 $P2 $P3 $P4

CUDA_VISIBLE_DEVICES=0 srun torchrun --standalone --nnodes=1 --nproc-per-node=1 train_classification.py --model resnet50_resize --batch-size 256 --lr 0.1 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --auto-augment ta_wide --epochs 120 --random-erase 0.1 --weight-decay 0.00002 --norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 --train-crop-size 224 --model-ema --val-resize-size 232 --val-crop-size 224 --first-conv-resize 80 --output-dir $WORK/results_resolution/ --data-path $DSDIR/imagenet &
P5=$!
CUDA_VISIBLE_DEVICES=1 srun torchrun --standalone --nnodes=1 --nproc-per-node=1 train_classification.py --model resnet50_resize --batch-size 256 --lr 0.1 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --auto-augment ta_wide --epochs 120 --random-erase 0.1 --weight-decay 0.00002 --norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 --train-crop-size 224 --model-ema --val-resize-size 232 --val-crop-size 224 --first-conv-resize 88 --output-dir $WORK/results_resolution/  --data-path $DSDIR/imagenet &
P6=$!
CUDA_VISIBLE_DEVICES=2 srun torchrun --standalone --nnodes=1 --nproc-per-node=1 train_classification.py --model resnet50_resize --batch-size 256 --lr 0.1 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --auto-augment ta_wide --epochs 120 --random-erase 0.1 --weight-decay 0.00002 --norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 --train-crop-size 224 --model-ema --val-resize-size 232 --val-crop-size 224 --first-conv-resize 56 --output-dir $WORK/results_resolution/  --data-path $DSDIR/imagenet &
P7=$!
wait $P5 $P6 $P7