#!/bin/bash
#SBATCH --job-name=VITB # nom du job
#SBATCH --output=log/VITB/%j/logs.out # fichier de sortie (%j = job ID)
#SBATCH --error=log/VITB/%j/errors.err # fichier d’erreur (%j = job ID)
#SBATCH --constraint=a100
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=8 #reserver 4 taches (ou processus)
#SBATCH --gres=gpu:8 # reserver 4 GPU
#SBATCH --cpus-per-task=8 # reserver 10 CPU par tache (et memoire associee)
#SBATCH --time=20:00:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --qos=qos_gpu-t3 # QoS
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
#SBATCH --account=sxq@a100 # comptabilite V100

module purge # nettoyer les modules herites par defaut
conda deactivate # desactiver les environnements herites par defaut
module load anaconda-py3/2023.09
conda activate $WORK/venvs/venvResolution
set -x # activer l’echo des commandes

export WANDB_DIR=$WORK/wandb/
export WANDB_MODE=offline

srun python3 train_classification.py \
    --model vit_custom --epochs 300 --batch-size 512 --opt adamw --lr 0.003 --wd 0.3\
    --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 30\
    --lr-warmup-decay 0.033 --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra\
    --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 --model-ema \
    --train-crop-size 112 --val-resize-size 120 --val-crop-size 112 \
    --patch_size 16 --num_layers 12 --num_heads 12 --hidden_dim 768 --mlp_dim 3072 --img_size 112 \
    --output-dir $WORK/results_resolution/ --data-path $DSDIR/imagenet