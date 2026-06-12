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
    --model vit_custom --epochs 300 --batch_size 512 --opt adamw --lr 0.003 --wd 0.3\
    --lr_scheduler cosineannealinglr --lr_warmup_method linear --lr_warmup_epochs 30\
    --lr_warmup_decay 0.033 --label_smoothing 0.11 --mixup_alpha 0.2 --auto_augment ra\
    --clip_grad_norm 1 --ra_sampler --cutmix_alpha 1.0 --model_ema \
    --train_crop_size 128 --val_resize_size 136 --val_crop_size 128 \
    --patch_size 16 --num_layers 12 --num_heads 12 --hidden_dim 768 --mlp_dim 3072 --img_size 128 \
    --output_dir $WORK/results_resolution/ --data_path $DSDIR/imagenet