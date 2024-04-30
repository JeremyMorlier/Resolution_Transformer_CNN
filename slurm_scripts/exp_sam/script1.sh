#!/bin/bash
#SBATCH --job-name=TravailGPU # nom du job
#SBATCH --output=log/%j/logs.out # fichier de sortie (%j = job ID)
#SBATCH --error=log/%j/errors.err # fichier d’erreur (%j = job ID)
#SBATCH --constraint=v100-16g # demander des GPU a 16 Go de RAM
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=1 # reserver 4 taches (ou processus)
#SBATCH --gres=gpu:4 # reserver 4 GPU
#SBATCH --cpus-per-task=10 # reserver 10 CPU par tache (et memoire associee)
#SBATCH --time=100:00:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --qos=qos_gpu-t4 # QoS
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
#SBATCH --account=sxq@v100 # comptabilite V100

module purge # nettoyer les modules herites par defaut
conda deactivate # desactiver les environnements herites par defaut
module load anaconda-py3/2023.09
conda activate ../venvs/venvResolution
set -x # activer l’echo des commandes
export CUDA_VISIBLE_DEVICES=0,1,2,3 
srun torchrun --nproc_per_node=4 distillation_sam.py --optim adamw  --model mobilesam_vit --learning_rate 0.001 --weight_decay 0.0005 --epochs 40 --batch_size 8 --sam_checkpoint $WORK/data/sam_vit_h_4b8939.pth --work_dir $WORK/adamw_lr_1e-3_wd_5e-4_bs_8_epoch_16_ds2 --root_feat $WORK/data/SAM_Features --root_path $WORK/Distillation --dataset_path $DSDIR/SegmentAnything_1B --ade_dataset $SCRATCH/ADE20K/ --val_dirs sa_000021 --train_dirs sa_000022 sa_000024

