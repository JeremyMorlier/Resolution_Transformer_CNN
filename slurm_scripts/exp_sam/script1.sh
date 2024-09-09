#!/bin/bash
#SBATCH --job-name=SAM # nom du job
#SBATCH --output=log/SAM/%j/logs.out # fichier de sortie (%j = job ID)
#SBATCH --error=log/SAM/%j/errors.err # fichier d’erreur (%j = job ID)
#SBATCH --constraint=v100-32g # demander des GPU a 16 Go de RAM
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=1 # reserver 4 taches (ou processus)
#SBATCH --gres=gpu:4 # reserver 4 GPU
#SBATCH --cpus-per-task=10 # reserver 10 CPU par tache (et memoire associee)
#SBATCH --time=100:00:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --qos=qos_gpu-t4 # QoS
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
#SBATCH --account=sxq@v100 # comptabilite V100
#SBATCH --signal=USR1@40 

module purge # nettoyer les modules herites par defaut
conda deactivate # desactiver les environnements herites par defaut
module load anaconda-py3/2023.09
conda activate $WORK/venvs/venvResolution
set -x # activer l’echo des commandes
export CUDA_VISIBLE_DEVICES=0,1,2,3 
export WANDB_DIR=$WORK/wandb/
export WANDB_MODE=offline
srun python3 distillation_sam.py --optim adamw  --model mobilesam_vit --learning_rate 0.001 --weight_decay 0.0005 --epochs 40 --batch_size 8 --sam_checkpoint $WORK/data/sam_vit_h_4b8939.pth --output_dir $WORK/results_sam/ --root_feat $WORK/data/SAM_Features --dataset_path $DSDIR/SegmentAnything_1B --ade_dataset $SCRATCH/ADE20K/ --val_dirs sa_000021 --train_dirs sa_000022 sa_000024