#!/bin/bash
#SBATCH --job-name=TravailGPU # nom du job
#SBATCH --output=TravailGPU%j.out # fichier de sortie (%j = job ID)
#SBATCH --error=TravailGPU%j.err # fichier d’erreur (%j = job ID)
#SBATCH --constraint=v100-16g # demander des GPU a 16 Go de RAM
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=1 # reserver 4 taches (ou processus)
#SBATCH --gres=gpu:4 # reserver 4 GPU
#SBATCH --cpus-per-task=10 # reserver 10 CPU par tache (et memoire associee)
#SBATCH --time=20:00:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --qos=qos_gpu-t4 # QoS
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
#SBATCH --account=sxq@v100 # comptabilite V100

module purge # nettoyer les modules herites par defaut
conda deactivate # desactiver les environnements herites par defaut
module load anaconda-py3/2023.09
conda activate ../venvs/venvDistillation
set -x # activer l’echo des commandes
srun python -u preprocess.py --dataset_path $DSDIR/SegmentAnything_1B --features_path $WORK/data/SAM_Features --device cuda:0 --sam_type vit_h --sam_ckpt $WORK/data/sam_vit_h_4b8939.pth --train_dirs sa_000030 sa_000031 sa_000032 sa_000033 sa_000034 sa_000035 sa_000036 sa_000037 sa_000038