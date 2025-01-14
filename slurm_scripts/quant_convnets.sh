#!/bin/bash
#SBATCH --job-name=QCO # nom du job
#SBATCH --output=log/QCO/%j/logs.out # fichier de sortie (%j = job ID)
#SBATCH --error=log/QCO/%j/errors.err # fichier d’erreur (%j = job ID)
#SBATCH --constraint=a100
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=1 #reserver 4 taches (ou processus)
#SBATCH --gres=gpu:1 # reserver 4 GPU
#SBATCH --cpus-per-task=8 # reserver 10 CPU par tache (et memoire associee)
#SBATCH --time=20:00:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --qos=qos_gpu_a100-t3 # QoS
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
#SBATCH --account=sxq@a100 # comptabilite 1100
#SBATCH --signal=USR1@40

module purge # nettoyer les modules herites par defaut
conda deactivate # desactiver les environnements herites par defaut
module load anaconda-py3/2023.09
conda activate $WORK/venvs/venvResolution

set -x # activer l’echo des commandes

export WANDB_DIR=$WORK/wandb/
export WANDB_MODE=offline

srun python3 quant_convnets.py