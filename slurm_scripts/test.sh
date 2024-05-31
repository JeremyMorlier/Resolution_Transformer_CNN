#!/bin/bash
#SBATCH --job-name=RegSeg # nom du job
#SBATCH --output=log/RegSeg/%j/logs.out # fichier de sortie (%j = job ID)
#SBATCH --error=log/RegSeg/%j/errors.err # fichier d’erreur (%j = job ID)
#SBATCH --constraint=v100-16g # demander des GPU a 16 Go de RAM
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=4 # reserver 4 taches (ou processus)
#SBATCH --gres=gpu:4 # reserver 4 GPU
#SBATCH --cpus-per-task=10 # reserver 10 CPU par tache (et memoire associee)
#SBATCH --time=20:00:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --qos=qos_gpu-t3 # QoS
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

CUDA_VISIBLE_DEVICES=0 srun torchrun --standalone --nnodes=1 --nproc-per-node=1 test_rank.py --patate 1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 srun torchrun --standalone --nnodes=1 --nproc-per-node=1 test_rank.py --patate 2 &
P2=$!
CUDA_VISIBLE_DEVICES=2 srun torchrun --standalone --nnodes=1 --nproc-per-node=1 test_rank.py --patate 3 &
P3=$!
CUDA_VISIBLE_DEVICES=3 srun torchrun --standalone --nnodes=1 --nproc-per-node=1 test_rank.py --patate 4 &
P4=$!
wait $P1 $P2 $P3 $P4
