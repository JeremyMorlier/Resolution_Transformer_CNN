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

CUDA_VISIBLE_DEVICES=0 srun torchrun --standalone --nnodes=1 --nproc-per-node=1 train_semantic.py --model regseg_custom --regseg_name exp48_decoder26 --output-dir $WORK/results_RegSeg/ --dataset cityscapes --data-path $SCRATCH/cityscapes/ --scale-low-size 400 --scale-high-size 1600 --random-crop-size 1024 --augmode randaug_reduced --exclude-classes 14 15 16 --epochs 1000 --momentum 0.9  --lr 0.05 -b 8 --lr-warmup-epochs 9 --lr-warmup-method linear --lr-warmup-start-factor 0.1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 srun torchrun --standalone --nnodes=1 --nproc-per-node=1 train_semantic.py --model regseg_custom --regseg_name exp48_decoder26 --output-dir $WORK/results_RegSeg/ --dataset cityscapes --data-path $SCRATCH/cityscapes/ --scale-low-size 400 --scale-high-size 1600 --random-crop-size 896 --augmode randaug_reduced --exclude-classes 14 15 16 --epochs 1000 --momentum 0.9  --lr 0.05 -b 8 --lr-warmup-epochs 9 --lr-warmup-method linear --lr-warmup-start-factor 0.1 &
P2=$!
CUDA_VISIBLE_DEVICES=2 srun torchrun --standalone --nnodes=1 --nproc-per-node=1 train_semantic.py --model regseg_custom --regseg_name exp48_decoder26 --output-dir $WORK/results_RegSeg/ --dataset cityscapes --data-path $SCRATCH/cityscapes/ --scale-low-size 400 --scale-high-size 1600 --random-crop-size 512 --augmode randaug_reduced --exclude-classes 14 15 16 --epochs 1000 --momentum 0.9  --lr 0.05 -b 8 --lr-warmup-epochs 9 --lr-warmup-method linear --lr-warmup-start-factor 0.1 &
P3=$!
CUDA_VISIBLE_DEVICES=3 srun torchrun --standalone --nnodes=1 --nproc-per-node=1 train_semantic.py --model regseg_custom --regseg_name exp48_decoder26 --output-dir $WORK/results_RegSeg/ --dataset cityscapes --data-path $SCRATCH/cityscapes/ --scale-low-size 400 --scale-high-size 1600 --random-crop-size 256 --augmode randaug_reduced --exclude-classes 14 15 16 --epochs 1000 --momentum 0.9  --lr 0.05 -b 8 --lr-warmup-epochs 9 --lr-warmup-method linear --lr-warmup-start-factor 0.1 &
P4=$!
wait $P1 $P2 $P3 $P4

# CUDA_VISIBLE_DEVICES=0 srun torchrun --standalone --nnodes=1 --nproc-per-node=1 train_semantic.py --model regseg_custom --regseg_name exp48_decoder26 --output-dir $WORK/results_RegSeg/ --dataset cityscapes --data-path $SCRATCH/cityscapes/ --scale-low-size 400 --scale-high-size 1600 --random-crop-size 1024 --augmode randaug_reduced --exclude-classes 14 15 16 --epochs 1000 --momentum 0.9  --lr 0.05 -b 8 --lr-warmup-epochs 9 --lr-warmup-method linear --lr-warmup-start-factor 0.1 &
# P5=$!
# CUDA_VISIBLE_DEVICES=1 srun torchrun --standalone --nnodes=1 --nproc-per-node=1 train_semantic.py --model regseg_custom --regseg_name exp48_decoder26 --output-dir $WORK/results_RegSeg/ --dataset cityscapes --data-path $SCRATCH/cityscapes/ --scale-low-size 400 --scale-high-size 1600 --random-crop-size 1024 --augmode randaug_reduced --exclude-classes 14 15 16 --epochs 1000 --momentum 0.9  --lr 0.05 -b 8 --lr-warmup-epochs 9 --lr-warmup-method linear --lr-warmup-start-factor 0.1 &
# P6=$!
# CUDA_VISIBLE_DEVICES=2 srun torchrun --standalone --nnodes=1 --nproc-per-node=1 train_semantic.py --model regseg_custom --regseg_name exp48_decoder26 --output-dir $WORK/results_RegSeg/ --dataset cityscapes --data-path $SCRATCH/cityscapes/ --scale-low-size 400 --scale-high-size 1600 --random-crop-size 1024 --augmode randaug_reduced --exclude-classes 14 15 16 --epochs 1000 --momentum 0.9  --lr 0.05 -b 8 --lr-warmup-epochs 9 --lr-warmup-method linear --lr-warmup-start-factor 0.1 &
# P7=$!
# CUDA_VISIBLE_DEVICES=3 srun torchrun --standalone --nnodes=1 --nproc-per-node=1 train_semantic.py --model regseg_custom --regseg_name exp48_decoder26 --output-dir $WORK/results_RegSeg/ --dataset cityscapes --data-path $SCRATCH/cityscapes/ --scale-low-size 400 --scale-high-size 1600 --random-crop-size 1024 --augmode randaug_reduced --exclude-classes 14 15 16 --epochs 1000 --momentum 0.9  --lr 0.05 -b 8 --lr-warmup-epochs 9 --lr-warmup-method linear --lr-warmup-start-factor 0.1 &
# P8=$!
# wait $P5 $P6 $P7 $P8
