# Resolution_Tranformer_CNN
The goal of this repository is to provide code to experiment the importance of resolution on CNN and Transformers in the context of image classification and semantic segmentation

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 train_classification.py --data-path PATH_TO_IMAGENET_DATASET --train-crop-size 176  --val-resize-size 232
```

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 train_classification.py --model resnet50 --batch-size 256 --lr 0.1 \
--lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
--auto-augment ta_wide --epochs 120 --random-erase 0.1 --weight-decay 0.00002 \
--norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
--train-crop-size 176 --model-ema --val-resize-size 232 --val-crop-size 224 --output-dir /nasbrain/j20morli/results/ --data-path PATH_TO_IMAGENET_DATASET

torchrun --standalone --nnodes=1 --nproc-per-node=1 train_classification.py --model resnet50 --batch-size 256 --lr 0.1 \
--lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
--auto-augment ta_wide --epochs 120 --random-erase 0.1 --weight-decay 0.00002 \
--norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
--train-crop-size 112 --model-ema --val-resize-size 152 --val-crop-size 144 --output-dir /nasbrain/j20morli/results/ --data-path PATH_TO_IMAGENET_DATASET

torchrun --standalone --nnodes=1 --nproc-per-node=1 train_classification.py --model resnet50 --batch-size 256 --lr 0.1 \
--lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
--auto-augment ta_wide --epochs 120 --random-erase 0.1 --weight-decay 0.00002 \
--norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
--train-crop-size 128 --model-ema --val-resize-size 168 --val-crop-size 160 --output-dir /nasbrain/j20morli/results/ --data-path PATH_TO_IMAGENET_DATASET

torchrun --standalone --nnodes=1 --nproc-per-node=1 train_classification.py --model resnet50 --batch-size 256 --lr 0.1 \
--lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
--auto-augment ta_wide --epochs 120 --random-erase 0.1 --weight-decay 0.00002 \
--norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
--train-crop-size 176 --model-ema --val-resize-size 232 --val-crop-size 224 --first-conv-resize 76 --channels 3 4 6 3 --output-dir /nasbrain/j20morli/results/ --data-path PATH_TO_IMAGENET_DATASET

```
ViT

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 train_classification.py \
    --model vit_custom --epochs 300 --batch-size 512 --opt adamw --lr 0.003 --wd 0.3\
    --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 30\
    --lr-warmup-decay 0.033 --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra\
    --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 --model-ema \
    --train-crop-size 224 --val-resize-size 232 --val-crop-size 224 \
    --patch_size 16 --num_layers 12 --num_heads 12 --hidden_dim 768 --mlp_dim 3072 --img_size 224 \
    --output-dir /nasbrain/j20morli/results/ --data-path PATH_TO_IMAGENET_DATASET

```

RegSeg
```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 train_semantic.py --model regseg_custom --regseg_name exp48_decoder26 --output-dir /nasbrain/j20morli/results/ \
--dataset cityscapes --data-path /nasbrain/datasets/cityscapes/ --scale-low-size 400 --scale-high-size 1600 --random-crop-size 1024 --augmode randaug_reduced --exclude-classes 14 15 16 \
--epochs 1000 --momentum 0.9  --lr 0.05 -b 8 \
--lr-warmup-epochs 9 --lr-warmup-method linear --lr-warmup-start-factor 0.1 
```

```bash
WANDB_DIR=../wandb WANDB_CACHE_DIR=../cache/ python3  train_semantic.py --output-dir /nasbrain/j20morli/results/ \
--dataset cityscapes --data-path /nasbrain/datasets/ cityscapes/ --exclude-classes 14 15 16 \
--scale-low-size 400 --scale-high-size 1600 --random-crop-size 1024 --augmode randaug_reduced \
--model regseg_custom --regseg_name exp48_decoder26 --regseg_gw 16 --regseg_channels 32 24 64 128 320 \
--epochs 1000 --momentum 0.9  --lr 0.05 -b 8 --lr-warmup-epochs 9 --lr-warmup-method linear --lr-warmup-start-factor 0.1 
```
## Segment Anything Distillation
```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 distillation_sam.py --optim adamw --learning_rate 0.001 --weight_decay 0.0005 --epochs 8 --batch_size 8 --model mobilesam_vit\
--work_dir test --root_path ./ --root_feat /users2/local/j20morli_sam_dataset/SAM_vit_h_features --dataset_path /users2/local/j20morli_sam_dataset/images/ \
--ade_dataset /nasbrain/datasets/ADE20k_full/ --sam_checkpoint /users/local/j20morli/data/sam_vit_h_4b8939.pth \
--val_dirs sa_000022 --train_dirs sa_000022 sa_000024 sa_000070 sa_000135 sa_000137 sa_000138 sa_000259 sa_000477 sa_000977
```

### ADE20K
instructions available: https://github.com/CSAILVision/ADE20K
1. Register and download the ADE20K dataset https://groups.csail.mit.edu/vision/datasets/ADE20K
2. Download the index file http://groups.csail.mit.edu/vision/datasets/ADE20K/toolkit/index_ade20k.pkl

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 distillation_sam.py --optim adamw --learning_rate 0.001 --weight_decay 0.0005 --epochs 8 --batch_size 8 --model mobilesam_vit\
--work_dir test --root_path ./ --root_feat /users2/local/j20morli_sam_dataset/SAM_vit_h_features --dataset_path /users2/local/j20morli_sam_dataset/images/ \
--ade_dataset /nasbrain/datasets/ADE20k_full/ --sam_checkpoint /users/local/j20morli/data/sam_vit_h_4b8939.pth \
--val_dirs sa_000022 --train_dirs sa_000022 sa_000024 sa_000070 sa_000135 sa_000137 sa_000138 sa_000259 sa_000477 sa_000977
```

### OpenCLIP
OpenCLIP training scripts are gathered from https://github.com/mlfoundations/open_clip 

#### test
```bash
torchrun -m --standalone --nnodes=1 --nproc-per-node=1 training.main \
    --imagenet-val /path/to/imagenet/validation \
    --model ViT-B-32-quickgelu \
    --pretrained laion400m_e32
```

```bash
torchrun -m --standalone --nnodes=1 --nproc-per-node=1 training.main --model ViT-B-32-quickgelu --dataset-type slip --dataset yfcc15m --root $DSDIR/YFCC100M/ --metadata $SCRATCH/YFCC100M/yfcc15m.pkl --imagenet-val $DSDIR/imagenet
```

WANDB_DIR=../wandb/ WANDB_CACHE_DIR=../cache