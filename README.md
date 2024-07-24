# Resolution_Tranformer_CNN
The goal of this repository is to provide code to experiment the importance of resolution on CNN and Transformers in the context of image classification and semantic segmentation

```bash
python3 train_classification.py --data_path PATH_TO_IMAGENET_DATASET --train_crop_size 176  --val_resize_size 232
```

```bash
python3 train_classification.py --model resnet50 --batch_size 256 --lr 0.1 \
--lr_scheduler cosineannealinglr --lr_warmup_epochs 5 --lr_warmup_method linear \
--auto_augment ta_wide --epochs 120 --random_erase 0.1 --weight_decay 0.00002 \
--norm_weight_decay 0.0 --label_smoothing 0.1 --mixup_alpha 0.2 --cutmix_alpha 1.0 \
--train_crop_size 176 --model_ema --val_resize_size 232 --val_crop_size 224 --output_dir /nasbrain/j20morli/results/ --data_path PATH_TO_IMAGENET_DATASET
```
```bash
python3 train_classification.py --model resnet50 --batch_size 256 --lr 0.1 \
--lr_scheduler cosineannealinglr --lr_warmup_epochs 5 --lr_warmup_method linear \
--auto_augment ta_wide --epochs 120 --random_erase 0.1 --weight_decay 0.00002 \
--norm_weight_decay 0.0 --label_smoothing 0.1 --mixup_alpha 0.2 --cutmix_alpha 1.0 \
--train_crop_size 112 --model_ema --val_resize_size 152 --val_crop_size 144 --output_dir /nasbrain/j20morli/results/ --data_path PATH_TO_IMAGENET_DATASET
```
```bash
python3 train_classification.py --model resnet50 --batch_size 256 --lr 0.1 \
--lr_scheduler cosineannealinglr --lr_warmup_epochs 5 --lr_warmup_method linear \
--auto_augment ta_wide --epochs 120 --random_erase 0.1 --weight_decay 0.00002 \
--norm_weight_decay 0.0 --label_smoothing 0.1 --mixup_alpha 0.2 --cutmix_alpha 1.0 \
--train_crop_size 128 --model_ema --val_resize_size 168 --val_crop_size 160 --output_dir /nasbrain/j20morli/results/ --data_path PATH_TO_IMAGENET_DATASET
```
```bash
python3 train_classification.py --model resnet50 --batch_size 256 --lr 0.1 \
--lr_scheduler cosineannealinglr --lr_warmup_epochs 5 --lr_warmup_method linear \
--auto_augment ta_wide --epochs 120 --random_erase 0.1 --weight_decay 0.00002 \
--norm_weight_decay 0.0 --label_smoothing 0.1 --mixup_alpha 0.2 --cutmix_alpha 1.0 \
--train_crop_size 176 --model_ema --val_resize_size 232 --val_crop_size 224 --first_conv_resize 76 --channels 3 4 6 3 --output_dir /nasbrain/j20morli/results/ --data_path PATH_TO_IMAGENET_DATASET

```
ViT

```bash
python3 train_classification.py \
    --model vit_custom --epochs 300 --batch_size 512 --opt adamw --lr 0.003 --wd 0.3\
    --lr_scheduler cosineannealinglr --lr_warmup_method linear --lr_warmup_epochs 30\
    --lr_warmup_decay 0.033 --label_smoothing 0.11 --mixup_alpha 0.2 --auto_augment ra\
    --clip_grad_norm 1 --ra_sampler --cutmix_alpha 1.0 --model_ema \
    --train_crop_size 224 --val_resize_size 232 --val_crop_size 224 \
    --patch_size 16 --num_layers 12 --num_heads 12 --hidden_dim 768 --mlp_dim 3072 --img_size 224 \
    --output_dir /nasbrain/j20morli/results/ --data_path PATH_TO_IMAGENET_DATASET

```

RegSeg
```bash
python3 train_semantic.py --model regseg_custom --regseg_name exp48_decoder26 --output_dir /nasbrain/j20morli/results/ \
--dataset cityscapes --data_path /nasbrain/datasets/cityscapes/ --scale_low_size 400 --scale_high_size 1600 --random_crop_size 1024 --augmode randaug_reduced --exclude_classes 14 15 16 \
--epochs 1000 --momentum 0.9  --lr 0.05 -b 8 --wd 0.0001\
--lr_warmup_epochs 9 --lr_warmup_method linear --lr_warmup_start_factor 0.1 
```

```bash
WANDB_DIR=../wandb WANDB_CACHE_DIR=../cache/ python3  train_semantic.py --output_dir /nasbrain/j20morli/results/ \
--dataset cityscapes --data_path /nasbrain/datasets/ cityscapes/ --exclude_classes 14 15 16 \
--scale_low_size 400 --scale_high_size 1600 --random_crop_size 1024 --augmode randaug_reduced \
--model regseg_custom --regseg_name exp48_decoder26 --regseg_gw 16 --regseg_channels 32 24 64 128 320 \
--epochs 1000 --momentum 0.9  --lr 0.05 -b 8 --wd 0.0001 --lr_warmup_epochs 9 --lr_warmup_method linear --lr_warmup_start_factor 0.1 
```
## Segment Anything Distillation
```bash
python3 distillation_sam.py --optim adamw --learning_rate 0.001 --weight_decay 0.0005 --epochs 8 --batch_size 8 --model mobilesam_vit\
--work_dir test --root_path ./ --root_feat /users2/local/j20morli_sam_dataset/SAM_vit_h_features --dataset_path /users2/local/j20morli_sam_dataset/images/ \
--ade_dataset /nasbrain/datasets/ADE20k_full/ --sam_checkpoint /users/local/j20morli/data/sam_vit_h_4b8939.pth \
--val_dirs sa_000022 --train_dirs sa_000022 sa_000024 sa_000070 sa_000135 sa_000137 sa_000138 sa_000259 sa_000477 sa_000977
```
```bash
python3 distillation_sam.py --optim adamw --learning_rate 0.001 --weight_decay 0.0005 --epochs 8 --batch_size 8 --model mobilesam_vit\
--work_dir test --root_path ./ --root_feat /users2/local/j20morli_sam_dataset/SAM_vit_h_features --dataset_path /users2/local/j20morli_sam_dataset/images/ \
--ade_dataset /nasbrain/datasets/ADE20k_full/ --sam_checkpoint /users/local/j20morli/data/sam_vit_h_4b8939.pth \
--val_dirs sa_000022 --train_dirs sa_000022 sa_000024 sa_000070 sa_000135 sa_000137 sa_000138 sa_000259 sa_000477 sa_000977
```

WANDB_DIR=../wandb/ WANDB_CACHE_DIR=../cache


## Slurm Helper
Use slurm_launcher.py as an helper to launch slurm scripts, useful if high number of runs necessary of small modifications
```bash
python3 slurm_launcher.py --job_name vitB_lengthscaling --output log/default/%j/logs.out --error log/default/%j/errors.err --constraint a100 --nodes 1 --ntasks 8 --gres gpu:4 --cpus_per_task 4 --qos qos_gpu-t3 --hint nomultithread --time 20:00:00 --account sxq@a100 --script train_classification \
--model vit_custom --epochs 300 --batch_size 512 --opt adamw --lr 0.003 --wd 0.3 \
--lr_scheduler cosineannealinglr --lr_warmup_method linear --lr_warmup_epochs 30 \
--lr_warmup_decay 0.033 --label_smoothing 0.11 --mixup_alpha 0.2 --auto_augment ra \
--clip_grad_norm 1 --ra_sampler --cutmix_alpha 1.0 --model_ema \
--train_crop_size 224 --val_resize_size 232 --val_crop_size 224 \
--patch_size 16 --num_layers 12 --num_heads 12 --hidden_dim 768 --mlp_dim 3072 --img_size 224 \
--output_dir $WORK/results_resolution/ --data_path $DSDIR/imagenet
 ```


## Datasets scripts
### ADE20K
instructions available: https://github.com/CSAILVision/ADE20K
1. Register and download the ADE20K dataset https://groups.csail.mit.edu/vision/datasets/ADE20K
2. Download the index file http://groups.csail.mit.edu/vision/datasets/ADE20K/toolkit/index_ade20k.pkl


### Cityscapes
instructions to download Cityscapes dataset
1. Register on  https://www.cityscapes-dataset.com/downloads/
2. install https://github.com/mcordts/cityscapesScripts, `pip install cityscapesscripts`
3. csDownload gtFine_trainvaltest.zip && csDownload gtCoarse.zip && csDownload
4. unzip gtFine_trainvaltest.zip && unzip gtCoarse.zip && unzip
6. CITYSCAPES_DATASET=PATH_TO_DATASET csCreateTrainIdLabelImgs
5. if needed find . -type f -exec cat {} \; &> /dev/null
