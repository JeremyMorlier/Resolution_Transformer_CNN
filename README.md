# Resolution Tranformer CNN

This repository contains the code for the experiments in the __[Input Resolution Downsizing as a Compression Technique for Vision Deep Learning Systems](https://arxiv.org/abs/2504.03749)__ paper (Accepted at IJCNN 2025).


## Setup

### Using Pip
```
python3 -m venv .venv
source .venv/bin/activate
pip install .
```
### Using UV
```
uv sync
source .venv/bin/activate
```
### Install Datasets
#### Cityscapes
instructions to download Cityscapes dataset
1. Register on  https://www.cityscapes-dataset.com/downloads/
2. install https://github.com/mcordts/cityscapesScripts, `pip install cityscapesscripts`
3. ``csDownload gtFine_trainvaltest.zip && csDownload gtCoarse.zip && csDownload leftImg8bit_trainvaltest.zip``
4. ``unzip gtFine_trainvaltest.zip && unzip gtCoarse.zip && unzip leftImg8bit_trainvaltest.zip``
6. ``CITYSCAPES_DATASET=PATH_TO_DATASET csCreateTrainIdLabelImgs``
5. if needed (to update the timestamps of files) ``find . -type f -exec cat {} \; &> /dev/null``

## Experiments
The experiments are meant to be used on a distributed system (mostly tested on SLURM but should work with torchrun)

### Training
Several examples scripts to train the neural networks used(ResNet50 and ViTs)
#### ResNet50
```bash
python3 train_classification.py --model resnet50 --batch_size 256 --lr 0.1 \
--lr_scheduler cosineannealinglr --lr_warmup_epochs 5 --lr_warmup_method linear \
--auto_augment ta_wide --epochs 120 --random_erase 0.1 --weight_decay 0.00002 \
--norm_weight_decay 0.0 --label_smoothing 0.1 --mixup_alpha 0.2 --cutmix_alpha 1.0 \
--train_crop_size 176 --model_ema --val_resize_size 232 --val_crop_size 224 --output_dir OUTPUT_PATH --data_path PATH_TO_IMAGENET_DATASET
```
```bash
python3 train_classification.py --model resnet50 --batch_size 256 --lr 0.1 \
--lr_scheduler cosineannealinglr --lr_warmup_epochs 5 --lr_warmup_method linear \
--auto_augment ta_wide --epochs 120 --random_erase 0.1 --weight_decay 0.00002 \
--norm_weight_decay 0.0 --label_smoothing 0.1 --mixup_alpha 0.2 --cutmix_alpha 1.0 \
--train_crop_size 112 --model_ema --val_resize_size 152 --val_crop_size 144 --output_dir OUTPUT_PATH --data_path PATH_TO_IMAGENET_DATASET
```
```bash
python3 train_classification.py --model resnet50 --batch_size 256 --lr 0.1 \
--lr_scheduler cosineannealinglr --lr_warmup_epochs 5 --lr_warmup_method linear \
--auto_augment ta_wide --epochs 120 --random_erase 0.1 --weight_decay 0.00002 \
--norm_weight_decay 0.0 --label_smoothing 0.1 --mixup_alpha 0.2 --cutmix_alpha 1.0 \
--train_crop_size 128 --model_ema --val_resize_size 168 --val_crop_size 160 --output_dir OUTPUT_PATH --data_path PATH_TO_IMAGENET_DATASET
```
```bash
python3 train_classification.py --model resnet50 --batch_size 256 --lr 0.1 \
--lr_scheduler cosineannealinglr --lr_warmup_epochs 5 --lr_warmup_method linear \
--auto_augment ta_wide --epochs 120 --random_erase 0.1 --weight_decay 0.00002 \
--norm_weight_decay 0.0 --label_smoothing 0.1 --mixup_alpha 0.2 --cutmix_alpha 1.0 \
--train_crop_size 176 --model_ema --val_resize_size 232 --val_crop_size 224 --first_conv_resize 76 --channels 3 4 6 3 --output_dir OUTPUT_PATH --data_path PATH_TO_IMAGENET_DATASET
```
#### RegSeg
```bash
python3 train_semantic.py --model regseg_custom --regseg_name exp48_decoder26 --output_dir OUTPUT_PATH \
--dataset cityscapes --data_path PATH_TO_CITYSCAPES --scale_low_size 400 --scale_high_size 1600 --random_crop_size 1024 --augmode randaug_reduced --exclude_classes 14 15 16 \
--epochs 1000 --momentum 0.9  --lr 0.05 -b 8 --wd 0.0001 \
--lr_warmup_epochs 9 --lr_warmup_method linear --lr_warmup_start_factor 0.1 
```
#### ViT
```bash
python3 train_classification.py \
--model vit_custom --epochs 300 --batch_size 512 --opt adamw --lr 0.003 --wd 0.3 \
--lr_scheduler cosineannealinglr --lr_warmup_method linear --lr_warmup_epochs 30 \
--lr_warmup_decay 0.033 --label_smoothing 0.11 --mixup_alpha 0.2 --auto_augment ra \
--clip_grad_norm 1 --ra_sampler --cutmix_alpha 1.0 --model_ema \
--train_crop_size 224 --val_resize_size 232 --val_crop_size 224 \
--patch_size 16 --num_layers 12 --num_heads 12 --hidden_dim 768 --mlp_dim 3072 --img_size 224 \
--output_dir OUTPUT_PATH --data_path PATH_TO_IMAGENET_DATASET
```
### Quantization

### Univit Evaluation

