# Resolution_Tranformer_CNN
The goal of this repository is to provide code to experiment the importance of resolution on CNN and Transformers in the context of image classification and semantic segmentation

```bash
python3 train_resnet50.py --data-path PATH_TO_IMAGENET_DATASET --train-crop-size 176  --val-resize-size 232
```

```bash
python train_resnet50.py --model resnet50 --batch-size 256 --lr 0.1 \
--lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
--auto-augment ta_wide --epochs 120 --random-erase 0.1 --weight-decay 0.00002 \
--norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
--train-crop-size 176 --model-ema --val-resize-size 232 --val-crop-size 224 --data-path PATH_TO_IMAGENET_DATASET

python train_resnet50.py --model resnet50 --batch-size 256 --lr 0.1 \
--lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
--auto-augment ta_wide --epochs 120 --random-erase 0.1 --weight-decay 0.00002 \
--norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
--train-crop-size 112 --model-ema --val-resize-size 152 --val-crop-size 144 --data-path PATH_TO_IMAGENET_DATASET

python train_resnet50.py --model resnet50 --batch-size 256 --lr 0.1 \
--lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
--auto-augment ta_wide --epochs 120 --random-erase 0.1 --weight-decay 0.00002 \
--norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
--train-crop-size 128 --model-ema --val-resize-size 168 --val-crop-size 160 --data-path PATH_TO_IMAGENET_DATASET

python train_resnet50.py --model resnet50 --batch-size 256 --lr 0.1 \
--lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
--auto-augment ta_wide --epochs 120 --random-erase 0.1 --weight-decay 0.00002 \
--norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
--train-crop-size 176 --model-ema --val-resize-size 232 --val-crop-size 224 --first-conv-resize 76 --channels 3 4 6 3 --data-path PATH_TO_IMAGENET_DATASET

```
ViT

```bash
python train_resnet50.py \
    --model vit_custom --epochs 300 --batch-size 512 --opt adamw --lr 0.003 --wd 0.3\
    --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 30\
    --lr-warmup-decay 0.033 --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra\
    --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 --model-ema
```

RegSeg
```bash
python3 train_semantic.py --data-path /nasbrain/datasets/cityscapes/ --lr 0.05 --dataset cityscapes -b 8 --model regseg_custom --epochs 1000 --momentum 0.9 --exclude-classes 14 15 16 \
--lr-warmup-epochs 9 --lr-warmup-method linear --lr-warmup-start-factor 0.1 --scale-low-size 400 --scale-high-size 1600 --random-crop-size 1024 --augmode randaug_reduced --regseg_name exp48_decoder26
```