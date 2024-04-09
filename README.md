# Resolution_Tranformer_CNN
The goal of this repository is to provide code to experiment the importance of resolution on CNN and Transformers in the context of image classification and semantic segmentation

```bash
python3 train_resnet50.py --data-path PATH_TO_IMAGENET_DATASET --train-crop-size 176  --val-resize-size 232
```


```bash
python3 train_semantic.py --data-path /nasbrain/datasets/cityscapes/ --lr 0.05 --dataset cityscapes -b 8 --model regseg_custom --epochs 500 --momentum 0.9 --exclude-classes 14 15 16 \
--lr-warmup-epochs 9 --lr-warmup-method linear --lr-warmup-start-factor 0.1
```