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
torchrun --standalone --nnodes=1 --nproc-per-node=1 train_semantic.py --model regseg_custom --regseg_name exp48_decoder26 --output-dir /nasbrain/j20morli/results/ \
--dataset cityscapes --data-path /nasbrain/datasets/cityscapes/ --scale-low-size 400 --scale-high-size 1600 --random-crop-size 1024 --augmode randaug_reduced --exclude-classes 14 15 16 \
--epochs 1000 --momentum 0.9  --lr 0.05 -b 8 \
--lr-warmup-epochs 9 --lr-warmup-method linear --lr-warmup-start-factor 0.1 
```
python3 train_semantic.py --model regseg_custom --regseg_name exp48_decoder26 --output_dir /nasbrain/j20morli/results/ \
--dataset cityscapes --data_path /nasbrain/datasets/cityscapes/ --scale_low_size 400 --scale_high_size 1600 --random_crop_size 1024 --augmode randaug_reduced --exclude_classes 14 15 16 \
--epochs 1000 --momentum 0.9  --lr 0.05 -b 8 --wd 0.0001\
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

### Scaling resource estimates
To compare model scaling against input-resolution scaling for ResNet and ViT, run:

```bash
python3 scaling_resources.py --output scaling_resources.csv --plot scaling_resources.png
```

The CSV contains forward FLOPs per sample, estimated training FLOPs per sample and per full run, activation memory per sample, and estimated peak training memory for each scaling point. The PNG plots estimated training memory against training FLOPs. ResNet model scaling changes the channel width. ViT model scaling is split into separate layer-count, hidden-dimension, and MLP-dimension sweeps. By default, training FLOPs are estimated as `3x` forward FLOPs, memory uses an AdamW-style parameter/gradient/optimizer-state estimate, and the batch size is `256`.

To measure one training epoch for the same scaling grid on SLURM, submit:

```bash
sbatch slurm_scripts/measure_scaling_training_time.sh
```

Each array task writes per-run timing/resource summaries under `OUTPUT_ROOT`. After the array finishes, collect the measured epoch times with:

```bash
python3 collect_scaling_training_times.py /SCRATCH/j20morli/results_resolution/training_time/scaling/JOB_ID
```

### Measured training cost (time and memory)

`scaling_resources.py` estimates training cost analytically. `benchmark_training_step.py`
measures it: it runs the real training step (forward, backward, optimizer) on synthetic batches
and reports step time, a forward/backward/optimizer breakdown, and a decomposition of training
memory into parameters, gradients, optimizer state and activations.

Synthetic batches are deliberate. `train_one_epoch` calls `.item()` three times per step and runs
JPEG decode plus RandAugment on the CPU, so a real ImageNet epoch at 112px is dataloader bound and
its wall time understates the compute saved by lowering the resolution. `--dataloader-mode`
quantifies that gap at a few points instead of inheriting it everywhere.

Both scripts enumerate their grid from `scaling_specs.py`, so their CSVs join on
`(family, model_name, resolution)`. **Pass the same grid flags to both** (`--families`,
`--resolutions`, `--include-resnet-depth`, ...): `--config-index` is an index into that
enumeration, and the join needs the two tables to cover the same points.

```bash
# list the grid (41 points for classification with the depth axis, 51 including RegSeg)
python3 benchmark_training_step.py --families resnet vit --include-resnet-depth --list-configs

# one point, locally, on CPU
python3 benchmark_training_step.py --config-index 13 --device cpu --batch-size 2 \
    --warmup-steps 1 --measure-steps 3 --output training_cost.csv

# the full grid on SLURM: A100, one array task per precision
sbatch slurm_scripts/benchmark_training_cost_gpu.sh
ISOLATE=1 sbatch slurm_scripts/benchmark_training_cost_gpu.sh   # one process per point

# CPU baseline (edge-training proxy) and RegSeg/Cityscapes dense prediction
sbatch slurm_scripts/benchmark_training_cost_cpu.sh
sbatch slurm_scripts/benchmark_training_cost_segmentation.sh
```

Measurement settings that matter and are recorded in every row:

- `--tf32` sets **both** `cuda.matmul.allow_tf32` and `cudnn.allow_tf32`. Torch defaults these
  differently (False and True), which speeds up convolutions but not matmuls and therefore biases
  ViT against ResNet. Run the whole grid with one setting.
- `--warmup-steps` (default 10) absorbs cuDNN autotune, lazy allocator growth and the lazy
  creation of the AdamW state. The first steps at a new input shape can be 10x the steady state.
- Batch size is fixed across the grid on purpose: points measured at different batch sizes are not
  comparable. The default of 64 keeps `vit_hidden_1280` and `vit_b_16@384` inside a 40 GB A100.
- `--shuffle-configs` and `--settle-seconds` spread thermal drift across the grid instead of
  letting it correlate with model size; `sm_clock_mhz_*` and `gpu_temp_c_*` columns make throttled
  rows detectable afterwards.
- Memory columns are empty on CPU (no allocator peak counter); those rows carry timing plus
  `cpu_max_rss_bytes` as a low-confidence upper bound.

Validate the synthetic numbers against real epochs, then join everything and draw the figures:

```bash
sbatch slurm_scripts/validate_epoch_time.sh
python3 collect_scaling_training_times.py /SCRATCH/.../training_time/validation/JOB_ID

python3 collect_training_cost.py /SCRATCH/.../training_cost/gpu/JOB_ID \
    /SCRATCH/.../training_cost/cpu/JOB_ID \
    --analytic scaling_resources.csv \
    --epoch-times /SCRATCH/.../training_time/validation/JOB_ID/scaling_training_times.csv \
    --output results/training_cost_joined.csv --figure-dir results/figures
```

`collect_training_cost.py` writes five figures: measured step time against analytic FLOPs (F1),
the training-memory decomposition along each scaling axis (F2), measured against analytic
activation memory including the fitted value of the analytic `--activation-training-multiplier`
(F3), GPU against CPU normalised step time (F4), and predicted against measured epoch time with
the dataloader-bound points above the diagonal (F5).
