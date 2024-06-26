import datetime
import os, stat
import time
import warnings

import torch
import torch.utils.data
import torchvision
from torch import nn
from torch.optim.lr_scheduler import PolynomialLR
from torchvision.transforms import functional as F, InterpolationMode

from torchvision_references.references.common import create_dir

from torchvision_references.models import get_model
import torchvision_references.datasets as datasets
import torchvision_references.references.segmentation.presets as presets
import torchvision_references.references.segmentation.RegSeg.presets as RS_presets
import torchvision_references.references.segmentation.utils as utils
from torchvision_references.references.segmentation.coco_utils import get_coco

import wandb

def get_dataset(args, is_train):
    def sbd(*args, **kwargs):
        kwargs.pop("use_v2")
        return datasets.SBDataset(*args, mode="segmentation", **kwargs)

    def voc(*args, **kwargs):
        kwargs.pop("use_v2")
        return datasets.VOCSegmentation(*args, **kwargs)

    def cityscapes(*args, **kwargs) :
        kwargs.pop("use_v2")
        return datasets.Cityscapes(*args, **kwargs)
    
    paths = {
        "voc": (args.data_path, voc, 21),
        "voc_aug": (args.data_path, sbd, 21),
        "coco": (args.data_path, get_coco, 21),
        "cityscapes": (args.data_path, cityscapes, 19)
    }
    p, ds_fn, num_classes = paths[args.dataset]

    image_set = "train" if is_train else "val"
    if args.dataset == "cityscapes" :
        ds = ds_fn(p, transforms=get_transform(is_train, args), split=image_set, mode="fine", target_type='semantic', class_uniform_pct=0.5, use_v2=args.use_v2)
    else : 
        ds = ds_fn(p, image_set=image_set, transforms=get_transform(is_train, args), use_v2=args.use_v2)
    return ds, num_classes

def get_transform(is_train, args):
    if args.dataset == "cityscapes" :
        if is_train:
            return RS_presets.build_train_transform2(args.scale_low_size, args.scale_high_size, args.random_crop_size, args.augmode, ignore_value=255)
        else :
            return RS_presets.build_val_transform(args.val_input_size, args.val_label_size)
    if is_train:
        return presets.SegmentationPresetTrain(base_size=520, crop_size=480, backend=args.backend, use_v2=args.use_v2)
    elif args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()

        def preprocessing(img, target):
            img = trans(img)
            size = F.get_dimensions(img)[1:]
            target = F.resize(target, size, interpolation=InterpolationMode.NEAREST)
            return img, F.pil_to_tensor(target)

        return preprocessing
    else:
        return presets.SegmentationPresetEval(base_size=520, backend=args.backend, use_v2=args.use_v2)

def get_param_model(args, num_classes) :
    if args.model == "resnet50_resize" :
        model = get_model(args.model, weights=args.weights, num_classes=num_classes, first_conv_resize=args.first_conv_resize, channels=args.channels, depths=args.depths)
    elif args.model == "vit_custom" :
        model = get_model(args.model, weights=args.weights, num_classes=num_classes, patch_size=args.patch_size, num_layers=args.num_layers, num_heads=args.num_heads, hidden_dim=args.hidden_dim, mlp_dim=args.mlp_dim, image_size=args.img_size)
    elif args.model == "regseg_custom" :
        channels = [32, 48, 128, 256, 320] if args.regseg_channels == None else args.regseg_channels
        gw = 16 if args.regseg_gw == 0 else args.regseg_gw
        model = get_model(args.model, weights=args.weights, weights_backbone=args.weights_backbone, num_classes=num_classes, aux_loss=args.aux_loss, regseg_name=args.regseg_name, channels=channels, gw=gw, first_conv_resize=args.first_conv_resize)
    else :
        model = get_model(args.model, weights=args.weights, num_classes=num_classes)
    
    return model

def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses["out"]

    return losses["out"] + 0.5 * losses["aux"]


def evaluate(model, data_loader, device, num_classes, exclude_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes, exclude_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output["out"]

            confmat.update(target.flatten(), output.argmax(1).flatten())
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            num_processed_samples += image.shape[0]

        confmat.reduce_from_all_processes()

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    return confmat

# Evaluate the trained model at different resolution
def resolution_evaluate(model_state_dict, device, num_classes, args) :

    val_resize_resolutions = [128, 256, 384, 512, 640, 768, 896, 1024, 1280, 1536]

    all_results = []

    # In evaluation, set training architecture modifications to 0
    args.first_conv_resize = 0
    model = get_param_model(args, num_classes=num_classes)
    model.load_state_dict(torch.load(model_state_dict)["model"])
    model.to(device)

    for val_resize_resolution in val_resize_resolutions :
        print("Dataset loading :", val_resize_resolution)

        # Load Dataset with desired validation resolution
        args.val_input_size, args.val_label_size = val_resize_resolution, val_resize_resolution
        dataset_test, _ = get_dataset(args, is_train=False)
        if args.distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
        else:
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
        )

        print(f"Dataset loaded")
        image_size = dataset_test[0][0].size()

        # Evaluate the model at specific resolution
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes, exclude_classes=args.exclude_classes)
        confmat.compute()
        results = [image_size[1], image_size[1], confmat.acc_global.item(), confmat.meanIU,confmat.mIOU_reduced]

        print(results)
        all_results.append(results)

    save_tensor = torch.tensor(all_results)
    torch.save(save_tensor, os.path.join(args.output_dir, "resolution.pth"))
    os.chmod(os.path.join(args.output_dir, "resolution.pth"), stat.S_IRWXU | stat.S_IRWXO)

    return all_results

def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
    # TODO : better logging of LR and loss values
        #wandb.log({"lr": optimizer.param_groups[0]["lr"], "train loss":loss.item()})


def main(args):
    if args.backend.lower() != "pil" and not args.use_v2:
        # TODO: Support tensor backend in V1?
        raise ValueError("Use --use-v2 if you want to use the tv_tensor or tensor backend.")
    if args.use_v2 and args.dataset != "coco":
        raise ValueError("v2 is only support supported for coco dataset for now.")

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    dataset, num_classes = get_dataset(args, is_train=True)
    dataset_test, _ = get_dataset(args, is_train=False)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        collate_fn=utils.collate_fn,
        drop_last=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    model =  get_param_model(args, num_classes)
    model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.model == "regseg_custom" :
        params_to_optimize = [
            {"params": [p for p in model_without_ddp.parameters() if p.requires_grad]}
        ]
    else :
        params_to_optimize = [
            {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
            {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
        ]
        if args.aux_loss:
            params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
            params_to_optimize.append({"params": params, "lr": args.lr * 10})

    optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    iters_per_epoch = len(data_loader)
    main_lr_scheduler = PolynomialLR(
        optimizer, total_iters=iters_per_epoch * (args.epochs - args.lr_warmup_epochs), power=args.power
    )

    if args.lr_warmup_epochs > 0:
        warmup_iters = iters_per_epoch * args.lr_warmup_epochs
        args.lr_warmup_method = args.lr_warmup_method.lower()
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_start_factor, end_factor=args.lr, total_iters=warmup_iters
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_iters]
        )
    else:
        lr_scheduler = main_lr_scheduler

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model_without_ddp.load_state_dict(checkpoint["model"], strict=not args.test_only)
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if args.amp:
                scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
        print(confmat)
        return

    best_mrIoU = 0.0
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq, scaler)
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes, exclude_classes=args.exclude_classes)
        confmat.compute()
        wandb.log({"reduced_iu": confmat.reduced_iu, "mIOU_reduced": confmat.mIOU_reduced})
        print(confmat)
        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
            "acc_global": confmat.acc_global, "acc": confmat.acc, "iu": confmat.iu, "mean_iu": confmat.meanIU, "reduced_iu": confmat.reduced_iu, "mIoU_reduced": confmat.mIOU_reduced
        }
        if args.amp:
            checkpoint["scaler"] = scaler.state_dict()
        if confmat.mIOU_reduced >= best_mrIoU :
            best_mrIoU = confmat.mIOU_reduced
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_best.pth"))
        utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    # Save and permissions modifications
    os.chmod(os.path.join(args.output_dir, f"model_best.pth"),stat.S_IRWXU | stat.S_IRWXO)
    os.chmod(os.path.join(args.output_dir, "checkpoint.pth"), stat.S_IRWXU | stat.S_IRWXO)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

    # Evaluate the trained model at different resolutions
    resolution_evaluate(os.path.join(args.output_dir, f"model_best.pth"), device, num_classes, args)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)

    parser.add_argument("--data-path", default="/datasets01/COCO/022719/", type=str, help="dataset path")
    parser.add_argument("--dataset", default="coco", type=str, help="dataset name")
    parser.add_argument("--model", default="fcn_resnet101", type=str, help="model name")
    parser.add_argument("--aux-loss", action="store_true", help="auxiliary loss")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=8, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=30, type=int, metavar="N", help="number of total epochs to run")

    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr-warmup-method", default="linear", type=str, help="the warmup method (default: linear)")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-warmup-start-factor", default=0.1, type=float, help="Linear learning rate scheduler start factor")

    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-backbone", default=None, type=str, help="the backbone weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")

    # Added argument
    parser.add_argument("--power", default=0.9, type=float, help="polynomial scheduler power")
    parser.add_argument('--exclude-classes', nargs='+', type=int)

    parser.add_argument("--scale-low-size", default=400, type=int, help="lower value of first random scaling")
    parser.add_argument("--scale-high-size", default=1600, type=int, help="upper value of first random scaling")
    parser.add_argument("--random-crop-size", default=1024, type=int, help="the random crop size used for training (default: 1024)")

    parser.add_argument("--val_input_size", default=1024, type=int, help="val input size")
    parser.add_argument("--val_label_size", default=1024, type=int, help="val label size")

    parser.add_argument("--augmode", default=None, type=str, help="augmentation mode")
    # RegSeg parser arguments
    parser.add_argument("--regseg_name", default="custom_decoder4", type=str, help="regseg instance name(defines encoder and decoder used)")
    parser.add_argument("--first_conv_resize", default=0, type=int, help="if different than 0 rescale the input activations after the first convolution")
    parser.add_argument('--regseg_channels', nargs='+', type=int, default=None, help="RegSeg channels list")
    parser.add_argument('--regseg_gw', type=int, default=0,  help="RegSeg gw")
    return parser
    
def get_name(args) :

    name_channel = ""
    if args.regseg_channels != None :
        for element in args.regseg_channels :
            name_channel += "_" + str(element)
    else :
        name_channel = "None"
    
    if "resnet" in args.model :
        name = args.model + "_" + str(args.train_crop_size) + "_" + str(args.val_crop_size)  + "_" + str(args.val_resize_size) + "_" + str(args.first_conv_resize) + name_channel
    elif "vit" in args.model:
        name = args.model + "_" + str(args.patch_size) + "_" + str(args.num_layers) + "_" + str(args.num_heads) + "_" + str(args.hidden_dim) + "_" + str(args.mlp_dim) + "_" + str(args.img_size)
    if "regseg" in args.model :
        name = args.model + "_" + str(args.scale_low_size) + "_" + str(args.scale_high_size)  + "_" + str(args.random_crop_size) + "_" + str(args.first_conv_resize) + "_" + str(args.regseg_gw) + "_" + name_channel

    return name

if __name__ == "__main__":
    args = get_args_parser().parse_args()

    name = get_name(args)
    create_dir(args.output_dir)
    args.output_dir = args.output_dir + "/" + name
    create_dir(args.output_dir)

    wandb.init(
        # set the wandb project where this run will be logged
        project="resolution_CNN_ViT",
        name=name,
        tags=["SemanticSegmentation", "RegSeg", "torchvision_reference", "train_crop_" + str(args.scale_low_size), "val_crop_" + str(args.scale_high_size)],
        
        # track hyperparameters and run metadata
        config=args
    )
    main(args)
    wandb.finish()