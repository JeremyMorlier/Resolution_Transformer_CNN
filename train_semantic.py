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

from references.common import create_dir

from models import get_model
import datasets as datasets
import references.segmentation.presets as presets
import references.segmentation.RegSeg.presets as RS_presets
import references.segmentation.utils as utils
from references.segmentation.coco_utils import get_coco

from references.common import get_name, init_signal_handler

from memory_flops import get_memory_flops

from logger import Logger

from args import get_segmentation_argsparse 

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

    # Evaluate Model on a range of crops
    memories = []
    total_memories = []
    flops_list = []
    model_sizes = []
    
    if "vit" not in args.model :
        val_crop_resolutions = [128, 256, 384, 512, 640, 768, 896, 1024, 1280, 1536]
    else :
        val_crop_resolutions = [args.img_size]
    val_crop_resolutions.append(args.random_crop_size)
    
    for val_crop_resolution in val_crop_resolutions :
        memory, flops, total_memory, model_size = get_memory_flops(model, val_crop_resolution, args)
        memories.append(memory)
        flops_list.append(flops)
        total_memories.append(total_memory)
        model_sizes.append(model_size)

    return model, memories, flops_list, val_crop_resolutions, total_memories, model_sizes

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
            output = model(image, shape=target.shape[-2:])
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

    # In evaluation, set training architecture modifications to 0
    args.first_conv_resize = 0
    model, memories, flops_list, val_crop_resolutions, _, _ = get_param_model(args, num_classes=num_classes)
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    model_without_ddp.load_state_dict(torch.load(model_state_dict)["model"])

    args.val_label_size = 1024

    dict_results = {}
    dict_results["best_mIOU"] = 0
    dict_results["best_resize"] = 0

    for val_resize_resolution in val_resize_resolutions :
        print("Dataset loading :", val_resize_resolution)

        # Load Dataset with desired validation resolution
        args.val_input_size = val_resize_resolution
        dataset_test, _ = get_dataset(args, is_train=False)
        if args.distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
        else:
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
        )

        print(f"Dataset loaded")

        # Evaluate the model at specific resolution
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes, exclude_classes=args.exclude_classes)
        confmat.compute()

        dict_results[val_resize_resolution] = [confmat.acc_global.item(), confmat.meanIU, confmat.iu.tolist(), confmat.mIOU_reduced]
        if confmat.mIOU_reduced >= dict_results["best_mIOU"] :
            dict_results["best_mIOU"] = confmat.mIOU_reduced
            dict_results["best_resize"] = val_resize_resolution

    return dict_results, val_resize_resolutions

def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image, shape=target.shape[-2:])
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

    utils.init_distributed_mode(args)
    print(args)

    init_signal_handler()

    # Change output directory and create it if necessary
    utils.create_dir(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, args.name)
    utils.create_dir(args.output_dir)

    # Setup
    if utils.is_main_process() :

        wandb_run_id = None

        if args.resume:
            if os.path.isfile(args.resume) :
                checkpoint = torch.load(args.resume, map_location="cpu")
                if "wandb_run_id" in checkpoint :
                    wandb_run_id = checkpoint["wandb_run_id"]
                print(wandb_run_id)
        
        logger = Logger(project_name="resolution_CNN_ViT",
                        run_name=args.name,
                        tags=[args.model],
                        resume=True,
                        id=wandb_run_id,
                        args=args,
                        mode=args.logger,
                        log_dir=args.output_dir)

        run_id = logger.id

    if args.backend.lower() != "pil" and not args.use_v2:
        # TODO: Support tensor backend in V1?
        raise ValueError("Use --use_v2 if you want to use the tv_tensor or tensor backend.")
    if args.use_v2 and args.dataset != "coco":
        raise ValueError("v2 is only support supported for coco dataset for now.")

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

    model, memories, flops_list, val_crop_resolutions, total_memories, model_sizes =  get_param_model(args, num_classes)
    if utils.is_main_process() :
        print(memories, flops_list)
        logger.log({"memory":memories})
        logger.log({"model_ops":flops_list})
        logger.log({"total_memories":total_memories})
        logger.log({"model_sizes":model_sizes})
        logger.log({"measured_crops":val_crop_resolutions})

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
        if os.path.isfile(args.resume) :
            checkpoint = torch.load(args.resume, map_location="cpu")
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
        return

    best_mrIoU = 0.0
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        if utils.is_main_process() :
            logger.log({"epoch": epoch})

        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq, scaler)
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes, exclude_classes=args.exclude_classes)
        confmat.compute()
        if utils.is_main_process() :
            logger.log({"reduced_iu": confmat.reduced_iu, "mIOU_reduced": confmat.mIOU_reduced, "cuda_memory_allocated": torch.cuda.memory_allocated(device)})
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
        if utils.is_main_process() :
            checkpoint["wandb_run_id"] = logger.id
        if confmat.mIOU_reduced >= best_mrIoU :
            best_mrIoU = confmat.mIOU_reduced
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_best.pth"))
        utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    # Save and permissions modifications
    if utils.is_main_process() :
        if os.path.isfile(os.path.join(args.output_dir, f"model_best.pth")) :
            os.chmod(os.path.join(args.output_dir, f"model_best.pth"),stat.S_IRWXU | stat.S_IRWXO)
        if os.path.isfile(os.path.join(args.output_dir, "checkpoint.pth")) :
            os.chmod(os.path.join(args.output_dir, "checkpoint.pth"), stat.S_IRWXU | stat.S_IRWXO)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

    # Evaluate the trained model at different resolutions
    dict_result, val_resizes = resolution_evaluate(os.path.join(args.output_dir, f"checkpoint.pth"), device, num_classes, args)
    if utils.is_main_process() :
        logger.log({"evaluations": dict_result})
        logger.log({"val_resizes": val_resizes})

    if utils.is_main_process() :
        logger.finish()

if __name__ == "__main__":
    args, unknown_args = get_segmentation_argsparse().parse_known_args()

    args.name = get_name(args)

    main(args)
