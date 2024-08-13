import datetime
import os, stat
import time
import warnings

from logger import Logger
import torch
import torch.utils.data
import torchvision
import torchvision.transforms

from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode

import references.classification.presets as presets
from references.classification.transforms import get_mixup_cutmix
import references.classification.utils as utils
from references.classification.sampler import RASampler
from references.common import get_name, init_signal_handler
from models import get_model

from args import get_classification_argsparse
from memory_flops import get_memory_flops

def get_param_model(args, num_classes) :
    
    # Evaluate all wanted resolutions of the model
    if "vit" not in args.model :
        val_crop_resolutions = [112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352]
    else :
        val_crop_resolutions = [args.img_size]
    val_crop_resolutions.append(args.val_crop_size)

    memories = []
    flops_list = []

    if args.model == "resnet50_resize" :
        model = get_model(args.model, weights=args.weights, num_classes=num_classes, first_conv_resize=args.first_conv_resize, channels=args.channels, depths=args.depths)
    elif args.model == "vit_custom" :
        model = get_model(args.model, weights=args.weights, num_classes=num_classes, patch_size=args.patch_size, num_layers=args.num_layers, num_heads=args.num_heads, hidden_dim=args.hidden_dim, mlp_dim=args.mlp_dim, image_size=args.img_size)
    else :
        model = get_model(args.model, weights=args.weights, num_classes=num_classes)

    for val_crop_resolution in val_crop_resolutions :
        memory, flops = get_memory_flops(model, val_crop_resolution, args)
        memories.append(memory)
        flops_list.append(flops)
    return model, memories, flops_list, val_crop_resolutions

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    running_loss = 0.0
    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)
        
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        running_loss += loss.detach().item()
        
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
    return {"running_loss": running_loss/(len(data_loader)*batch_size), "average acc1":metric_logger.acc1.avg, "average acc5":metric_logger.acc5.avg}


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

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

    metric_logger.synchronize_between_processes()
   
    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg, metric_logger.acc5.global_avg

def resolution_evaluate(model_state_dict, criterion, device, num_classes, val_resize_resolutions,  args) :

    # In evaluation, set training architecture modifications to 0
    args.first_conv_resize = 0
    model, memory, flops, val_crop_resolutions = get_param_model(args, num_classes=num_classes)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    model_without_ddp.load_state_dict(torch.load(model_state_dict)["model"])

    val_crop_resolutions = list(set(val_crop_resolutions))
    val_crop_resolutions.sort()

    dict_results = {}
    dict_results["best_acc1"] = 0
    dict_results["best_crop"] = 0
    dict_results["best_resize"] = 0

    for val_crop_resolution in val_crop_resolutions :

        val_resize_resolutions = [val_crop_resolution - 8, val_crop_resolution - 16, val_crop_resolution + 8, val_crop_resolution + 16,  val_crop_resolution + 24, val_crop_resolution, int(val_crop_resolution*232/224)]
        val_resize_resolutions.sort()
        
        dict_results[val_crop_resolution] = {}
        dict_results[val_crop_resolution]["best_acc1"] = 0
        dict_results[val_crop_resolution]["best_resize"] = 0

        for val_resize_resolution in val_resize_resolutions :
            print("Dataset loading :", val_crop_resolution, val_resize_resolution)

            # Load Dataset 
            dataset_test, test_sampler = load_val_data(os.path.join(args.data_path, "val"), val_crop_resolution, val_resize_resolution, args)
            data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True)

            print("Dataset loaded")

            # Evaluate on all models
            results = evaluate(model, criterion, data_loader_test, device)

            if results[0] >=  dict_results["best_acc1"] :
                dict_results["best_acc1"] = results[0]
                dict_results["best_crop"] = val_crop_resolution
                dict_results["best_resize"] = val_resize_resolution
            if results[0] >= dict_results[val_crop_resolution]["best_acc1"] :
                dict_results[val_crop_resolution]["best_acc1"] = results[0]
                dict_results[val_crop_resolution]["best_resize"] = val_resize_resolution

            dict_results[val_crop_resolution][val_resize_resolution] = results


    if utils.is_main_process() :
        torch.save(dict_results, os.path.join(args.output_dir, "resolution.pth"))
        os.chmod(os.path.join(args.output_dir, "resolution.pth"), stat.S_IRWXU | stat.S_IRWXO)

    return dict_results, val_crop_resolutions, val_resize_resolution

def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path

def load_val_data(valdir, resolution_crop_size, resolution_resize_size, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size = (
        resolution_resize_size,
        resolution_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_test from {cache_path}")
        # TODO: this could probably be weights_only=True
        dataset_test, _ = torch.load(cache_path, weights_only=False)
    else:
        if args.weights and args.test_only:
            weights = torchvision.models.get_weight(args.weights)
            preprocessing = weights.transforms(antialias=True)
            if args.backend == "tensor":
                preprocessing = torchvision.transforms.Compose([torchvision.transforms.PILToTensor(), preprocessing])

        else:
            preprocessing = presets.ClassificationPresetEval(
                crop_size=val_crop_size,
                resize_size=val_resize_size,
                interpolation=interpolation,
                backend=args.backend,
                use_v2=args.use_v2,
            )

        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            preprocessing,
        )
        if args.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset_test, test_sampler

def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_train from {cache_path}")
        # TODO: this could probably be weights_only=True
        dataset, _ = torch.load(cache_path, weights_only=False)
    else:
        # We need a default value for the variables below because args may come
        # from train_quantization.py which doesn't define them.
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        ra_magnitude = getattr(args, "ra_magnitude", None)
        augmix_severity = getattr(args, "augmix_severity", None)
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            presets.ClassificationPresetTrain(
                crop_size=train_crop_size,
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
                ra_magnitude=ra_magnitude,
                augmix_severity=augmix_severity,
                backend=args.backend,
                use_v2=args.use_v2,
            ),
        )
        if args.cache_dataset:
            print(f"Saving dataset_train to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_test from {cache_path}")
        # TODO: this could probably be weights_only=True
        dataset_test, _ = torch.load(cache_path, weights_only=False)
    else:
        if args.weights and args.test_only:
            weights = torchvision.models.get_weight(args.weights)
            preprocessing = weights.transforms(antialias=True)
            if args.backend == "tensor":
                preprocessing = torchvision.transforms.Compose([torchvision.transforms.PILToTensor(), preprocessing])

        else:
            preprocessing = presets.ClassificationPresetEval(
                crop_size=val_crop_size,
                resize_size=val_resize_size,
                interpolation=interpolation,
                backend=args.backend,
                use_v2=args.use_v2,
            )

        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            preprocessing,
        )
        if args.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


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
                        tags=[args.model , "torchvision_reference", "train_crop_" + str(args.train_crop_size), "val_crop_" + str(args.val_crop_size)],
                        resume=True,
                        id=wandb_run_id,
                        args=args,
                        mode=args.logger,
                        log_dir=args.output_dir)
        run_id = logger.id

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

    num_classes = len(dataset.classes)
    num_classes = 1000
    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, num_categories=num_classes, use_v2=args.use_v2
    )
    if mixup_cutmix is not None:

        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))

    else:
        collate_fn = default_collate

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )
    print(num_classes)
    print("Creating model")
    model, memory, flops, val_crop_resolutions = get_param_model(args, num_classes=num_classes)
    model.to(device)

    if utils.is_main_process() :
        print(memory, flops)
        logger.log({"memory":memory})
        logger.log({"model_ops":flops})
        logger.log({"measured_crops":val_crop_resolutions})

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler


    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        if os.path.isfile(args.resume) :
            checkpoint = torch.load(args.resume, map_location="cpu")
            print(checkpoint.keys())
            model_without_ddp.load_state_dict(checkpoint["model"])
            if not args.test_only:
                if "optimizer" in checkpoint :
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if "lr_scheduler" in checkpoint :
                    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            if "epoch" in checkpoint :
                args.start_epoch = checkpoint["epoch"] + 1
            if model_ema:
                model_ema.load_state_dict(checkpoint["model_ema"])
            if scaler:
                scaler.load_state_dict(checkpoint["scaler"])

    # if args.test_only:
    #     # We disable the cudnn benchmarking because it can noticeably affect the accuracy
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True
    #     if model_ema:
    #         evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
    #     else:
    #         evaluate(model, criterion, data_loader_test, device=device)
    #     return

    print("Start training")
    start_time = time.time()

    best_acc1 = 0.0

    for epoch in range(args.start_epoch, args.epochs):

        if utils.is_main_process() :
            logger.log({"epoch": epoch})

        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_results = train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler)
        if utils.is_main_process() :
            logger.log(train_results)
        lr_scheduler.step()
        acc1_epoch, acc5_epoch = evaluate(model, criterion, data_loader_test, device=device)

        if utils.is_main_process() :
            logger.log({"val global acc1":acc1_epoch, "val global acc5":acc5_epoch})

        if model_ema:
            acc1_ema, acc5_ema = evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
            if utils.is_main_process() :
                logger.log({"ema acc1":acc1_ema, "ema acc5":acc5_ema})
        if args.output_dir:
            print("Saving model")
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
                "acc1_epoch": acc1_epoch,
                "acc5_epoch": acc5_epoch,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            if utils.is_main_process() :
                checkpoint["wandb_run_id"] = logger.id
            if acc1_epoch >= best_acc1 :
                best_acc1 = acc1_epoch
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_best.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))
    
    if utils.is_main_process() :
        if os.path.isfile(os.path.join(args.output_dir, f"model_best.pth")) :
            os.chmod(os.path.join(args.output_dir, f"model_best.pth"),stat.S_IRWXU | stat.S_IRWXO)
        if os.path.isfile(os.path.join(args.output_dir, f"checkpoint.pth")) :
            os.chmod(os.path.join(args.output_dir, "checkpoint.pth"), stat.S_IRWXU | stat.S_IRWXO)

    training_time = time.time()
    total_time = training_time - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

    # Evaluate the model on a range of crop and resize resolutions
    val_resize_resolutions = [120, 136, 152, 168, 184, 200, 216, 232, 248, 264, 280, 296, 312, 328, 344, 360]

    results, val_crop_resolutions, val_resize_resolutions = resolution_evaluate(os.path.join(args.output_dir, f"checkpoint.pth"), criterion, device, num_classes, val_resize_resolutions,  args)
    
    if utils.is_main_process():
        logger.log({"acc_resolutions": results, "val_crop_resolutions": val_crop_resolutions, "val_resize_resolution": val_resize_resolutions})

    total_time = time.time() - training_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Evaluation time {total_time_str}")

    # Close Logger
    if utils.is_main_process():
        logger.finish()

if __name__ == "__main__":
    args, unknown_args = get_classification_argsparse().parse_known_args()
    print(args)
    args.name = get_name(args)

    main(args)
