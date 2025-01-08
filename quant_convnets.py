# Static Post-Training Quantization of Resnets and Regsegs
import os
import json
from argparse import Namespace
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.datasets import ImageNet

from models import get_model
from references.common import init_signal_handler

# Semantic Segmentation
from train_semantic import evaluate, get_dataset
import references.segmentation.utils as utils


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res

def evaluate_classification(model, data_loader, device):

    accs1 = []
    accs5 = []

    num_processed_samples = 0

    print("Evaluating :")
    with torch.inference_mode():
        for image, target in tqdm(data_loader):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]

            accs1.append(acc1)
            accs5.append(acc5)
            num_processed_samples += batch_size

        accs1 = torch.tensor(accs1, dtype=torch.float32)
        accs5 = torch.tensor(accs5, dtype=torch.float32)
        print("Acc1 : " , accs1.mean().item())
        print("Acc5 : " , accs5.mean().item())
    return accs1.mean().item(), accs5.mean().item()

def classification_evaluation(model, criterion, device, val_crop_resolutions,  args) :

    val_crop_resolutions = list(set(val_crop_resolutions))
    val_crop_resolutions.sort()

    dict_results = {}
    dict_results["best_acc1"] = 0
    dict_results["best_crop"] = 0
    dict_results["best_resize"] = 0

    for val_crop_resolution in val_crop_resolutions :

        val_resize_resolutions = [val_crop_resolution - 8, val_crop_resolution - 16, val_crop_resolution + 8, val_crop_resolution + 16,  val_crop_resolution + 24, val_crop_resolution + 32, val_crop_resolution, int(val_crop_resolution*232/224)]
        val_resize_resolutions.sort()
        
        dict_results[val_crop_resolution] = {}
        dict_results[val_crop_resolution]["best_acc1"] = 0
        dict_results[val_crop_resolution]["best_resize"] = 0

        for val_resize_resolution in val_resize_resolutions :
            print("Dataset loading :", val_crop_resolution, val_resize_resolution)

            # Load Dataset 
            val_transform = v2.Compose([v2.ToImage(), v2.Resize(val_resize_resolutions), v2.CenterCrop(val_crop_resolution), v2.ToDtype(torch.float, scale=True), v2.Normalize(mean = torch.tensor([0.485, 0.456, 0.406]), std = torch.tensor([0.229, 0.224, 0.225]))])
            val = ImageNet(root=args.data_path, split="val", transform=val_transform)
            val_loader =  DataLoader(val, 128, shuffle=True, num_workers=6)
            print("Dataset loaded")

            # Evaluate on all models
            results = evaluate_classification(model, val_loader, device)

            if results[0] >=  dict_results["best_acc1"] :
                dict_results["best_acc1"] = results[0]
                dict_results["best_crop"] = val_crop_resolution
                dict_results["best_resize"] = val_resize_resolution
            if results[0] >= dict_results[val_crop_resolution]["best_acc1"] :
                dict_results[val_crop_resolution]["best_acc1"] = results[0]
                dict_results[val_crop_resolution]["best_resize"] = val_resize_resolution

            dict_results[val_crop_resolution][val_resize_resolution] = results

    return dict_results

# Evaluate the trained model at different resolution
def segmentation_evaluation(model, device, args) :

    val_resize_resolutions = [128, 256, 384, 512, 640, 768, 896, 1024, 1280, 1536]

    args.val_label_size = 1024

    dict_results = {}
    dict_results["best_mIOU"] = 0
    dict_results["best_resize"] = 0

    for val_resize_resolution in val_resize_resolutions :
        print("Dataset loading :", val_resize_resolution)

        # Load Dataset with desired validation resolution
        args.val_input_size = val_resize_resolution
        dataset_test, _ = get_dataset(args, is_train=False)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
        )

        print(f"Dataset loaded")

        # Evaluate the model at specific resolution
        confmat = evaluate(model, data_loader_test, device=device, num_classes=args.num_classes, exclude_classes=args.exclude_classes)
        confmat.compute()

        dict_results[val_resize_resolution] = [confmat.acc_global.item(), confmat.meanIU, confmat.iu.tolist(), confmat.mIOU_reduced]
        if confmat.mIOU_reduced >= dict_results["best_mIOU"] :
            dict_results["best_mIOU"] = confmat.mIOU_reduced
            dict_results["best_resize"] = val_resize_resolution

    return dict_results


def get_args() :
    args = Namespace()
    args.dataset = "cityscapes"
    args.data_path = "/nasbrain/datasets/cityscapes/"
    args.batch_size = 32
    args.workers = 8
    args.scale_low_size, args.scale_high_size, args.random_crop_size, args.augmode, args.val_input_size, args.val_label_size = 400, 1600, 1024, "randaug_reduced",1024, 1024
    args.backend, args.use_v2 = "pil", None
    args.weights, args.test_only = None, None
    args.exclude_classes = [14, 15, 16]

    return args

def get_args(model_name, imagenet_path, cityscapes_path) :
    list_args = model_name.split("_")

    args = Namespace()
    args.model_name = list_args[0]
    if "regseg" in args.model_name :
        args.dataset = "cityscapes"
        args.data_path = cityscapes_path
        args.batch_size = 32
        args.workers = 8
        args.backend = "pil"
        args.use_v2 = None
        args.weights = None
        args.test_only = None
        args.exclude_classes = [14, 15, 16]
        args.augmode = "randaug_reduced"
        args.num_classes = 19

        args.scale_low_size, args.scale_high_size, args.random_crop_size, args.val_input_size, args.val_label_size = int(list_args[2]), int(list_args[3]), int(list_args[4]), int(list_args[4]), int(list_args[4])
        args.model = "regseg_custom"
        args.regseg_name="exp48_decoder26"
        args.first_conv_resize = 0
        args.gw = int(list_args[6])
        args.channels = [int(element) for element in list_args[-5:]]

    elif "resnet" in args.model_name :
        args.dataset = "imagenet"
        args.data_path = imagenet_path
        args.batch_size = 128
        args.workers = 8
        args.num_classes = 1000

        args.model = "resnet_resize"
        args.first_conv_resize = 0
        args.random_resized_crop = int(list_args[2])
        args.val_resize = int(list_args[4])
        args.val_crop = int(list_args[3])
        args.channels = [int(element) for element in list_args[-5:]]

    return args

def get_args_model(args, path, device) :
    if "resnet" in args.model :
        model = get_model("resnet50_resize", weights=None, num_classes=args.num_classes, first_conv_resize=args.first_conv_resize, channels=args.channels)
    elif "regseg" in args.model :
        model = get_model("regseg_custom", weights=None, weights_backbone=None, num_classes=args.num_classes, aux_loss=None, regseg_name=args.regseg_name, channels=args.channels, gw=args.gw, first_conv_resize=args.first_conv_resize)
    
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict["model"])
    model.to(device)

    return model

def init_log(model_dir) :
    filepath = os.path.join(model_dir, "log.txt")
    if os.path.isfile(filepath) :        
        with open(filepath, "r") as file :
            lines = file.readlines()
            log = json.loads(lines[0])
    else :
        log = {}
        log["evaluated models"] = []
    
    return log
def save_log(model_dir, log) :
    with open(os.path.join(model_dir, "log.txt"), "w") as file :
        json.dump(log, file)
        file.write("\n")

def test_regseg() :

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device  = torch.device("cpu")

    regseg_path = "/nasbrain/j20morli/jeanzay/results_resolution/regseg_custom_200_800_512_0_16__32_48_128_256_320/checkpoint.pth"
    dataset_path = "/nasbrain/datasets/cityscapes"
    regseg_res = 512

    model = get_model("regseg_custom", weights=None, weights_backbone=None, num_classes=19, aux_loss=None, regseg_name="exp48_decoder26", channels=[32, 48, 128, 256, 320], gw=16, first_conv_resize=0)
    state_dict = torch.load(regseg_path)
    model.load_state_dict(state_dict["model"])
    model.to(device)
    args = get_args()

    dataset_train, num_classes = get_dataset(args, True)
    dataset_test, _ = get_dataset(args, False)

    train_sampler = torch.utils.data.RandomSampler(dataset_train)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        collate_fn=utils.collate_fn,
        drop_last=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )
    # confmat = evaluate(model, data_loader_test, device, 19, args.exclude_classes)
    # confmat.compute()
    # print("reduced_iu", confmat.reduced_iu, "mIOU_reduced", confmat.mIOU_reduced)
    model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    model_fp32_prepared = torch.ao.quantization.prepare(model)
    calibrate(model_fp32_prepared, data_loader, 5)
    model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
    for module in model_int8.modules() :
        print(module)
    confmat = evaluate(model_int8, data_loader_test, device, 19, args.exclude_classes)
    confmat.compute()
    print("reduced_iu", confmat.reduced_iu, "mIOU_reduced", confmat.mIOU_reduced)

def calibrate(model, args, n_samples=100) :

    #  Load training dataset
    if "imagenet" in args.dataset :
        train_transform = v2.Compose([v2.ToImage(), v2.RandomResizedCrop(args.random_resized_crop), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean = torch.tensor([0.485, 0.456, 0.406]), std = torch.tensor([0.229, 0.224, 0.225]))])
        train = ImageNet(root=args.data_path, split="train", transform=train_transform)
        train_loader =  DataLoader(train, 128, shuffle=True, num_workers=6)
    elif "cityscapes" in args.dataset :
        dataset_train, num_classes = get_dataset(args, True)
        dataset_test, _ = get_dataset(args, False)

        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.workers,
            collate_fn=utils.collate_fn,
            drop_last=True,
        )

    # Calibrate
    i = 0
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(n_samples)) :
            image, target = next(iter(train_loader))
            model(image)

def evaluate_args(model, args) :
    #  Load Validation dataset
    if "imagenet" in args.dataset :
        criterion = torch.nn.CrossEntropyLoss()
        val_resize_resolutions = [120, 152, 184, 216, 248, 280, 312, 344]
        return classification_evaluation(model, criterion, device, val_resize_resolutions,  args)
    elif "cityscapes" in args.dataset :
        return segmentation_evaluation(model, device, args)
        
if __name__ == "__main__" :

    # Paths 
    models_dir = os.path.join(os.getenv("WORK"), "Convnets")
    imagenet_path = os.path.join(os.getenv("DSDIR"), "imagenet")
    cityscapes_path = os.path.join(os.getenv("SCRATCH"), "cityscapes")

    init_signal_handler()
    device = torch.device('cpu')

    models = os.listdir(models_dir)
    
    log = init_log(models_dir)
    for model_name in models :
        if ".txt" not in model_name :
            path = os.path.join(os.path.join(models_dir, model_name, "checkpoint.pth"))
            log[model_name] = {}
            args = get_args(model_name, imagenet_path, cityscapes_path)

            model = get_args_model(args, path, device)
            results = evaluate_args(model, args)
            log[model_name]["results"] = results
            model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
            model_fp32_prepared = torch.ao.quantization.prepare(model)

            calibrate(model_fp32_prepared, args)
            model_int8 = torch.ao.quantization.convert(model_fp32_prepared)

            results = evaluate_args(model, args)

            log[model_name]["args"] = args.__dict__
            log[model_name]["ptq_results"] = results
            log["evaluated models"].append(model_name)
            save_log(models_dir, log)

# if __name__ == "__main__" :

#     test_regseg()
#     # Paths
#     resnet_path = "/nasbrain/j20morli/jeanzay/results_resolution/resnet50_resize_128_162_167_0_64_64_128_256_512/checkpoint.pth"
#     resnet_path = "/nasbrain/j20morli/jeanzay/results_resolution/resnet50_resize_176_224_232_0_128_128_256_512_1024/checkpoint.pth"
#     resnet_path = "/nasbrain/j20morli/jeanzay/2_results_resnet/resnet50_resize_128_224_232_0_64_64_128_256_512/checkpoint.pth"
#     resnet_res = 162
#     resnet_res = 128
#     dataset_dir = "/nasbrain/j20morli/eval_clip/imagenet"

#     # Load Dataset 
#     train_transform = v2.Compose([v2.ToImage(), v2.RandomResizedCrop(resnet_res), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean = torch.tensor([0.485, 0.456, 0.406]), std = torch.tensor([0.229, 0.224, 0.225]))])
#     train = ImageNet(root=dataset_dir, split="train", transform=train_transform)
#     train_loader =  DataLoader(train, 128, shuffle=True, num_workers=6)

#     val_transform = v2.Compose([v2.ToImage(), v2.Resize(int(resnet_res*232/224)), v2.CenterCrop(resnet_res), v2.ToDtype(torch.float, scale=True), v2.Normalize(mean = torch.tensor([0.485, 0.456, 0.406]), std = torch.tensor([0.229, 0.224, 0.225]))])
#     val = ImageNet(root=dataset_dir, split="val", transform=val_transform)
#     val_loader =  DataLoader(val, 128, shuffle=True, num_workers=6)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     device  = torch.device("cpu")
#     # Model Loading
#     # model = get_model("resnet50_resize", weights=None, num_classes=1000, first_conv_resize=0, channels=[64, 64, 128, 256, 512])
#     # model = get_model("resnet50_resize", weights=None, num_classes=1000, first_conv_resize=0, channels=[128, 128, 256, 512, 1024])
#     model = get_model("resnet50_resize", weights=None, num_classes=1000, first_conv_resize=0, channels=[64, 64, 128, 256, 512])
#     state_dict = torch.load(resnet_path)
#     print(state_dict.keys(), state_dict["epoch"], state_dict["acc1_epoch"])
#     model.load_state_dict(torch.load(resnet_path)["model"])
#     #model = resnet50(weights=ResNet50_Weights.DEFAULT)
#     model.to(device)

#     #evaluate(model, val_loader, torch.device("cuda"))
#     model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
#     model_fp32_prepared = torch.ao.quantization.prepare(model)
#     calibrate(model_fp32_prepared, train_loader)
#     model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
#     for module in model_int8.modules() :
#         print(module)
#     evaluate_classification(model_int8, val_loader, torch.device("cpu"))