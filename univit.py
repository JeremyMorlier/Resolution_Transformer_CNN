# Evaluate pretrained ViT-S on multiple resolutions using positional embedding interpolations
import os
from argparse import Namespace
import json

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2
from torchvision.datasets import ImageNet

from models import get_model, interpolate_embeddings
from memory_flops import get_memory_flops
from references.common import get_name, init_signal_handler

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

def evaluate(model, data_loader, device):

    accs1 = []
    accs5 = []

    num_processed_samples = 0

    print("Evaluating :")
    with torch.inference_mode():
        for image, target in data_loader:
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


def setup_args(patch_size, num_layers, num_heads, hidden_dim, mlp_dim, model_img_size) :
    args = Namespace()
    args.model = "vit_custom"
    args.patch_size = patch_size
    args.num_layers = num_layers
    args.num_heads = num_heads
    args.hidden_dim = hidden_dim
    args.mlp_dim = mlp_dim
    args.model_img_size = model_img_size

    return args

def get_args(model_name) :
    list_args = model_name.split("_")
    patch_size, num_layers, num_heads, hidden_dim, mlp_dim, model_img_size = [int(element) for element in list_args[2:]]
    args = setup_args(patch_size, num_layers, num_heads, hidden_dim, mlp_dim, model_img_size)

    return args 

def get_named_model(model_name, resolution, device=None, path=None) :

    list_args = model_name.split("_")
    patch_size, num_layers, num_heads, hidden_dim, mlp_dim, model_img_size = [int(element) for element in list_args[2:]]
    model = get_model("vit_custom", weights=None, num_classes=1000, patch_size=patch_size, num_layers=num_layers, num_heads=num_heads, hidden_dim=hidden_dim, mlp_dim=mlp_dim, image_size=resolution)
    
    if path != None :
        state_dict = torch.load(path)["model"]
        interpolate_embeddings(val_crop_resolution, patch_size, state_dict)
        model.load_state_dict(state_dict)
    if device != None :
        model.to(device)

    return model

def test_model_names(model_names) :

    valid_models = []
    for model_name in model_names :
        try :
            args = get_args(model_name)
            model = get_named_model(model_name, args.model_img_size, None, None)
            valid_models.append(model_name)
            print(model_name)
        except Exception as e:
            print("model not possible: ", e)

    return valid_models

if __name__ == "__main__" :

    init_signal_handler()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset_dir = os.path.join(os.getenv("DSDIR"), "imagenet")
    model_dir = os.path.join(os.getenv("WORK"), "vit")
    # format is vit dir + dir per model + checkpoint.pth
    
    models = os.listdir(model_dir)
    valid_models = test_model_names(models)
    print(models)
    print(valid_models)
    log = {}
    for model_name in models :
        if model_name in valid_models :
            path = os.path.join(os.path.join(model_dir, model_name, "checkpoint.pth"))
            log[model_name] = {}
            args = get_args(model_name)
            val_crop_resolutions = [int(args.patch_size*k) for k in range(4, 40)]
            print(args.patch_size, val_crop_resolutions)
            for val_crop_resolution in val_crop_resolutions :

                model = get_named_model(model_name, val_crop_resolution, device, path)
                memory, flops, total_memory, model_size = get_memory_flops(model, val_crop_resolution, args)

                val_resize_resolution = int(232/224*val_crop_resolution)
                # Load Dataset 
                val_transform = v2.Compose([v2.ToImage(), v2.Resize(val_resize_resolution), v2.CenterCrop(val_crop_resolution), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean = torch.tensor([0.485, 0.456, 0.406]), std = torch.tensor([0.229, 0.224, 0.225]))])
                val = ImageNet(root=dataset_dir, split="val", transform=val_transform)
                val_loader =  DataLoader(val, 128, shuffle=True, num_workers=6)

                results = evaluate(model, val_loader, device)
                log[model_name]["args"] = args.__dict__
                log[model_name]["model_size"] = model_size
                log[model_name][val_crop_resolution] = {}
                log[model_name][val_crop_resolution]["acc1"] = results[0]
                log[model_name][val_crop_resolution]["flops"] = flops
                log[model_name][val_crop_resolution]["act_memory"] = memory
                log[model_name][val_crop_resolution]["total_memory"] = total_memory
        with open(os.path.join(model_dir, "log.txt"), "a") as file :
            json.dump(log, file)
            file.write("\n")