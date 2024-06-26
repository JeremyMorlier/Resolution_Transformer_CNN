import os

import torch
from torch.utils.data import DataLoader, Subset

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torchvision.datasets import ImageNet

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

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--dir", default="checkpoints/", type=str, help="Directory that contains the different model checkpoints")
    parser.add_argument("--output", default="test", type=str, help="name of the output")
    parser.add_argument("--checkpoint_name", default="model_best", type=str, help="name of the selected checkpoint")

    parser.add_argument("--dataset_dir", default="/nasbrain/j20morli/eval_clip/imagenet/", type=str, help="Directory that contains the dataset")
    parser.add_argument("--model", default="resnet50", type=str, help="model that is evaluated")

    return parser


def get_all_weights_path(dir, checkpoint_name) :

    import os

    path_list = []
    print(os.listdir(dir))
    for element_path in os.listdir(dir) :

        if os.path.isdir(os.path.join(dir, element_path)) :

            checkpoint_path = os.path.join(dir,element_path, checkpoint_name + ".pth")

            if os.path.exists(checkpoint_path) :

                path_list.append(checkpoint_path)
                         
    return path_list

if __name__ == "__main__" :
    
    args = get_args_parser().parse_args()

    val_crop_resolutions = [112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352]
    val_resize_resolutions = [120, 136, 152, 168, 184, 200, 216, 232, 248, 264, 280, 296, 312, 328, 344, 360]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path_list = get_all_weights_path(args.dir, args.checkpoint_name)
    print(path_list)
    all_results = []
    for val_crop_resolution in val_crop_resolutions :
        global_results = []
        for val_resize_resolution in val_resize_resolutions :
            local_results = []
            print("Dataset loading :", val_crop_resolution, val_resize_resolution)

            # Load Dataset 
            val_transform = v2.Compose([v2.ToImage(), v2.Resize(val_resize_resolution), v2.CenterCrop(val_crop_resolution), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean = torch.tensor([0.485, 0.456, 0.406]), std = torch.tensor([0.229, 0.224, 0.225]))])
            val = ImageNet(root=args.dataset_dir, split="val", transform=val_transform)
            val_loader =  DataLoader(val, 128, shuffle=True, num_workers=6)

            print("Dataset loaded")

            for model_path in path_list :
                print(val_crop_resolution, val_resize_resolution, model_path)
                model = torchvision.models.get_model(args.model, num_classes=1000)
                model.to(device)
                model.load_state_dict(torch.load(model_path)["model"])

                # Evaluate on all models
                results = evaluate(model, val_loader, device)

                local_results.append(results)
            
            global_results.append(local_results)
        all_results.append(global_results)
    save_tensor = torch.tensor(all_results)
    torch.save(save_tensor, os.path.join(args.dir, args.output + ".pth"))
    
        