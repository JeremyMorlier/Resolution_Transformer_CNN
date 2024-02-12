import datetime
import time
import sys
import time
import math
import torch
import yaml
import torch.cuda.amp as amp

import copy
import random
import numpy as np
from train_utils import get_loss_fun, get_optimizer, get_dataset_loaders, get_model, get_scheduler
from project_utils.distillation import distillate_one

import argparse
import wandb
from torchinfo import summary

def evaluate(model, data_loader, device, confmat,mixed_precision):
    model.eval()
    assert(isinstance(confmat,ConfusionMatrix))
    with torch.no_grad():
        for i,(image, target) in enumerate(data_loader):
            image, target = image.to(device), target.to(device)
            with amp.autocast(enabled=mixed_precision):
                output = model(image)
            output = torch.nn.functional.interpolate(output, size=target.shape[-2:], mode='bilinear', align_corners=False)
            confmat.update(target.flatten(), output.argmax(1).flatten())

    return confmat

def train_one_epoch(model, loss_fun, optimizer, loader, lr_scheduler, mixed_precision, scaler, epoch, n_epoch, batch_size):
    model.train()
    losses=0
    for t, x in enumerate(loader):
        image, target=x
        image, target = image.cuda(), target.cuda()
        with amp.autocast(enabled=mixed_precision):
            output = model(image)
            loss = loss_fun(output, target)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        lr_scheduler.step()
        scaler.step(optimizer)
        scaler.update()

        losses+=loss.item()

        sys.stdout.write(f'\r {time.strftime("%H:%M:%S", time.gmtime())} : Epoch - {epoch + 1}/{n_epoch} Loader {t}/{len(loader)} - loss {round(loss.item() / (batch_size), 3)} ' f' - running loss {round(losses / ((t + 1) * batch_size), 3)}')
    print()
    num_iter = len(loader)

    wandb.log({"training running loss": losses/num_iter}, commit=False)
    return losses/num_iter

def save(model,optimizer,scheduler,epoch,path,best_mIU,scaler,run):
    dic={
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': scheduler.state_dict(),
        'scaler':scaler.state_dict(),
        'epoch': epoch,
        'best_mIU':best_mIU,
        "run":run
    }
    torch.save(dic,path)

def setup_env(config):
    torch.backends.cudnn.benchmark=True
    seed=0
    if "RNG_seed" in config:
        seed=config["RNG_seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed) # might remove dependency on np later

def train_one(config, device):
    setup_env(config)

    checkpoints = config["checkpoints"]
    batch_size = config["batch_size"]

    teacher_model = get_model(config["teacher_name"], config["teacher_type"], config["teacher_pretrained"])
    student_model = get_model(config["student_name"], config["student_type"], config["student_pretrained"])

    train_loader = get_dataset_loaders(config)

    total_iterations = config["msam_batch_size"] * config["msam_iters"] / batch_size
    epochs = math.ceil(total_iterations / len(train_loader))
    # epochs = math.ceil(config["msam_batch_size"] * config["msam_iters"] / (len(train_loader) * batch_size))
    optimizer = get_optimizer(student_model.model.image_encoder, config)
    loss_fun = get_loss_fun(config)
    scheduler = get_scheduler(config, optimizer)

    # # Save Model Weight and Total Flops 
    # input_size = next(iter(train_loader)).size()
    # input_size = (1, input_size[1], input_size[2], input_size[3])
    # statistics = summary(student_model.model.image_encoder, input_size=input_size, verbose=0)

    wandb.config.update({"epochs" : epochs, "total_iters" : total_iterations})

    distillate_one(config["name"], teacher_model.model.image_encoder, student_model.model.image_encoder, loss_fun, optimizer, scheduler,
                   device, train_loader, epochs, "SAM_dataset", batch_size, None, None, config["student_dim"], config["teacher_dim"], checkpoints)

    return None

def train_main(config, device):

    train_one(config, device)

def load_yaml(config_filename) :
    with open(config_filename) as file:
        config=yaml.full_load(file)
    return config

def get_args_parser(add_help=True) :
    parser = argparse.ArgumentParser(description="Foundation models distillation", add_help=add_help)

    parser.add_argument("--config", default="configs/hit_uav_500epochs.yaml", type=str, help="config path")
    parser.add_argument("--do", default=False, help="do specified config or not (default False)", action="store_true")

    parser.add_argument("--dataset_path", default="", type=str, help="dataset path")
    return parser

if __name__=='__main__':

    args = get_args_parser().parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    config_filenames = ["distillate_sam"]

    config_filename = config_filenames[0]
    config = load_yaml("configs/" + config_filename + ".yaml")
    config["dataset_dir"] = args.dataset_path if args.dataset_path else config["dataset_dir"]
    # Setup WandB
    wandb.init(
        # set the wandb project where this run will be logged
        project="Distillation_Dataset",
        name= "test",
        tags=["sam", "vit_h", "vit_t"],
        
        # track hyperparameters and run metadata
        config=config
    )

    train_main(config, device)
    wandb.finish()
