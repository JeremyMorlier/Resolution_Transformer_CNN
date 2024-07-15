
import argparse

import numpy as np

import torch
import torch.optim as optim

from mobile_sam.modeling import TinyViT

# Contains common functions betwwen train.py and train_parallel.py
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # datasets paths
    parser.add_argument('--dataset_path', type=str, default="/dataset/vyueyu/sa-1b", help='root path of dataset')
    parser.add_argument('--ade_dataset', type=str, default=None, help='Path of ADE20k dataset folder')

    # Model settings
    parser.add_argument("--model", type=str, default="mobilesam_vit", help="model that will be distilled (sam_vit_h,l,b,t mobilesam_vit or Resnet50)")
    parser.add_argument("--sam_checkpoint", type=str, default=None, help="path of Segment Anything Pretrained ViT_H")

    # training epochs, batch size and so on
    parser.add_argument('--epochs', type=int, default=8, help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    # multi gpu settings
    parser.add_argument("--local_rank", type=int, default=-1)

    # cuda settings
    # parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--seed', type=int, default=1234, help='seed')
    parser.add_argument('--deterministic', type=bool, default=True, help='deterministic')
    parser.add_argument('--benchmark', type=bool, default=False, help='benchmark')

    # learning process settings
    parser.add_argument('--optim', type=str, default='sgd', choices=['adam', 'sgd', 'adamw'])
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # print and evaluate frequency during training
    parser.add_argument('--print_iters', type=int, default=200, help='print loss iterations')
    parser.add_argument('--eval_nums', type=int, default=200, help='evaluation numbers')
    parser.add_argument('--eval_iters', type=int, default=500, help='evaluation iterations')

    # file and folder paths
    parser.add_argument('--root_path', type=str, default="/users/local/j20morli/MobileSAM-pytorch/MobileSAM/", help='root path')
    parser.add_argument('--root_feat', type=str, default="/users/local/j20morli/MobileSAM-pytorch/MobileSAM/", help='root features path')
    parser.add_argument('--work_dir', type=str, default="work_dir", help='work directory')
    parser.add_argument('--save_dir', type=str, default="ckpt", help='save directory')
    parser.add_argument('--log_dir', type=str, default="log", help='save directory')
    parser.add_argument('--save_iters', type=int, default=50000, help='save iterations')

    # SAM directories used for training and evaluation
    parser.add_argument('--train_dirs', nargs='+', type=str)
    parser.add_argument('--val_dirs', nargs='+', type=str)
    args = parser.parse_args()
    return args

# Utilities
def find_center(img): 
    img_binary = (img != 0)*1
    if np.sum(img_binary) > 100 : 
        mass_x, mass_y = np.where(img_binary == 1)
        cent_x = int(np.average(mass_x))
        cent_y = int(np.average(mass_y))
        center_point = np.array([[cent_y, cent_x]]) 
        if img_binary[cent_x, cent_y] == 1 : 
            valid_mask = True
        else : 
            valid_mask = False
        return center_point, valid_mask
    else : 
        return np.array([0,0]), False 

def calculate_iou(segmentation1, segmentation2):
    intersection = np.sum(segmentation1 * segmentation2)
    union = np.sum(segmentation1) + np.sum(segmentation2) - intersection
    iou = (intersection / union) * 100
    return iou

def get_optimizer(args, model):
    if args.optim == 'adam':
        return optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        return optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        return optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(args.optim)

def get_scheduler(args, optimizer):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.5)
    
def customized_mseloss(pred_feats, target_feats):
    # return (0.5 * (pred_feats - target_feats) ** 2).sum(1).mean()
    return ((pred_feats - target_feats) ** 2).sum(1).mean().sqrt()