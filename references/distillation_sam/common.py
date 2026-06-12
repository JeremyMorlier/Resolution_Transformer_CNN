
import argparse

import numpy as np

import torch
import torch.optim as optim

from mobile_sam.modeling import TinyViT

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