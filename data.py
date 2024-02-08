
import matplotlib.pyplot as plt

from data_utils import *
import torch
from torch.utils.data import DataLoader

from torchvision.datasets import ImageNet
from datasets.ADE20k import ADE20k

import transforms as T

def build_train_transform_semgsem(aug_mode) :
    return None

def build_val_transform_semgsem(aug_mode) : 
    return None

def build_train_transform_classification(aug_mode) :
    return None

def build_val_transform_classification(aug_mode) : 
    return None


def get_ADE20k(root, train_split, val_split, batch_size, aug_mode, num_workers) :
    train_transform=build_train_transform_semgsem(aug_mode)
    val_transform=build_val_transform_semgsem(aug_mode)

    train = ADE20k(root, train_split, train_transform)
    val = ADE20k(root, val_split, val_transform)

    train_loader = get_dataloader_train(train, batch_size, num_workers)
    val_loader = get_dataloader_val(val, num_workers)

    return train_loader, val_loader, train

def get_ImageNet(root, train_split, val_split, batch_size_train, batch_size_val, aug_mode, size_train, size_val, num_workers, shuffle) :
    train_transform=build_train_transform_classification(aug_mode, size_train)
    val_transform=build_val_transform_classification(aug_mode, size_val)

    train = ImageNet(root, train_split, train_transform)
    val = ImageNet(root, val_split, val_transform)

    train_loader = DataLoader(train, batch_size_train, shuffle, num_workers=num_workers)
    val_loader =  DataLoader(val, batch_size_val, shuffle, num_workers=num_workers)

    return train_loader, val_loader, train

if __name__ == "__main__" :
    import yaml

    def load_yaml(config_filename) :
        with open(config_filename) as file:
            config=yaml.full_load(file)
        return config
    
    config_filename = "test_config"
    config = load_yaml("configs/" + config_filename + ".yaml")
    root = config["root"]
    split = config["split"]

    transforms = T.ToTensor()
    dataset = ADE20k(root, split, transforms)
    print(len(dataset))
    image, target = dataset[0]

    print(image.size(), image.size())