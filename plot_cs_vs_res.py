import datetime
import os, stat
import time
import warnings
import itertools 
import csv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tikzplotlib as tkplot

import torch
import torch.utils.data
import torchvision
import torchvision.transforms

from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode

import torchvision_references.references.classification.presets as presets
from torchvision_references.references.classification.transforms import get_mixup_cutmix
import torchvision_references.references.classification.utils as utils
from torchvision_references.references.classification.sampler import RASampler
from torchvision_references.models import get_model

import wandb

from torchinfo import summary
def get_param_model(first_conv_resize, channels, num_classes) :
    model = get_model("resnet50_resize", weights=None, num_classes=num_classes, first_conv_resize=first_conv_resize, depths=None, channels=channels)
    
    return model


if __name__ == "__main__" :

    font = {'weight' : 'bold',
            'size'   : 22}

    matplotlib.rc('font', **font)
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20) 
    matplotlib.rc('axes', titlesize=20) 

    markersize = 3
    marker='o'
    linewidth=2

    num_classes = 1000
    first_conv_resize = 0
    channels = [64, 128, 256, 512]
    
    # get inference flops and memory for default Resnet50 at different input resolutions
    model = get_param_model(first_conv_resize, channels, num_classes)
    train_crop_size = np.arange(100, 400)
    macs = []
    memories = []
    for train_crop in train_crop_size :
        input_size_train = (1, 3, train_crop, train_crop)
        info = summary(model, input_size_train, col_names=("output_size", "num_params", "mult_adds"))

        macs.append(info.total_mult_adds)
        memories.append(info.max_memory)
        print(input_size_train, info.total_mult_adds, info.max_memory)


    # get inference flops and memory for Resnet50 with different depth scaling
    eval_channels = [16, 32, 64, 128, 256, 512]
    macs2 = []
    memories2 = []
    train_crop_size2 = []
    channels_list = []
    for channels in itertools.product(eval_channels, eval_channels, eval_channels, eval_channels) :
        model = get_param_model(first_conv_resize, channels, num_classes)
        input_size_train = (1, 3, 232, 232)
        info = summary(model, input_size_train, verbose=0, col_names=("output_size", "num_params", "mult_adds"))

        channels_list.append(channels)
        macs2.append(info.total_mult_adds)
        memories2.append(info.max_memory)
        train_crop_size2.append(232)
        print(channels, info.total_mult_adds, info.max_memory)
    


    # Save results in csv files
    header = ["Channels", "Channels", "Channels", "Channels", "Input Resolution", "Total Mults Adds", "Max Memory"]
    with open("results1.csv", "w+") as file :
        csvwriter = csv.writer(file)
        csvwriter.writerow(header)
        for train_crop, mac, memory in zip(train_crop_size, macs, memories) :
            csvwriter.writerow([64, 128, 256, 512, train_crop, mac, memory])

    with open("results2.csv", "w+") as file :
        csvwriter = csv.writer(file)
        csvwriter.writerow(header)
        for channels, mac, memory in zip(channels_list, macs2, memories2) :
            csvwriter.writerow(list(channels) + [232, mac, memory])

    plt.figure(1, figsize=(10, 10))
    #plt.subplot(121)
    plt.plot(train_crop_size, macs, marker=marker, linewidth=linewidth, markersize=markersize, color="blue", label="Resolution Scaling")
    plt.scatter(train_crop_size2, macs2, marker=marker, s=markersize, color="red", label="Channel Scaling")
    plt.grid()
    
    plt.legend(loc='upper right')
    plt.ylabel("Macs")
    plt.xlabel("Input Image resolution")
    #plt.xlim((90, 420))
    
    plt.title("Resize Resolution on model ")

    plt.savefig("model.png")
    tkplot.save("model.tex")
    plt.close()


    plt.figure(1, figsize=(10, 10))
    #plt.subplot(122)
    plt.plot(train_crop_size, memories, marker=marker, linewidth=linewidth, markersize=markersize, color="blue", label="Resolution Scaling")
    plt.scatter(train_crop_size2, memories2, marker=marker, s=markersize, color="red", label="Channel Scaling")
    plt.grid()
    
    plt.legend(loc='upper right')
    plt.ylabel("Inference Memory")
    plt.xlabel("Input Image resolution")
    #plt.xlim((90, 420))
    
    plt.title("Resize Resolution on model ")

    plt.savefig("model3.png")
    tkplot.save("model3.tex")
    plt.close()