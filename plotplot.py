import matplotlib.pyplot as plt
import numpy as np

import torch


if __name__=="__main__" :

    val_crop_resolutions = [112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352]
    val_resize_resolutions = [112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352]
    models = ["112_144_152", "128_160_168", "176_224_232"]
    results = torch.load("test.pth")
    # 1: val_crop_resolution
    # 2: val_resize_resolution
    # 3: models
    # 4: acc1, acc5
    size = results.size()

    plt.figure(1, figsize=(30, 10))
    for k in range(0, size[2]) :
        plt.subplot(130 + k + 1)
        for i in range(0, size[1]) :

            plt.scatter(val_crop_resolutions, results[:, i, k, 0], label=val_resize_resolutions[i])
            plt.plot(val_crop_resolutions, results[:, i, k, 0] )
        plt.grid()
        
        plt.legend(loc='upper right')
        plt.ylabel("Acc1 on ImageNet")
        plt.xlabel("Validation Crop Resolution")
        plt.xlim((100, 400))
        plt.ylim((0, 80))
        
        plt.title("Resize Resolution on model " + models[k] + " Experiment")

    plt.savefig("model.svg")
    plt.close()