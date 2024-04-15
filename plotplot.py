import matplotlib.pyplot as plt
import numpy as np

import torch


def score(metric1, metric2) :
    return metric1 + metric2

if __name__=="__main__" :

    val_crop_resolutions = [112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352]
    val_resize_resolutions = [112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352]
    models = ["112_144_152", "128_160_168", "176_224_232"]
    results = torch.load("results/test.pth")
    # 1: val_crop_resolution
    # 2: val_resize_resolution
    # 3: models
    # 4: acc1, acc5
    memories_macs = torch.load("results/memories_macs.pth")
    memories_indices = np.array(val_crop_resolutions) - 100
    # 0 : macs, 1 : memories
    print(memories_macs)
    size = results.size()
    print(memories_macs.size(), memories_indices)
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

    plt.savefig("model.png")
    plt.close()

    plt.figure(1, figsize=(30, 10))
    colors = ["red", "blue", "green"]
    for k in range(0, size[2]) :
        run_score = score(memories_macs[0, memories_indices], 1000*memories_macs[1, memories_indices])
        for i in range(0, size[1]) :
            plt.scatter(run_score, results[:, i, k, 0], color=colors[k])
            plt.plot(run_score, results[:, i, k, 0], color=colors[k], label=models[k])

        plt.grid()
        
        plt.legend(loc='upper right')
        plt.ylabel("Acc1 on ImageNet")
        plt.xlabel("Flops + memory")
        #plt.xlim((100, 400))
        plt.ylim((0, 80))
        plt.title("ImageNetAcc1/FlopsMemory tradeoff")

    plt.savefig("memory_macs.png")
    plt.close()