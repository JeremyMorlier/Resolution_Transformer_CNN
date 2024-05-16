import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import torch
import math
import tikzplotlib as tkplot

def score(metric1, metric2) :
    return metric1 + metric2


def pretty_print(model_name, max_scores, max_resolutions, max_max, max_max_resolutions) :
    test = [f"{resize} resize, accuracy: {round(accuracy.item(), 3)} " for resize, accuracy in zip(max_resolutions, max_scores)]
    print("Model Name: ", model_name, "\n", test, "\n Best score : ", round(max_max.item(), 3), " at ", max_max_resolutions[0].item()," resize and crop", max_max_resolutions[1].item(), "\n")


if __name__=="__main__" :
    font = {'weight' : 'bold',
            'size'   : 22}

    matplotlib.rc('font', **font)
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20) 
    matplotlib.rc('axes', titlesize=20) 

    val_crop_resolutions = [112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352]
    val_resize_resolutions = [112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352]
    models = ["112_144_152", "128_160_168", "176_224_232", "176_224_232", "80", "88"]
    models_indices = [0, 1, 3, 4, 5]
    results = torch.load("results/test.pth")
    results2 = torch.load("results/result_16_04_2024.pth")
    result = torch.cat((results, results2), 2)
    print(result.size(), results2.size())
    # 1: val_crop_resolution
    # 2: val_resize_resolution
    # 3: models
    # 4: acc1, acc5
    size = result.size()
    memories_macs = torch.load("results/memories_macs.pth")
    memories_indices = np.array(val_crop_resolutions) - 100

    colors = ["red", "blue", "green","yellow", "violet", "black"]

    plt.figure(1, figsize=(len(models_indices)*10, 10))
    for k, j in zip(models_indices,range(0, len(models_indices)))  :
        plt.subplot(160 + j + 1)
        for i in range(0, size[1]) :

            #plt.scatter(val_crop_resolutions, result[:, i, k, 0], label=val_resize_resolutions[i], )
            plt.plot(val_crop_resolutions, result[:, i, k, 0],label=val_resize_resolutions[i], marker='o', linewidth=2, markersize=12)
        plt.grid()
        
        plt.legend(loc='upper right')
        plt.ylabel("Acc1 on ImageNet")
        plt.xlabel("Validation Crop Resolution")
        plt.xlim((100, 400))
        plt.ylim((0, 80))
        
        plt.title("Resize Resolution on model " + models[k] + " Experiment")

    tkplot.save("model.tex")
    plt.savefig("model.png")
    plt.close()


    plt.figure(1, figsize=(30, 10))
    # Get max of each training method regardless of resize size
    for k, j in zip(models_indices,range(0, len(models_indices)))  :
        run_score = score(memories_macs[0, memories_indices], 1000*memories_macs[1, memories_indices])
        max_score, indices = torch.max(result[:, :, k, 0], 1)
        max_max, max_indices = torch.max(max_score, 0)
        #plt.scatter(run_score, max_score, color=colors[k])
        #plt.plot(run_score, max_score, color=colors[k], label=models[k])
        plt.plot(run_score, max_score, color=colors[k], label=models[k], marker='o', linewidth=2, markersize=12)

        # Print informations
        pretty_print(models[j], max_score, torch.tensor(val_resize_resolutions)[indices], max_max, (torch.tensor(val_resize_resolutions)[indices[max_indices]], torch.tensor(val_crop_resolutions)[max_indices]))
    plt.grid()
   
    plt.legend(loc='upper right')
    plt.ylabel("Acc1 on ImageNet")
    plt.xlabel("Flops + memory")
    #plt.xlim((100, 400))
    plt.ylim((60, 80))
    plt.title("Imagenet Top 1 accuracy in function of the number of operations and memory for several ResNet")
    # locs, labels = plt.xticks()
    # plt.xticks(locs, val_crop_resolutions)
    plt.savefig("memory_macs3.png", dpi=500)
    tkplot.save("memory_macs3.tex")
    plt.close()
    
    plt.figure(1, figsize=(30, 10))
    for k, j in zip(models_indices,range(0, len(models_indices)))  :
        run_score = score(memories_macs[0, memories_indices], 1000*memories_macs[1, memories_indices])
        for i in range(0, size[1]) :
            plt.scatter(run_score, result[:, i, k, 0], color=colors[k])
            plt.plot(run_score, result[:, i, k, 0], color=colors[k], label=models[k])

    plt.grid()
    
    plt.legend(loc='upper right')
    plt.ylabel("Acc1 on ImageNet")
    plt.xlabel("Flops + memory")
    #plt.xlim((100, 400))
    plt.ylim((60, 80))
    plt.title("ImageNetAcc1/FlopsMemory tradeoff")

    plt.savefig("memory_macs2.png")
    tkplot.save("memory_macs2.tex")
    plt.close()