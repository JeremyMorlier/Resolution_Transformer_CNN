import csv
import torch
import torchvision
from torchinfo import summary

import numpy as np
import matplotlib.pyplot as plt

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--model", default="resnet18", type=str, help="model name")

    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    train_crop_size = np.arange(224, 100, -1)
    model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=1000)

    macs = []

    for train_crop in train_crop_size :
        input_size_train = (1, 3, train_crop, train_crop)
        if train_crop in [160, 161] :
            info = summary(model, input_size_train, col_names=("output_size", "num_params", "mult_adds"))
        else :
            info = summary(model, input_size_train, verbose=0, col_names=("output_size", "num_params", "mult_adds"))
        macs.append(info.total_mult_adds)
        print(input_size_train, info.total_mult_adds)
    macs = np.array(macs)
    plt.plot(train_crop_size, macs/macs[0])
    plt.scatter(train_crop_size, macs/macs[0])
    plt.grid()
    plt.xlabel("input image size")
    plt.ylabel("Total MultAdds normalized by largest")
    plt.savefig("test_Normalize.png")
    plt.close()

    plt.plot(train_crop_size, macs)
    plt.scatter(train_crop_size, macs)
    plt.grid()
    plt.xlabel("input image size")
    plt.ylabel("Total MultAdds")
    plt.savefig("test_notNormalize.png")
