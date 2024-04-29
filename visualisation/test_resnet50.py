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

    train_crop_size = [176, 112, 128, 224, 144, 160, 232, 152, 168]
    train_crop_size.sort()
    model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=1000)

    for train_crop in train_crop_size :
        input_size_train = (1, 3, train_crop, train_crop)
        print(input_size_train)
        info = summary(model, input_size_train, verbose=0, col_names=("output_size", "num_params", "mult_adds"))
        # print(str(info))
        # print(str(info).split("[")[1].split("]")[0])
        # print(info.summary_list)
        test = info.summary_list
        kernel_sizes = []
        input_sizes = []
        output_sizes = []
        num_params_to_strs = []
        macs_to_strs = []
        names = []
        for layer_info in test :
            names.append(str(layer_info))
            kernel_sizes.append(layer_info.kernel_size)
            input_sizes.append(layer_info.input_size)
            output_sizes.append(layer_info.output_size)
            num_params_to_strs.append(layer_info.num_params_to_str(False))
            macs_to_strs.append(layer_info.macs_to_str(False))
            #print(layer_info.kernel_size, layer_info.input_size,layer_info.output_size, layer_info.num_params_to_str(False), layer_info.macs_to_str(False))

        with open("test_" + str(train_crop) +".csv", "w+") as file :
            cswwriter = csv.writer(file, delimiter="%")
            n = len(names)
            for k in range(0, n) :
                #cswwriter.writerow([names[k]]  + [macs_to_strs[k]])
                # cswwriter.writerow([names[k]])
                #cswwriter.writerow([str(int(num_params_to_strs[k].replace(",", ""))) if num_params_to_strs[k] != "--" else num_params_to_strs[k] ])
                cswwriter.writerow([str(int(macs_to_strs[k].replace(",", ""))) if macs_to_strs[k] != "--" else macs_to_strs[k] ])
                #cswwriter.writerow([macs_to_strs[k]])
        
        print(info.total_mult_adds)


sizes = np.array([232,224,176,168,160,152,144,128,112])
macs = np.array([2072814952,1814083944,1168353640,1113974120,925809000,876488040,789850472,592705896,485370216])

plt.plot(sizes, macs/macs[0])
plt.scatter(sizes, macs/macs[0])
plt.grid()
plt.xlabel("input image size")
plt.ylabel("Total MultAdds normalized by largest")
plt.savefig("test.png")

