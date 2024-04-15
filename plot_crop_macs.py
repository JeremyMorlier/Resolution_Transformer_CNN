import csv
import torch
import torchvision
from torchinfo import summary

import numpy as np
import matplotlib.pyplot as plt
import time
from torch.profiler import profile, record_function, ProfilerActivity
def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--model", default="resnet18", type=str, help="model name")

    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    train_crop_size = np.arange(400, 100, -1)
    model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=1000).to(device)

    macs = []
    memories = []
    times = []

    for train_crop in train_crop_size :
        input_size_train = (1, 3, train_crop, train_crop)
        input_size_train2 = (100, 3, train_crop, train_crop)
        test = torch.rand(input_size_train2).to(device)

        if train_crop in [160, 161] :
            info = summary(model, input_size_train, col_names=("output_size", "num_params", "mult_adds"))
        else :
            info = summary(model, input_size_train, verbose=0, col_names=("output_size", "num_params", "mult_adds"))
        macs.append(info.total_mult_adds)
        memories.append(info.max_memory)
        print(input_size_train, "Total Mult Adds: ", info.total_mult_adds, "  Memory Needed for inference: ", info.max_memory)

        # # Warmup
        # model(test)
        # model(test)
        # model(test)
        # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        #     with record_function("model_inference"):
        #         model(test)
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        # #print(prof.key_averages())
        # #print(prof.total_average())
        # print(prof.profiler.self_cpu_time_total)
        # #print(prof.function_events.self_cpu_time_total)

    results = []
    results.append(macs)
    results.append(memories)
    torch_results = torch.tensor(results)
    torch.save(torch_results, "results/memories_macs.pth")
    macs = np.array(macs)
    plt.plot(train_crop_size, macs/macs[0])
    plt.scatter(train_crop_size, macs/macs[0])
    plt.grid()
    plt.xlabel("input image size")
    plt.ylabel("Total MultAdds normalized by largest")
    plt.savefig("test_Normalize.svg")
    plt.close()

    plt.plot(train_crop_size, macs)
    plt.scatter(train_crop_size, macs)
    plt.grid()
    plt.xlabel("input image size")
    plt.ylabel("Total MultAdds")
    plt.savefig("test_notNormalize.svg")
    