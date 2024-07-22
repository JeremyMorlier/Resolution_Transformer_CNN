import argparse
import torch


def args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Test", add_help=add_help)
    parser.add_argument("--path", default="", type=str, help="Path of checkpoint file")

    return parser

def pth_reader(path) :
    torch_dict = torch.load(path)
    print(torch_dict)
    print(torch_dict.keys())

if __name__ == "__main" :
    args = args_parser.parse_args()
    pth_reader(args.path)