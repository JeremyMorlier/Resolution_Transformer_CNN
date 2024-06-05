
import torchvision_references.references.segmentation.utils as utils1
import torchvision_references.references.segmentation.utils as utils2

def main(args, parser1):

    if parser1:
        utils1.init_distributed_mode(args)
    else :
        utils2.init_distributed_mode(args)
    print(args)

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)

    parser.add_argument("--patate", default="2", type=str, help="dataset path")
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    print(args)
    main(args, True)
