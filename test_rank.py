
import torchvision_references.references.segmentation.utils as utils


def main(args):

    utils.init_distributed_mode(args)
    print(args)

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)

    parser.add_argument("--patate", default="2", type=str, help="dataset path")

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
