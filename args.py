import argparse


def slurm_args_parser(parser) :
    parser.add_argument("--job_name", type=str, default="default", help="slurm job name")
    parser.add_argument("--output", type=str, default="log/default/%j/logs.out", help="slurm log folder path")
    parser.add_argument("--error", type=str, default="log/default/%j/errors.err", help="slurm job name")
    parser.add_argument("--constraint", type=str, default="v100-16g", help="slurm node constraint")
    parser.add_argument("--nodes", type=int, default=1, help="number of nodes")
    parser.add_argument("--ntasks", type=int, default=1, help="number of tasks (ideally number of nodes)")
    parser.add_argument("--gres", type=str, default="gpu:4", help="number of gpus")
    parser.add_argument("--cpus_per_task", type=int, default=4, help="number of cpus cores per task (and associated memory)")
    parser.add_argument("--time", type=str, default="1:00:00", help="maximum time of script")
    parser.add_argument("--qos", type=str, default="qos_gpu-dev", help="maximum time of script")
    parser.add_argument("--hint", type=str, default="nomultithread", help="maximum time of script")
    parser.add_argument("--account", type=str, default="sxq@v100", help="maximum time of script")

    parser.add_argument("--script", type=str, default="train_classification.py", help="python script to launch")
    parser.add_argument("--add_resume", action="store_true", help="Add Resume to existing run in file")

def common_args_parser(parser) :
    parser.add_argument("--data_path", default="/datasets01/imagenet_full_size/061417/", type=str, help="dataset path")

    parser.add_argument("--model", default="fcn_resnet101", type=str, help="model name")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")

    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("-b", "--batch_size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)")

    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--wd","--weight_decay",default=1e-4,type=float,metavar="W",help="weight decay (default: 1e-4)",dest="weight_decay")

    parser.add_argument("--lr_warmup_decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr_warmup_method", default="linear", type=str, help="the warmup method (default: linear)")
    parser.add_argument("--lr_warmup_epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist_url", default="env://", type=str, help="url used to set up distributed training")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use_v2", action="store_true", help="Use V2 transforms")

    parser.add_argument(
        "--use_deterministic_algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )

    parser.add_argument("--print_freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output_dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")

    parser.add_argument(
        "--test_only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # Type of Logger
    parser.add_argument("--logger", default="txt", type=str, help="Type of logger (wandb or json)")

    # Name of the wandb logging and result folder
    parser.add_argument("--name", default="", type=str, help="Name of the experiment (used in WandB and Checkpoint)")

    # Resolution scaling specific arguments
    parser.add_argument("--first_conv_resize", default=0, type=int, help="if different than 0 rescale the input activations after the first convolution")

    # ViT Specific Args
    parser.add_argument("--patch_size",  default=16, type=int, help="ViT patch size(default to vit_b_16)")
    parser.add_argument("--num_layers",  default=12, type=int, help="ViT number of layers (default to vit_b_16)")
    parser.add_argument("--num_heads",  default=12, type=int, help="ViT number of Attention heads (default to vit_b_16)")
    parser.add_argument("--hidden_dim",  default=768, type=int, help="ViT hidden dimension (default to vit_b_16)")
    parser.add_argument("--mlp_dim",  default=3072, type=int, help="ViT hidden mlp dimension (default to vit_b_16)")
    parser.add_argument("--img_size",  default=224, type=int, help="ViT img size (default to vit_b_16)")

def classification_args_parser(parser):

    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument(
        "--norm_weight_decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias_weight_decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer_embedding_decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label_smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--mixup_alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix_alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr_scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr_step_size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr_gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr_min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument(
        "--cache_dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync_bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument("--auto_augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra_magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix_severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random_erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    parser.add_argument(
        "--model_ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model_ema_steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model_ema_decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val_resize_size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val_crop_size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train_crop_size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument("--clip_grad_norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra_sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra_reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )

    # Resnet Specific args
    parser.add_argument("--channels",default=None,  nargs="+", type=int, help="channels of ResNet")
    parser.add_argument("--depths", default=None, nargs="+", type=int, help="layers depths of ResNet")

def semantic_segmentation_args_parser(parser):

    parser.add_argument("--dataset", default="coco", type=str, help="dataset name")
    parser.add_argument("--aux_loss", action="store_true", help="auxiliary loss")

    parser.add_argument("--lr_warmup_start_factor", default=0.1, type=float, help="Linear learning rate scheduler start factor")

    parser.add_argument("--weights_backbone", default=None, type=str, help="the backbone weights enum name to load")
    # Added argument
    parser.add_argument("--power", default=0.9, type=float, help="polynomial scheduler power")
    parser.add_argument('--exclude_classes', nargs='+', type=int)

    parser.add_argument("--scale_low_size", default=400, type=int, help="lower value of first random scaling")
    parser.add_argument("--scale_high_size", default=1600, type=int, help="upper value of first random scaling")
    parser.add_argument("--random_crop_size", default=1024, type=int, help="the random crop size used for training (default: 1024)")

    parser.add_argument("--val_input_size", default=1024, type=int, help="val input size")
    parser.add_argument("--val_label_size", default=1024, type=int, help="val label size")

    parser.add_argument("--augmode", default=None, type=str, help="augmentation mode")
    # RegSeg parser arguments
    parser.add_argument("--regseg_name", default="custom_decoder4", type=str, help="regseg instance name(defines encoder and decoder used)")
    parser.add_argument('--regseg_channels', nargs='+', type=int, default=None, help="RegSeg channels list")
    parser.add_argument('--regseg_gw', type=int, default=0,  help="RegSeg gw")

def vit_segmentation_args_parser(parser) :
    parser.add_argument('--cityscapes_size', type=int, default=1024,  help="Size to evaluate and train Cityscapes")

def get_classification_argsparse(add_help=True) :
    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    common_args_parser(parser)
    classification_args_parser(parser)  
    return parser

def get_segmentation_argsparse(add_help=True) :
    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)

    common_args_parser(parser)
    semantic_segmentation_args_parser(parser)  
    return parser

def get_vitsegmentation_argsparse(add_help=True) :
    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)

    common_args_parser(parser)
    semantic_segmentation_args_parser(parser)
    vit_segmentation_args_parser(parser)
    return parser

def get_slurm_scheduler_argsparse(add_help=True) :
    parser = argparse.ArgumentParser(description="Slurm launcher, facilitates the deploiement of this repo training scripts to slurm environments", add_help=add_help)

    common_args_parser(parser)
    slurm_args_parser(parser)
    classification_args_parser(parser)
    semantic_segmentation_args_parser(parser)
    vit_segmentation_args_parser(parser)
    return parser

# Test
if __name__ == "__main__" :
    parser = argparse.ArgumentParser(prog='FooBoo')

    # Add both parsers to parser
    common_args_parser(parser)
    slurm_args_parser(parser)
    classification_args_parser(parser)
    semantic_segmentation_args_parser(parser)
    print(parser.parse_args())