import os
import sys
import signal
import stat

import torch
import torch.distributed as dist

def create_dir(dir) :
    if not os.path.isdir(dir) :
        os.mkdir(dir)
        os.chmod(dir, stat.S_IRWXU | stat.S_IRWXO)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        print("Using slurm")
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
        args.slurm_jobid = int(os.environ["SLURM_JOB_ID"])
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True
    print(args.gpu)
    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def create_dir(dir) :
    if is_main_process() :
        if not os.path.isdir(dir) :
            os.mkdir(dir)
            os.chmod(dir, stat.S_IRWXU | stat.S_IRWXO)

def get_name(args) :

    name_channel = ""

    # Classification 
    if hasattr(args, "channels") :
        if args.channels != None :
            for element in args.channels :
                name_channel += "_" + str(element)
    elif hasattr(args, "regseg_channels") :
        if args.regseg_channels != None :
            for element in args.regseg_channels :
                name_channel += "_" + str(element)
    else :
        name_channel = "_None"
    
    if "resnet" in args.model :
        name = args.model + "_" + str(args.train_crop_size) + "_" + str(args.val_crop_size)  + "_" + str(args.val_resize_size) + "_" + str(args.first_conv_resize) + name_channel
    elif "vit" in args.model:
        name = args.model + "_" + str(args.patch_size) + "_" + str(args.num_layers) + "_" + str(args.num_heads) + "_" + str(args.hidden_dim) + "_" + str(args.mlp_dim) + "_" + str(args.img_size)
    elif "regseg" in args.model :
        name = args.model + "_" + str(args.scale_low_size) + "_" + str(args.scale_high_size)  + "_" + str(args.random_crop_size) + "_" + str(args.first_conv_resize) + "_" + str(args.regseg_gw) + "_" + name_channel
    
    return name

def sig_handler(signum, frame):
    prod_id = int(os.environ['SLURM_PROCID'])
    if prod_id == 0:
        os.system('scontrol requeue ' + os.environ['SLURM_JOB_ID'])
    sys.exit(-1)

def init_signal_handler():
    """
    Handle signals sent by SLURM for time limit.
    """
    signal.signal(signal.SIGUSR1, sig_handler)