
import argparse

import torch
import torch.optim as optim

from mobile_sam.modeling import TinyViT

# Contains common functions betwwen train.py and train_parallel.py
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # dataset paths
    parser.add_argument('--dataset_path', type=str, default="/dataset/vyueyu/sa-1b", help='root path of dataset')

    # training epochs, batch size and so on
    parser.add_argument('--epochs', type=int, default=8, help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')

    # multi gpu settings
    parser.add_argument("--local_rank", type=int, default=-1)

    # cuda settings
    # parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--seed', type=int, default=1234, help='seed')
    parser.add_argument('--deterministic', type=bool, default=True, help='deterministic')
    parser.add_argument('--benchmark', type=bool, default=False, help='benchmark')

    # learning process settings
    parser.add_argument('--optim', type=str, default='sgd', choices=['adam', 'sgd', 'adamw'])
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # print and evaluate frequency during training
    parser.add_argument('--print_iters', type=int, default=200, help='print loss iterations')
    parser.add_argument('--eval_nums', type=int, default=200, help='evaluation numbers')
    parser.add_argument('--eval_iters', type=int, default=500, help='evaluation iterations')

    # file and folder paths
    parser.add_argument('--root_path', type=str, default="/users/local/j20morli/MobileSAM-pytorch/MobileSAM/", help='root path')
    parser.add_argument('--root_feat', type=str, default="/users/local/j20morli/MobileSAM-pytorch/MobileSAM/", help='root features path')
    parser.add_argument('--work_dir', type=str, default="work_dir", help='work directory')
    parser.add_argument('--save_dir', type=str, default="ckpt", help='save directory')
    parser.add_argument('--log_dir', type=str, default="log", help='save directory')
    parser.add_argument('--save_iters', type=int, default=50000, help='save iterations')

    # SAM directories used for training
    parser.add_argument('--train_dirs', nargs='+', type=str)

    args = parser.parse_args()
    return args

def build_model():
    model = TinyViT(img_size=1024, in_chans=3, num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8
            )
    
    ## load pretrained TinyViT weights, please download from https://github.com/wkcn/TinyViT?tab=readme-ov-file
    # pretrained_weights = torch.load("path_to_pth")["model"]
    # del pretrained_weights["head.weight"]
    # del pretrained_weights["head.bias"]
    # model.load_state_dict(pretrained_weights, strict=False)
    
    return model

def get_optimizer(args, model):
    if args.optim == 'adam':
        return optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        return optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        return optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(args.optim)

def get_scheduler(args, optimizer):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.5)
    
def customized_mseloss(pred_feats, target_feats):
    # return (0.5 * (pred_feats - target_feats) ** 2).sum(1).mean()
    return ((pred_feats - target_feats) ** 2).sum(1).mean().sqrt()