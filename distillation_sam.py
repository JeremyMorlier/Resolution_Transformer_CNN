import os
import numpy as np
import argparse
import random
import cv2
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn 

from torchvision_references.datasets.segment_anything_dataset import transform, sa1b_dataset, normal_distribution_dataset

from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler

from torchvision_references.references.distillation_sam.common import parse_option, build_model, get_optimizer, get_scheduler, customized_mseloss
            
def test(args, model, test_loader, local_rank):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for idx, (imgs, target_feats, mask_paths) in enumerate(test_loader):
            imgs, target_feats = imgs.cuda(local_rank), target_feats.cuda(local_rank)
            pred_feats = model.module(imgs)
            test_loss += customized_mseloss(pred_feats, target_feats).item()

    return test_loss / len(test_loader)

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt
    
def main(args):

    local_rank = int(os.environ["LOCAL_RANK"])
    # multi gpu settings
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.distributed.init_process_group(backend='nccl')

    # file folder creating
    if local_rank == 0:
        if not os.path.exists(os.path.join(args.root_path, args.work_dir, args.save_dir)):
            os.makedirs(os.path.join(args.root_path, args.work_dir, args.save_dir))
        
        if not os.path.exists(os.path.join(args.root_path, args.work_dir, args.log_dir)):
            os.makedirs(os.path.join(args.root_path, args.work_dir, args.log_dir))

    # seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = args.deterministic
        cudnn.benchmark = args.benchmark
    
    # dataset
    #train_dirs = ["sa_" + str(i).zfill(6) for i in range(20)]*
    train_dirs = args.train_dirs
    #train_dirs = ["sa_000022"]
    val_dirs = ['sa_000022']
    train_dataset = sa1b_dataset(args.dataset_path, args.root_feat, train_dirs, transform)
    #train_dataset = normal_distribution_dataset(train_dirs, transform)
    val_dataset = sa1b_dataset(args.dataset_path, args.root_feat, val_dirs, transform, args.eval_nums)
    # training sampler
    train_sampler = DistributedSampler(train_dataset)
    # data loader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=(train_sampler is None), num_workers=args.num_workers, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # model
    model = build_model()
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # optimizer and scheduler
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)

    total_iters = 0

    if local_rank == 0:
        init_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # new epoch
        if local_rank == 0:
            print("------start epoch {}------".format(epoch), time.strftime("%d %b %Y %H:%M:%S", time.gmtime()))
        train_sampler.set_epoch(epoch)

        # training
        model.train()
        for batch_idx, (imgs, target_feats, mask_paths) in enumerate(train_loader):
            total_iters += 1
            
            imgs, target_feats = imgs.to(device), target_feats.to(device)
            optimizer.zero_grad()
            pred_feats = model(imgs)
            loss = customized_mseloss(pred_feats, target_feats)
            loss.backward()
            optimizer.step()
            loss = reduce_mean(loss, dist.get_world_size())
            
            # if is master process
            if local_rank == 0:
                #print training info
                if (batch_idx + 1) % args.print_iters == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tMSE Loss: {:.6f}'.format(
                        epoch, batch_idx * len(imgs) * dist.get_world_size(), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), loss.item()))
                    # writer.add_scalar("mse_loss", loss.item(), total_iters)
                
                # save model
                if total_iters % args.save_iters == 0:
                    save_path = os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_" + str(total_iters) + ".pth")
                    print("save model to {}".format(save_path))
                    torch.save(model.state_dict(), save_path)

                # evaluation
                '''
                if total_iters % args.eval_iters == 0:
                    test_loss = test(args, model, val_loader)
                    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
                    writer.add_scalar("eval_mse_loss", test_loss, total_iters)
                '''

        dist.barrier()
        scheduler.step()

    # save final model
    if local_rank == 0:
        torch.save(model.state_dict(), os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_final.pth"))
        #writer.close()
        
        training_time = time.time() - init_time
        print("Training finished ! Training Time: ", training_time)
    
if __name__ == "__main__":
    args = parse_option()
    main(args)
