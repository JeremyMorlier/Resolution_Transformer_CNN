import os, sys
import numpy as np
import argparse
import random
import cv2
import json
import time
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn 

from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler

from torchvision_references.references.distillation_sam.common import parse_option, get_optimizer, get_scheduler, customized_mseloss, calculate_iou, find_center
from torchvision_references.datasets.segment_anything_dataset import transform, sa1b_dataset, normal_distribution_dataset
from torchvision_references.references.distillation_sam.adaptor import Resnet50_Adaptor
from torchvision_references.models import get_model

# Libraries for ADE20k evaluation
import pickle as pkl
import torchvision_references.references.distillation_sam.utils_ade20k as utils_ade20k
import glob
from mobile_sam import SamPredictor

def evaluate_ADE20K(args, model) :

    # Paths
    ade20k_path = args.ade_dataset
    index_file = 'ADE20K_2021_17_01/index_ade20k.pkl'

    device = torch.device('cuda:0')

    # Load index file
    with open(os.path.join(ade20k_path, index_file), 'rb') as f:
        index_ade20k = pkl.load(f)

    # Get original model and retrieve decoder
    sam = get_model("sam_vit_h", checkpoint = args.sam_checkpoint)
    sam.image_encoder = model
    sam.to(device=device)

    iou_liste = []
    nb_crops = 0
    n = len(index_ade20k['folder'])
    for i in range(n) : 
        full_file_name = os.path.join(index_ade20k['folder'][i], index_ade20k['filename'][i])
        folder_name = os.path.join(ade20k_path,full_file_name.replace(".jpg", ''))
        folder_files = glob.glob(f"{folder_name}/*")
        print(os.path.join(ade20k_path, full_file_name))
        info = utils_ade20k.loadAde20K(os.path.join(ade20k_path, full_file_name))
        image = cv2.imread(info['img_name'])[:,:,::-1]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        predictor = SamPredictor(sam)
        predictor.set_image(image)
        input_label = np.array([1])
        n2 = len(folder_files)
        
        # Gather all instances in the image and compute iou on each
        for j, image_test in enumerate(folder_files) : 
            aux = cv2.imread(image_test)[:,:,0]
            label = (aux != 0)*1
            center_point, valid_mask = find_center(aux)

            if valid_mask : 
                mask, _, _ = predictor.predict(
                point_coords=center_point,
                point_labels=input_label,
                multimask_output=False,
                )
                iou = calculate_iou(mask*1,label*1)
                iou_liste.append(iou)
                sys.stdout.write(f'\r {time.strftime("%H:%M:%S", time.gmtime())} : {i} / {n}, {j} / {n2}   IoU: {iou}')
                nb_crops += 1
        sys.stdout.write(f'\r {time.strftime("%H:%M:%S", time.gmtime())} : {i} / {n}, {j} / {n2}   IoU: {iou} mIoU: {np.mean(iou_liste)}')
                
    mean = np.mean(iou_liste)
    print(mean)
    return mean

def evaluate_against_sam(args, model, val_loader) :
    device = torch.device('cuda:0')

    # Get original model
    sam = get_model("sam_vit_h", checkpoint = args.sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # Get original model and change encoder
    sam2 = get_model("sam_vit_h", checkpoint = args.sam_checkpoint)
    sam2.image_encoder = model
    sam2.to(device=device)
    predictor2 = SamPredictor(sam2)

    iou_liste = []
    n = len(val_loader)
    for batch_idx, (image, target, mask_path) in enumerate(val_loader) :
        # Retrieve input point and label for SAM Dataset
        mask = open(mask_path[0])
        label = json.load(mask)
        index = random.randint(0,len(label['annotations'])-1)
        input_point = np.array(label['annotations'][index]['point_coords'])
        input_label = np.array([1])
        
        # Image is expected in numpy format
        image = image[0].numpy()
        # Set the image for both predictors
        predictor.set_image(image)
        predictor2.set_image(image)

        # Predict masks
        mask, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        mask_neighbor, _, _ = predictor2.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        iou = calculate_iou(mask*1,mask_neighbor*1)
        sys.stdout.write(f'\r {time.strftime("%H:%M:%S", time.gmtime())} : {batch_idx} / {n}   IoU: {iou}')
        iou_liste.append(iou)

    mean = np.mean(iou_liste)
    print("\n ", mean)
    return mean

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

def get_param_model(args) :
    if "sam" in args.model :
        if "mobile" in args.model :
            model= get_model(args.model)
        else :
            model= get_model(args.model).image_encoder
    if "resnet" in args.model :
        resnet50 = get_model(args.model, num_classes=1000)
        adaptor = Resnet50_Adaptor(2048, 256)
        model = nn.Sequential(resnet50.encoder, adaptor)
    
    return model

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
    train_dirs = args.train_dirs

    train_dataset = sa1b_dataset(args.dataset_path, train_dirs, transform, feat_root=args.root_feat)

    # training sampler
    train_sampler = DistributedSampler(train_dataset)

    # data loader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=(train_sampler is None), num_workers=args.num_workers, sampler=train_sampler, drop_last=True)

    # model
    model = get_param_model(args)
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
                    wandb.log({"mse_loss":loss.item()})
                    # writer.add_scalar("mse_loss", loss.item(), total_iters)
                
                # save model
                if total_iters % args.save_iters == 0:
                    save_path = os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_" + str(total_iters) + ".pth")
                    print("save model to {}".format(save_path))
                    torch.save(model.state_dict(), save_path)

        dist.barrier()
        scheduler.step()

    if local_rank == 0:
        # save final model
        torch.save(model.module.state_dict(), os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_final.pth"))

        # Evaluate
        model = get_param_model(args)
        
        model.load_state_dict(torch.load(os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_final.pth")))
        model.to(device)
        model.eval()

        if args.val_dirs != None :
            print("Evaluate against ViT_H", time.strftime("%d %b %Y %H:%M:%S", time.gmtime()))
            val_dirs = args.val_dirs
            val_dataset = sa1b_dataset(args.dataset_path, val_dirs)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
            result = evaluate_against_sam(args, model, val_loader)
            print("Evaluation finished: ", time.strftime("%d %b %Y %H:%M:%S", time.gmtime()), " mIoU: ", result)
            wandb.log({"ViT_H_mIoU":result})

        if args.ade_dataset != None :
            print("Evaluate on ADE20k", time.strftime("%d %b %Y %H:%M:%S", time.gmtime()))
            result = evaluate_ADE20K(args, model)
            print("Evaluation on ADE20k finished: ", time.strftime("%d %b %Y %H:%M:%S", time.gmtime()), " mIoU: ", result) 
            wandb.log({"ADE20K_mIoU":result})
        
        training_time = time.time() - init_time
        print("Training finished ! Training Time: ", training_time)
        wandb.log({"training_time":training_time})
    
if __name__ == "__main__":
    args = parse_option()
    name = str(args.model) + "_" + str(args.learning_rate) + str(args.batch_size) + "_" + str(args.epochs) + "_" + str(args.optim) +  "_" + str(args.learning_rate) +  "_" + str(args.weight_decay) + "_" + str(args.momentum)
    wandb.init(
    # set the wandb project where this run will be logged
    project="Data_Distillation",
    name=name,
    tags=["SAM"],
    
    # track hyperparameters and run metadata
    config=args
    )
    main(args)
    wandb.finish()
