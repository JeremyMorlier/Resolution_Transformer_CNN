import os, sys
import numpy as np
import random
import cv2
import json
import time
import datetime
import warnings

import torch
import torch.nn as nn

from logger import Logger
from args import get_sam_argsparse
import utils

from datasets.segment_anything_dataset import transform, sa1b_dataset, normal_distribution_dataset
from references.distillation_sam.common import get_optimizer, get_scheduler, customized_mseloss, calculate_iou, find_center
from references.distillation_sam.adaptor import Resnet50_Adaptor
import references.distillation_sam.utils_ade20k as utils_ade20k
from models import get_model

import pickle as pkl
import glob
from mobile_sam import SamPredictor

def evaluate_ADE20K(args, backbone, device) :

    # Paths
    ade20k_path = args.ade_dataset
    index_file = 'ADE20K_2021_17_01/index_ade20k.pkl'

    # Load index file
    with open(os.path.join(ade20k_path, index_file), 'rb') as f:
        index_ade20k = pkl.load(f)

    # Get original model and retrieve decoder
    sam = get_model("sam_vit_h", checkpoint = args.sam_checkpoint)
    sam.to(device=device)
    sam.image_encoder = backbone
    if args.distributed :
        sam = torch.nn.SyncBatchNorm.convert_sync_batchnorm(sam)
        sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=True)

    iou_liste = []
    nb_crops = 0
    n = len(index_ade20k['folder'])
    for i in range(n) : 
        full_file_name = os.path.join(index_ade20k['folder'][i], index_ade20k['filename'][i])
        folder_name = os.path.join(ade20k_path,full_file_name.replace(".jpg", ''))
        folder_files = glob.glob(f"{folder_name}/*")
        
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

def evaluate_against_sam(args, model, val_loader, device) :

    # Get original model
    sam = get_model("sam_vit_h", checkpoint = args.sam_checkpoint)
    sam.to(device=device)
    if args.distributed :
        sam = torch.nn.SyncBatchNorm.convert_sync_batchnorm(sam)
        sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=True)
    predictor = SamPredictor(sam)

    # Get original model and change encoder
    sam2 = get_model("sam_vit_h", checkpoint = args.sam_checkpoint)
    sam2.image_encoder = model
    sam2.to(device=device)
    if args.distributed :
        sam2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(sam2)
        sam2 = torch.nn.parallel.DistributedDataParallel(sam2, device_ids=[args.gpu], find_unused_parameters=True)
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

def train_epoch(model, optimizer, train_loader, device, epoch, total_iters) :

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
        # loss = reduce_mean(loss, utils.get_world_size())
        
        # if is master process
        if utils.is_main_process() :
            #print training info
            if (batch_idx + 1) % args.print_iters == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tMSE Loss: {:.6f}'.format(
                    epoch, batch_idx * len(imgs) * utils.get_world_size(), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
            
            # save model
            if total_iters % args.save_iters == 0:
                save_path = os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_" + str(total_iters) + ".pth")
                print("save model to {}".format(save_path))
                torch.save(model.state_dict(), save_path)

    return loss.item()

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

def main(args) :
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # Signal Handler to automatically relaunch slurm job
    utils.init_signal_handler()

    # Folder Setup
    utils.create_dir(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, args.name)
    utils.create_dir(args.output_dir)

    # log only on main process
    if utils.is_main_process() :
        # similar API to wandb except mode and log_dir
        logger = Logger(project_name="distillation_sam",
                run_name=args.name,
                tags=["test1"],
                resume=True,
                args=args,
                mode=args.logger,
                log_dir=args.output_dir)
        
    # Dataset
    train_dirs = args.train_dirs

    train_dataset = sa1b_dataset(args.dataset_path, train_dirs, transform, feat_root=args.root_feat)
    if args.distributed :
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else :
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, sampler=train_sampler, drop_last=True)

    # model
    model = get_param_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed :
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # optimizer and scheduler
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)

    if utils.is_main_process() == 0:
        init_time = time.time()

    total_iters = 0
    for epoch in range(1, args.epochs + 1):
        # new epoch
        if utils.is_main_process() == 0:
            print("------start epoch {}------".format(epoch), time.strftime("%d %b %Y %H:%M:%S", time.gmtime()))
            logger.log({"epoch": epoch})
        if args.distributed :
            train_sampler.set_epoch(epoch)

        result = train_epoch(model, optimizer, train_loader, device, epoch, total_iters)
        logger.log({"loss": result})
        scheduler.step()
    
    if utils.is_main_process() :
        torch.save(model_without_ddp.state_dict(), os.path.join(args.output_dir, "iter_final.pth"))

    # Evaluate
    model.eval()
    if args.val_dirs != None :
        print("Evaluate against ViT_H", time.strftime("%d %b %Y %H:%M:%S", time.gmtime()))

        # Dataset
        val_dirs = args.val_dirs
        val_dataset = sa1b_dataset(args.dataset_path, val_dirs)
        if args.distributed :
            train_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        else :
            train_sampler = torch.utils.data.RandomSampler(val_dataset)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, sampler=train_sampler, num_workers=args.num_workers)

        result = evaluate_against_sam(args, model_without_ddp, val_loader, device)
        print("Evaluation finished: ", time.strftime("%d %b %Y %H:%M:%S", time.gmtime()), " mIoU: ", result)
        logger.log({"ViT_H_mIoU":result})

    if args.ade_dataset != None :
        print("Evaluate on ADE20k", time.strftime("%d %b %Y %H:%M:%S", time.gmtime()))
        result = evaluate_ADE20K(args, model_without_ddp, device)
        print("Evaluation on ADE20k finished: ", time.strftime("%d %b %Y %H:%M:%S", time.gmtime()), " mIoU: ", result) 
        logger.log({"ADE20K_mIoU":result})
    
    training_time = time.time() - init_time
    print("Training finished ! Training Time: ", training_time)
    logger.log({"training_time":training_time})

    if utils.is_main_process() :
        logger.finish()
        
if __name__ == "__main__" :
    args, unknown_args = get_sam_argsparse().parse_known_args()
    args.name = "test"
    main(args)