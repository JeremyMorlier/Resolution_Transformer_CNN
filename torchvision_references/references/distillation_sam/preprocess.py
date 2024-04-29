import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

import torch
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def parse_option():
    parser = argparse.ArgumentParser('argument for pre-processing')

    parser.add_argument('--dataset_path', type=str, default="/users2/local/j20morli_sam_dataset/images/", help='root path of dataset')
    parser.add_argument('--features_path', type=str, default="/users2/local/j20morli_sam_dataset/SAM_vit_h_features/", help='root path of dataset')

    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--sam_type', type=str, default="vit_h")
    parser.add_argument('--sam_ckpt', type=str, default="/users2/local/j20morli_sam_dataset/checkpoints/sam_vit_h_4b8939.pth")

    # SAM directories used for training
    parser.add_argument('--train_dirs', nargs='+', type=str)

    args = parser.parse_args()
    return args


def extract_features(dataset_path, features_path, dataset_dir, predictor) :

    test_image_dir = os.path.join(dataset_path, dataset_dir)
    print(test_image_dir)

    test_image_paths = [img_name for img_name in os.listdir(test_image_dir)]
    
    if not os.path.isdir(os.path.join(features_path, dataset_dir)) :
        os.mkdir(os.path.join(features_path, dataset_dir))

    for i, test_image_path in enumerate(tqdm(test_image_paths)):
        if ".jpg" in test_image_path:

            results_path = os.path.join(features_path, dataset_dir, test_image_path).replace(".jpg", ".npy")
            
            # Check if results is already computed 
            # could fail in case of process ending during saving
            if not os.path.exists(results_path) :
                # Load and process image
                test_image = cv2.imread(os.path.join(test_image_dir, test_image_path))
                test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

                # Compute model results
                predictor.set_image(test_image)
                feature = predictor.features

                # Save
                np.save(os.path.join(features_path, dataset_dir, test_image_path).replace(".jpg", ".npy"), feature.cpu().numpy())# .astype(np.float16))

if __name__ == '__main__':

    args = parse_option()

    device = args.device

    sam = sam_model_registry[args.sam_type](checkpoint=args.sam_ckpt)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    dataset_dirs = ["sa_000022", "sa_000024", "sa_000070", "sa_000135", "sa_000137", "sa_000138", "sa_000477", "sa_000259", "sa_000977"]
    dataset_dirs = args.train_dirs
    for dataset_dir in dataset_dirs :
        print(dataset_dir)
        extract_features(args.dataset_path, args.features_path, dataset_dir, predictor)
        print(dataset_dir, " : preprocessed")

    print("Finished !")
