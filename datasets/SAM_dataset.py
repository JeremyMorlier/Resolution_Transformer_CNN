import os
import json

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torch.nn import functional as F

from segment_anything.utils.transforms import ResizeLongestSide

from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
from torch.utils.data import Sampler

mean=[123.675, 116.28, 103.53]
std=[58.395, 57.12, 57.375]

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device) 
    return image.permute(2, 0, 1).contiguous()

def preprocess(x: torch.Tensor, size) -> torch.Tensor:
    """pad to a square input."""
    # Pad
    h, w = x.shape[-2:]
    padh = size - h
    padw = size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x
    
def SAM_norm() :
    return  v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
    ])

class Segment_Anything_Dataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None, SAM_transform=ResizeLongestSide, sam_size=1024):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.sam_size = sam_size

        self.SAM_transform  = SAM_transform
        if self.SAM_transform != None :
            self.SAM_transform = SAM_transform(sam_size)

        self.indices = [int(name[3:-4]) for name in os.listdir(root) if (os.path.isfile(os.path.join(root, name)) and name[-3:] == "jpg")]

    def __len__(self) :
         return len(self.indices)
    
    def __getitem__(self, idx):
        index = str(self.indices[idx])
        file_name = "sa_" + index
        img_name = file_name + ".jpg"
        masks_name = file_name + ".json"

        img_path = os.path.join(self.root, img_name)
        masks_path = os.path.join(self.root, masks_name)
        if os.path.exists(img_path) :
            image = read_image(img_path)
            #masks = json.load(open(masks_path, 'r'))
            
            if self.SAM_transform != None:
                size = image.shape[-2:]
                image = SAM_norm()(image)
                image = image.unsqueeze(0)    
                image = self.SAM_transform.apply_image_torch(image)
                image = preprocess(image, self.sam_size)
                image = image.squeeze(0)
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)

            return image
        else :
            print("error", idx)
        
        return None

if __name__ == "__main__" :
    # transform_test = SAM_transform(1024)

    root = "/nasbrain/datasets/sam_dataset/images/"
    folder_name = ""

    from segment_anything import SamPredictor, sam_model_registry

    # sam = sam_model_registry["vit_b"](checkpoint=root + "checkpoints/sam_vit_b_01ec64.pth")
    sam = sam_model_registry["vit_b"]()

    dataset = Segment_Anything_Dataset(root + folder_name, None, None, ResizeLongestSide, sam.image_encoder.img_size)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    print(len(dataset))
    images = next(iter(dataloader))
    print(images.size())