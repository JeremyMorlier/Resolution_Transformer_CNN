import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as transforms
from torchvision.transforms import v2

from segment_anything.utils.transforms import ResizeLongestSide

from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
from torch.utils.data import Sampler


class Distribution_dataset(Dataset):
    def __init__(self, distribution, size, transform=None):
        self.transform = transform
        self.distribution = distribution
        self.size = size

    def __len__(self) :
        return 10000000
    
    def __getitem__(self, idx):
        image = self.distribution.sample(self.size)
        image = image.permute(2, 0, 1)   

        if self.transform :
            image = self.transform(image)
        return image

class White_dataset(Dataset):
    def __init__(self, size, transform=None):
        self.transform = transform
        self.size = size

    def __len__(self) :
         return 10000000
    
    def __getitem__(self, idx):
        image = torch.ones(self.size)
        
        if self.transform :
            image = self.transform(image)
        return image