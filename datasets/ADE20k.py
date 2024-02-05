import os
from collections import namedtuple
from typing import Any, Callable, List, Optional, Union, Tuple
import torch.utils.data as data
from PIL import Image

import third_party.ADE20k.utils.utils_ade20k as ade20k

class ADE20k(data.Dataset) :
    def __init__(self, root, split, transforms) :
        self.root = root
        self.split = split
        self.transforms = transforms
    def __get_item__(self, index) :


        if self.transforms is not None :
            image, target = self.transforms(image, target)
        return image, target
    def __len__(self) :
        return len(self.imgs)
    