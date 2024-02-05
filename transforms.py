
import numpy as np

import torch
from torchvision.transforms import functional as F

class ToTensor(object):
    def __call__(self, image, target):
        image, target = image.copy(), target.copy()
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target
