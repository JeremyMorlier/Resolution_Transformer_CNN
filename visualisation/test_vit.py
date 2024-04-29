import datetime
import os
import time
import warnings

import torch
import torch.utils.data
import torchvision
import torchvision.transforms

from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode

import torchvision_references.references.classification.presets as presets
from torchvision_references.references.classification.transforms import get_mixup_cutmix
import torchvision_references.references.classification.utils as utils
from torchvision_references.references.classification.sampler import RASampler
from torchvision_references.models import get_model


if __name__ == "__main__" :
    model_name = "vit_b_16"
    model = get_model(model_name)

    input_image = torch.rand(1, 3, 224, 224)

    print(model(input_image).size())