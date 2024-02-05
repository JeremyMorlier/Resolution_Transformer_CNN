
import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision.transforms import functional as F

from datasets.ADE20k import ADE20k


class ToTensor(object):
    def __call__(self, image, target):
        image, target = image.copy(), target.copy()
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target

if __name__ == "__main__" :
    import yaml

    def load_yaml(config_filename) :
        with open(config_filename) as file:
            config=yaml.full_load(file)
        return config
    
    config_filename = "test_config"
    config = load_yaml("configs/" + config_filename + ".yaml")
    root = config["root"]
    split = config["split"]

    transforms = ToTensor()
    dataset = ADE20k(root, split, transforms)
    print(len(dataset))
    image, target = dataset[0]

    print(image.size(), image.size())