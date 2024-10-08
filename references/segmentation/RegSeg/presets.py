from . import transforms as T
from .data_utils import *

def build_val_transform(val_input_size,val_label_size):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transforms=[]
    transforms.append(
        T.ValResize(val_input_size,val_label_size)
    )
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(
        mean,
        std
    ))
    return T.Compose(transforms)

def build_val_transform2(val_input_size,val_label_size):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transforms=[]
    transforms.append(
        T.CenterCrop(val_label_size)
    )
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(
        mean,
        std
    ))
    return T.Compose(transforms)
def build_train_transform2(train_min_size, train_max_size, train_crop_size, aug_mode,ignore_value):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    fill = tuple([int(v * 255) for v in mean])
    #ignore_value = 255
    edge_aware_crop=False
    resize_mode="uniform"
    transforms = []
    transforms.append(
        T.RandomResize(train_min_size, train_max_size, resize_mode)
    )
    if isinstance(train_crop_size,int):
        crop_h,crop_w=train_crop_size,train_crop_size
    else:
        crop_h,crop_w=train_crop_size
    transforms.append(
        T.RandomCrop2(crop_h,crop_w,edge_aware=edge_aware_crop)
    )
    transforms.append(T.RandomHorizontalFlip(0.5))
    if aug_mode == "baseline":
        pass
    elif aug_mode == "randaug":
        transforms.append(T.RandAugment(2, 0.2, "full",prob=1.0, fill=fill,
                                        ignore_value=ignore_value))
    elif aug_mode=="randaug_reduced":
        transforms.append(T.RandAugment(2, 0.2, "reduced",prob=1.0, fill=fill,
                                        ignore_value=ignore_value))
    elif aug_mode== "colour_jitter":
        transforms.append(T.ColorJitter(0.3, 0.3,0.3, 0,prob=1))
    elif aug_mode=="rotate":
        transforms.append(T.RandomRotation((-10,10), mean=fill, ignore_value=ignore_value,prob=1.0,expand=False))
    elif aug_mode=="noise":
        transforms.append(T.AddNoise(15,prob=1.0))
    elif aug_mode=="noise2":
        transforms.append(T.AddNoise2(10,prob=1.0))
    elif aug_mode=="noise3":
        transforms.append(T.AddNoise3(10,prob=1.0))
    elif aug_mode == "custom1":
        transforms.append(T.RandAugment(2, 0.2, "reduced",prob=1.0, fill=fill,
                                        ignore_value=ignore_value))
        transforms.append(T.AddNoise(10,prob=0.2))
    elif aug_mode == "custom2":
        transforms.append(T.RandAugment(2, 0.2, "reduced2",prob=1.0, fill=fill,
                                        ignore_value=ignore_value))
        transforms.append(T.AddNoise(10,prob=0.1))
    elif aug_mode=="custom3":
        transforms.append(T.ColorJitter(0.3, 0.4,0.5, 0,prob=1))
    else:
        raise NotImplementedError()
    transforms.append(T.RandomPad(crop_h,crop_w,fill,ignore_value,random_pad=True))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(
        mean,
        std
    ))
    return T.Compose(transforms)