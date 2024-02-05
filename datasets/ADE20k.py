import os
from collections import namedtuple
from typing import Any, Callable, List, Optional, Union, Tuple
import torch.utils.data as data
from PIL import Image
import pickle as pkl
import glob
import cv2
import third_party.ADE20K_utils.utils.utils_ade20k as ade20k

class ADE20k(data.Dataset) :
    def __init__(self, root, split, transforms) :
        self.root = root
        self.split = split
        self.transforms = transforms

        print(root)
        index_file = 'index_ade20k.pkl'
        # Load pickle file
        if(os.path.exists(os.path.join(root, "ADE20K_2021_17_01",  index_file))) :
            
            with open('{}/{}/{}'.format(root, "ADE20K_2021_17_01", index_file), 'rb') as f:
                self.index_ade20k = pkl.load(f)
        else :
            print("Error")
            return None
        
        print(self.index_ade20k.keys())
        # print(index_ade20k["folder"])
        self.nfiles = len(self.index_ade20k['filename'])
    def __getitem__(self, index) :
        full_file_name = '{}/{}'.format(self.index_ade20k['folder'][index], self.index_ade20k['filename'][index])
        print(self.index_ade20k['folder'][index])
        # folder_name = '{}/{}'.format(self.root,full_file_name.replace(".jpg", ''))
        # print(folder_name)
        # folder_files = glob.glob(f"{folder_name}/*")

        file_name = self.index_ade20k['filename'][index]
        info = ade20k.loadAde20K('{}/{}'.format(self.root, full_file_name))
        image = cv2.imread(info['img_name'])[:,:,::-1]
        target = info["class_mask"]

        if self.transforms is not None :
            image, target = self.transforms(image, target)
        return image, target
    def __len__(self) :
        return self.nfiles
    