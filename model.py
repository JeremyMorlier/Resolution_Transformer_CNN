import segment_anything as sam
import mobile_sam as msam

import torch
import torch.nn as nn
class SAM_model(nn.Module) :
    def __init__(self, name, model_type, pretrained="") :
        super().__init__()

        self.name = name
        self.model_type = model_type

        if self.name == "vit_t" :
            self.model = msam.sam_model_registry[name](checkpoint=pretrained)
        else :
            self.model = sam.sam_model_registry[name](checkpoint=pretrained)
        
    def forward(self, x) :
        return self.model(x) 