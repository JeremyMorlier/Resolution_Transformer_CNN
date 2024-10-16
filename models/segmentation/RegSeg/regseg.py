from functools import partial
from typing import Any, Optional, Sequence
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from torchvision.transforms._presets import SemanticSegmentation
from ..._api import register_model, Weights, WeightsEnum
#from .._meta import _VOC_CATEGORIES
from ..._utils import _ovewrite_value_param, handle_legacy_interface, IntermediateLayerGetter

from .blocks import *
from .competitor_blocks import BiseNetDecoder,SFNetDecoder,FaPNDecoder

__all__ = [
    "RegSeg_model",
    "regseg_custom",
]



class RegSeg_model(nn.Module):
    # exp48_decoder26 is what we call RegSeg in our paper
    # exp53_decoder29 is a larger version of exp48_decoder26
    # all the other models are for ablation studies
    def __init__(self, regseg_name, num_classes, pretrained="", ablate_decoder=False,change_num_classes=False, gw=16, channels=[32, 48, 128, 256, 320], first_conv_resize=0):
        super().__init__()
        in_channels = channels[0]
        # TODO: change for config parameter
        #print(first_conv_resize, gw, channels)
        self.stem=ConvBnAct(3,in_channels,3,2,1, first_conv_resize=first_conv_resize)
        #print(regseg_name)
        body_name, decoder_name=regseg_name.split("_")
        if "exp30" == body_name:
            self.body=RegSegBody(5*[[1,4]]+8*[[1,10]], gw, channels)
        elif "exp43"==body_name:
            self.body=RegSegBody([[1],[1,2],[1,4],[1,6],[1,8],[1,10]]+7*[[1,12]], gw, channels)
        elif "exp46"==body_name:
            self.body=RegSegBody([[1],[1,2],[1,4],[1,6],[1,8]]+8*[[1,10]], gw, channels)
        elif "exp47"==body_name:
            self.body=RegSegBody([[1],[1,2],[1,4],[1,6],[1,8],[1,10],[1,12]]+6*[[1,14]], gw, channels)
        elif "exp48"==body_name:
            self.body=RegSegBody([[1],[1,2]]+4*[[1,4]]+7*[[1,14]], gw, channels)
        elif "exp49"==body_name:
            self.body=RegSegBody([[1],[1,2]]+6*[[1,4]]+5*[[1,6,12,18]], gw, channels)
        elif "exp50"==body_name:
            self.body=RegSegBody([[1],[1,2],[1,4],[1,6],[1,8],[1,10]]+7*[[1,3,6,12]], gw, channels)
        elif "exp51"==body_name:
            self.body=RegSegBody([[1],[1,2],[1,4],[1,6],[1,8],[1,10]]+7*[[1,4,8,12]], gw, channels)
        elif "exp52"==body_name:
            self.body=RegSegBody([[1],[1,2],[1,4]]+10*[[1,6]], gw, channels)
        elif "exp53"==body_name:
            self.body=RegSegBody2([[1],[1,2]]+4*[[1,4]]+7*[[1,14]])
        # small RegSeg body
        elif "custom"==body_name :
            self.body=RegSegBody([[1],[2]]+4*[[4]]+7*[[14]], gw, channels)
        elif "regnety600mf"==body_name:
            self.body=RegNetY600MF()
        else:
            raise NotImplementedError()
        if "decoder4" ==decoder_name:
            self.decoder=Exp2_Decoder4(num_classes,self.body.channels())
        elif "decoder10" ==decoder_name:
            self.decoder=Exp2_Decoder10(num_classes,self.body.channels())
        elif "decoder12" ==decoder_name:
            self.decoder=Exp2_Decoder12(num_classes,self.body.channels())
        elif "decoder14"==decoder_name:
            self.decoder=Exp2_Decoder14(num_classes,self.body.channels())
        elif "decoder26"==decoder_name:
            self.decoder=Exp2_Decoder26(num_classes,self.body.channels())
        elif "decoder29"==decoder_name:
            self.decoder=Exp2_Decoder29(num_classes,self.body.channels())
        elif "BisenetDecoder"==decoder_name:
            self.decoder=BiseNetDecoder(num_classes,self.body.channels())
        elif "SFNetDecoder"==decoder_name:
            self.decoder=SFNetDecoder(num_classes,self.body.channels())
        elif "FaPNDecoder"==decoder_name:
            self.decoder=FaPNDecoder(num_classes,self.body.channels())
        else:
            raise NotImplementedError()
        #print(pretrained, ablate_decoder)
        if pretrained != "" and not ablate_decoder:
            dic = torch.load(pretrained, map_location='cpu')
            if type(dic)==dict and "model" in dic:
                dic=dic['model']
            if change_num_classes:
                current_model=self.state_dict()
                new_state_dict={}
                print("change_num_classes: True")
                for k in current_model:
                    if dic[k].size()==current_model[k].size():
                        new_state_dict[k]=dic[k]
                    else:
                        print(k)
                        new_state_dict[k]=current_model[k]
                self.load_state_dict(new_state_dict,strict=True)
            else:
                self.load_state_dict(dic,strict=True)


    def forward(self,x, shape=None):
        output_shape=x.shape[-2:]
        if shape :
            output_shape = shape
        # Encoder starts here
        x=self.stem(x)
        x=self.body(x)
        # Decoder starts here
        x=self.decoder(x)
        x = F.interpolate(x, size=output_shape, mode='bilinear', align_corners=False)

        # Torchvision compatible output
        result = OrderedDict()
        result["out"] = x

        # if self.aux_classifier is not None:
        #     x = features["aux"]
        #     x = self.aux_classifier(x)
        #     x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        #     result["aux"] = x

        return result

@register_model()
def regseg_custom(
    *,
    weights = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    aux_loss: Optional[bool] = None,
    weights_backbone = None,
    **kwargs: Any,
) :
    """Constructs a RegSeg model

    .. betastatus:: segmentation module

    Reference: `Rethinking Atrous Convolution for Semantic Image Segmentation <https://arxiv.org/abs/1706.05587>`__.

    Args:
        weights (:class:`~torchvision.models.segmentation.DeepLabV3_ResNet50_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.DeepLabV3_ResNet50_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        weights_backbone (:class:`~torchvision.models.ResNet50_Weights`, optional): The pretrained weights for the
            backbone
        **kwargs: unused

    .. autoclass:: torchvision.models.segmentation.DeepLabV3_ResNet50_Weights
        :members:
    """
    #weights = DeepLabV3_ResNet50_Weights.verify(weights)
    #weights_backbone = ResNet50_Weights.verify(weights_backbone)
    

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
        aux_loss = _ovewrite_value_param("aux_loss", aux_loss, True)

    model = RegSeg_model(num_classes=num_classes, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model