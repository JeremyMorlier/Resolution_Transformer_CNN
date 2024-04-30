from .._api import register_model, Weights, WeightsEnum

from mobile_sam import sam_model_registry
from mobile_sam.modeling import TinyViT


__all__ = [
    "sam_vit_h",
    "sam_vit_l",
    "sam_vit_b",
    "sam_vit_t",
    "mobilesam_vit",
]

@register_model()
def sam_vit_h(checkpoint = None) :
    return sam_model_registry["vit_h"](checkpoint = checkpoint)

@register_model()
def sam_vit_l(checkpoint = None) :
    return sam_model_registry["vit_l"](checkpoint = checkpoint)

@register_model()
def sam_vit_b(checkpoint = None) :
    return sam_model_registry["vit_b"](checkpoint = checkpoint)

@register_model()
def sam_vit_t(checkpoint = None) :
    return sam_model_registry["vit_t"](checkpoint = checkpoint)

@register_model()
def mobilesam_vit():
    model = TinyViT(img_size=1024, in_chans=3, num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8
            )
    ## load pretrained TinyViT weights, please download from https://github.com/wkcn/TinyViT?tab=readme-ov-file
    # pretrained_weights = torch.load("path_to_pth")["model"]
    # del pretrained_weights["head.weight"]
    # del pretrained_weights["head.bias"]
    # model.load_state_dict(pretrained_weights, strict=False)
    
    return model