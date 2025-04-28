from .cityscapes.cityscapes import Cityscapes

from .imagenet import ImageNet

__all__ = (
    "ImageFolder",
    "DatasetFolder",
    "FakeData",
    "Cityscapes",
    "ImageNet",
)


# We override current module's attributes to handle the import:
# from torchvision.datasets import wrap_dataset_for_transforms_v2
# without a cyclic error.
# Ref: https://peps.python.org/pep-0562/
def __getattr__(name):
    if name in ("wrap_dataset_for_transforms_v2",):
        from torchvision.tv_tensors._dataset_wrapper import wrap_dataset_for_transforms_v2

        return wrap_dataset_for_transforms_v2

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
