"""Shared architecture specifications and scaling-grid enumeration.

The same model grid is consumed by the analytic cost model (``scaling_resources.py``),
the measured training-step benchmark (``benchmark_training_step.py``) and the SLURM
scripts. Keeping the definitions here means the three cannot drift apart, which would
silently break the join between the analytic and the measured CSV.
"""

from dataclasses import dataclass


DEFAULT_RESOLUTIONS = (112, 144, 176, 224, 256, 288, 320, 384)
DEFAULT_BASELINE_RESOLUTION = 224
DEFAULT_VIT_LAYER_VALUES = (6, 9, 12, 18, 24)
DEFAULT_VIT_HIDDEN_DIM_VALUES = (384, 576, 768, 1024, 1280)
DEFAULT_VIT_MLP_DIM_VALUES = (1536, 2304, 3072, 4096, 5120)

# Cityscapes training crops. The benchmark feeds R x 2R inputs to keep the dataset aspect ratio.
DEFAULT_SEGMENTATION_RESOLUTIONS = (256, 384, 512, 768, 1024)
SEGMENTATION_NUM_CLASSES = 19


@dataclass(frozen=True)
class ResNetSpec:
    name: str
    channels: tuple[int, int, int, int, int]
    depths: tuple[int, int, int, int] = (3, 4, 6, 3)
    first_conv_resize: int = 0


@dataclass(frozen=True)
class ViTSpec:
    name: str
    patch_size: int
    num_layers: int
    num_heads: int
    hidden_dim: int
    mlp_dim: int


@dataclass(frozen=True)
class RegSegSpec:
    name: str
    regseg_name: str
    channels: tuple[int, int, int, int, int]
    gw: int = 16
    first_conv_resize: int = 0


@dataclass(frozen=True)
class ScalingPoint:
    """One measurable point of the grid: a concrete architecture at a concrete resolution."""

    family: str
    model_name: str
    scaling_axis: str
    scale_index: int
    resolution: int
    spec: object


RESNET_SPECS = (
    ResNetSpec("resnet50_c0.50", (32, 32, 64, 128, 256)),
    ResNetSpec("resnet50_c0.75", (48, 48, 96, 192, 384)),
    ResNetSpec("resnet50_c1.00", (64, 64, 128, 256, 512)),
    ResNetSpec("resnet50_c1.50", (96, 96, 192, 384, 768)),
    ResNetSpec("resnet50_c2.00", (128, 128, 256, 512, 1024)),
)

# Depth axis: the chapter names depth alongside width as a CNN scaling variable, so it needs
# its own axis rather than being folded into the width sweep. Depths follow the standard
# ResNet-26/50/101/152 bottleneck stage layouts at constant width.
RESNET_DEPTH_SPECS = (
    ResNetSpec("resnet50_d26", (64, 64, 128, 256, 512), (2, 2, 2, 2)),
    ResNetSpec("resnet50_d38", (64, 64, 128, 256, 512), (3, 3, 3, 3)),
    ResNetSpec("resnet50_d50", (64, 64, 128, 256, 512), (3, 4, 6, 3)),
    ResNetSpec("resnet50_d101", (64, 64, 128, 256, 512), (3, 4, 23, 3)),
    ResNetSpec("resnet50_d152", (64, 64, 128, 256, 512), (3, 8, 36, 3)),
)

VIT_SPECS = (
    ViTSpec("vit_ti_16", 16, 12, 3, 192, 768),
    ViTSpec("vit_s_16", 16, 12, 6, 384, 1536),
    ViTSpec("vit_b_16", 16, 12, 12, 768, 3072),
    ViTSpec("vit_l_16", 16, 24, 16, 1024, 4096),
)

# RegSeg width sweep around the exp48_decoder26 default channel list used in train_semantic.py.
# ``gw`` (group width) is scaled together with the channels: RegSeg's DilatedConv asserts that
# ``channels / num_dilation_splits`` is divisible by ``gw``, so a fixed gw=16 makes the narrow
# variants unbuildable. Scaling it keeps the block structure self-similar across the sweep.
REGSEG_SPECS = (
    RegSegSpec("regseg_c0.50", "exp48_decoder26", (16, 24, 64, 128, 160), 8),
    RegSegSpec("regseg_c0.75", "exp48_decoder26", (24, 36, 96, 192, 240), 12),
    RegSegSpec("regseg_c1.00", "exp48_decoder26", (32, 48, 128, 256, 320), 16),
    RegSegSpec("regseg_c1.50", "exp48_decoder26", (48, 72, 192, 384, 480), 24),
    RegSegSpec("regseg_c2.00", "exp48_decoder26", (64, 96, 256, 512, 640), 32),
)

REGSEG_BASELINE_INDEX = 2


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split()]


def vit_heads_for_hidden_dim(hidden_dim: int) -> int:
    if hidden_dim % 64 != 0:
        raise ValueError(f"ViT hidden_dim must be divisible by 64 for this sweep, got {hidden_dim}")
    return hidden_dim // 64


def spec_details(spec) -> str:
    """Human-readable architecture summary, written to the ``details`` column of both CSVs."""
    if isinstance(spec, ResNetSpec):
        return f"channels={list(spec.channels)};depths={list(spec.depths)}"
    if isinstance(spec, ViTSpec):
        return (
            f"patch={spec.patch_size};layers={spec.num_layers};heads={spec.num_heads};"
            f"hidden={spec.hidden_dim};mlp={spec.mlp_dim}"
        )
    if isinstance(spec, RegSegSpec):
        return f"regseg={spec.regseg_name};channels={list(spec.channels)};gw={spec.gw}"
    raise TypeError(f"Unsupported spec type: {type(spec)!r}")


def spec_signature(spec, resolution: int) -> tuple:
    """Canonical key used to avoid measuring the same network twice.

    The grid deliberately emits the baseline architecture under several axes (for ViT,
    ``vit_layers_12``, ``vit_hidden_768``, ``vit_mlp_3072`` and ``vit_b_16@224`` are the same
    network), so the benchmark measures once and reuses the result for every axis.
    """
    if isinstance(spec, ResNetSpec):
        return ("resnet", spec.channels, spec.depths, spec.first_conv_resize, resolution)
    if isinstance(spec, ViTSpec):
        return (
            "vit",
            spec.patch_size,
            spec.num_layers,
            spec.num_heads,
            spec.hidden_dim,
            spec.mlp_dim,
            resolution,
        )
    if isinstance(spec, RegSegSpec):
        return ("regseg", spec.regseg_name, spec.channels, spec.gw, spec.first_conv_resize, resolution)
    raise TypeError(f"Unsupported spec type: {type(spec)!r}")


def family_of(spec) -> str:
    if isinstance(spec, ResNetSpec):
        return "resnet"
    if isinstance(spec, ViTSpec):
        return "vit"
    if isinstance(spec, RegSegSpec):
        return "regseg"
    raise TypeError(f"Unsupported spec type: {type(spec)!r}")


def build_model(spec, resolution: int, num_classes: int):
    """Instantiate the model for a spec.

    Mirrors ``train_classification.get_param_model`` and the ``regseg_custom`` branch of
    ``train_semantic.main`` so the benchmarked network is the one that actually gets trained.
    ``models`` is imported lazily to keep this module cheap to import.
    """
    from models import get_model

    if isinstance(spec, ResNetSpec):
        return get_model(
            "resnet50_resize",
            weights=None,
            num_classes=num_classes,
            channels=list(spec.channels),
            depths=list(spec.depths),
            first_conv_resize=spec.first_conv_resize,
        )
    if isinstance(spec, ViTSpec):
        return get_model(
            "vit_custom",
            weights=None,
            num_classes=num_classes,
            patch_size=spec.patch_size,
            num_layers=spec.num_layers,
            num_heads=spec.num_heads,
            hidden_dim=spec.hidden_dim,
            mlp_dim=spec.mlp_dim,
            image_size=resolution,
        )
    if isinstance(spec, RegSegSpec):
        return get_model(
            "regseg_custom",
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
            aux_loss=None,
            regseg_name=spec.regseg_name,
            channels=list(spec.channels),
            gw=spec.gw,
            first_conv_resize=spec.first_conv_resize,
        )
    raise TypeError(f"Unsupported spec type: {type(spec)!r}")


def baseline_spec_for_family(family: str, resnet_baseline_index: int = 2, vit_baseline_index: int = 2):
    if family == "resnet":
        return RESNET_SPECS[resnet_baseline_index]
    if family == "vit":
        return VIT_SPECS[vit_baseline_index]
    if family == "regseg":
        return REGSEG_SPECS[REGSEG_BASELINE_INDEX]
    raise ValueError(f"Unsupported family: {family}")


def iter_scaling_points(
    families=("resnet", "vit"),
    resolutions=DEFAULT_RESOLUTIONS,
    baseline_resolution: int = DEFAULT_BASELINE_RESOLUTION,
    resnet_baseline_index: int = 2,
    vit_baseline_index: int = 2,
    vit_layer_values=DEFAULT_VIT_LAYER_VALUES,
    vit_hidden_dim_values=DEFAULT_VIT_HIDDEN_DIM_VALUES,
    vit_mlp_dim_values=DEFAULT_VIT_MLP_DIM_VALUES,
    include_resnet_depth: bool = False,
    segmentation_resolutions=DEFAULT_SEGMENTATION_RESOLUTIONS,
) -> list[ScalingPoint]:
    """Enumerate the scaling grid.

    The emission order is part of the contract: it is the order of the rows in
    ``scaling_resources.csv``, and SLURM array task ids index into this list.
    """
    points: list[ScalingPoint] = []

    for family in families:
        if family == "resnet":
            baseline_spec = RESNET_SPECS[resnet_baseline_index]
            for index, spec in enumerate(RESNET_SPECS):
                points.append(ScalingPoint("resnet", spec.name, "model", index, baseline_resolution, spec))
            if include_resnet_depth:
                for index, spec in enumerate(RESNET_DEPTH_SPECS):
                    points.append(ScalingPoint("resnet", spec.name, "model_depth", index, baseline_resolution, spec))
            for index, resolution in enumerate(resolutions):
                points.append(
                    ScalingPoint("resnet", baseline_spec.name, "resolution", index, resolution, baseline_spec)
                )

        elif family == "vit":
            baseline_spec = VIT_SPECS[vit_baseline_index]
            for index, num_layers in enumerate(vit_layer_values):
                spec = ViTSpec(
                    f"vit_layers_{num_layers}",
                    baseline_spec.patch_size,
                    num_layers,
                    baseline_spec.num_heads,
                    baseline_spec.hidden_dim,
                    baseline_spec.mlp_dim,
                )
                points.append(ScalingPoint("vit", spec.name, "model_layers", index, baseline_resolution, spec))
            for index, hidden_dim in enumerate(vit_hidden_dim_values):
                spec = ViTSpec(
                    f"vit_hidden_{hidden_dim}",
                    baseline_spec.patch_size,
                    baseline_spec.num_layers,
                    vit_heads_for_hidden_dim(hidden_dim),
                    hidden_dim,
                    baseline_spec.mlp_dim,
                )
                points.append(ScalingPoint("vit", spec.name, "model_hidden_dim", index, baseline_resolution, spec))
            for index, mlp_dim in enumerate(vit_mlp_dim_values):
                spec = ViTSpec(
                    f"vit_mlp_{mlp_dim}",
                    baseline_spec.patch_size,
                    baseline_spec.num_layers,
                    baseline_spec.num_heads,
                    baseline_spec.hidden_dim,
                    mlp_dim,
                )
                points.append(ScalingPoint("vit", spec.name, "model_mlp_dim", index, baseline_resolution, spec))
            for index, resolution in enumerate(resolutions):
                points.append(ScalingPoint("vit", baseline_spec.name, "resolution", index, resolution, baseline_spec))

        elif family == "regseg":
            baseline_spec = REGSEG_SPECS[REGSEG_BASELINE_INDEX]
            for index, spec in enumerate(REGSEG_SPECS):
                points.append(ScalingPoint("regseg", spec.name, "model", index, 1024, spec))
            for index, resolution in enumerate(segmentation_resolutions):
                points.append(
                    ScalingPoint("regseg", baseline_spec.name, "resolution", index, resolution, baseline_spec)
                )

        else:
            raise ValueError(f"Unsupported family: {family}")

    return points
