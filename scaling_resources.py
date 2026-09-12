import argparse
import csv
import os
from pathlib import Path

import torch
from torchinfo import summary

from memory_flops import flops_per_sequence, total_memory_per_sequence
from scaling_specs import (
    DEFAULT_SEGMENTATION_RESOLUTIONS,
    REGSEG_SPECS,
    RESNET_SPECS,
    SEGMENTATION_NUM_CLASSES,
    VIT_SPECS,
    RegSegSpec,
    ResNetSpec,
    ViTSpec,
    baseline_spec_for_family,
    build_model,
    iter_scaling_points,
    parse_ints,
    spec_details,
    vit_heads_for_hidden_dim,
)


IMAGENET_TRAIN_SAMPLES = 1_281_167
CITYSCAPES_TRAIN_SAMPLES = 2_975
MIB = 1024**2

# Re-exported for backwards compatibility: these used to be defined here, before the grid moved
# to scaling_specs.py so that the analytic and the measured benchmark could share it.
__all__ = [
    "RESNET_SPECS", "VIT_SPECS", "REGSEG_SPECS", "ResNetSpec", "ViTSpec", "RegSegSpec",
    "parse_ints", "vit_heads_for_hidden_dim",
]


def bytes_per_parameter_state(optimizer: str, parameter_bytes: int) -> int:
    # Parameter + gradient + optimizer state, assuming fp32 optimizer state.
    if optimizer == "adamw":
        return parameter_bytes + parameter_bytes + 8
    if optimizer == "sgd_momentum":
        return parameter_bytes + parameter_bytes + parameter_bytes
    if optimizer == "sgd":
        return parameter_bytes + parameter_bytes
    raise ValueError(f"Unsupported optimizer memory model: {optimizer}")


def vit_parameter_count(spec: ViTSpec, resolution: int, num_classes: int) -> int:
    seq_len = (resolution // spec.patch_size) ** 2 + 1
    patch_embed = 3 * spec.patch_size * spec.patch_size * spec.hidden_dim + spec.hidden_dim
    embeddings = spec.hidden_dim + seq_len * spec.hidden_dim
    encoder_norms = 4 * spec.hidden_dim
    attention = 4 * spec.hidden_dim * spec.hidden_dim + 4 * spec.hidden_dim
    mlp = 2 * spec.hidden_dim * spec.mlp_dim + spec.mlp_dim + spec.hidden_dim
    head = spec.hidden_dim * num_classes + num_classes
    return patch_embed + embeddings + spec.num_layers * (encoder_norms + attention + mlp) + head


def estimate_training_memory_mb(
    parameter_count: int,
    activation_bytes_per_sample: float,
    batch_size: int,
    parameter_bytes: int,
    optimizer: str,
    activation_training_multiplier: float,
) -> float:
    state_bytes = parameter_count * bytes_per_parameter_state(optimizer, parameter_bytes)
    activation_bytes = activation_bytes_per_sample * batch_size * activation_training_multiplier
    return (state_bytes + activation_bytes) / MIB


def resnet_row(
    spec: ResNetSpec,
    resolution: int,
    args: argparse.Namespace,
    scaling_axis: str,
    scale_index: int,
    baseline_forward_flops: float | None,
    baseline_memory_mb: float | None,
) -> dict[str, object]:
    model = build_model(spec, resolution, args.num_classes)
    info = summary(
        model,
        (1, 3, resolution, resolution),
        verbose=0,
        col_names=("output_size", "num_params", "mult_adds"),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    forward_flops = 2 * info.total_mult_adds
    activation_bytes = float(info.total_output_bytes)
    training_memory_mb = estimate_training_memory_mb(
        parameter_count,
        activation_bytes,
        args.batch_size,
        args.parameter_bytes,
        args.optimizer_memory,
        args.activation_training_multiplier,
    )
    return make_row(
        family="resnet",
        model_name=spec.name,
        scaling_axis=scaling_axis,
        scale_index=scale_index,
        resolution=resolution,
        batch_size=args.batch_size,
        epochs=args.epochs,
        train_samples=args.train_samples,
        parameter_count=parameter_count,
        forward_flops=forward_flops,
        activation_memory_mb=activation_bytes / MIB,
        estimated_training_memory_mb=training_memory_mb,
        args=args,
        baseline_forward_flops=baseline_forward_flops,
        baseline_memory_mb=baseline_memory_mb,
        details=spec_details(spec),
    )


def regseg_row(
    spec: RegSegSpec,
    resolution: int,
    args: argparse.Namespace,
    scaling_axis: str,
    scale_index: int,
    baseline_forward_flops: float | None,
    baseline_memory_mb: float | None,
) -> dict[str, object]:
    """Analytic row for RegSeg. Inputs keep the Cityscapes aspect ratio (R x 2R)."""
    model = build_model(spec, resolution, SEGMENTATION_NUM_CLASSES)
    info = summary(
        model,
        (1, 3, resolution, 2 * resolution),
        verbose=0,
        col_names=("output_size", "num_params", "mult_adds"),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    forward_flops = 2 * info.total_mult_adds
    activation_bytes = float(info.total_output_bytes)
    training_memory_mb = estimate_training_memory_mb(
        parameter_count,
        activation_bytes,
        args.segmentation_batch_size,
        args.parameter_bytes,
        args.optimizer_memory,
        args.activation_training_multiplier,
    )
    return make_row(
        family="regseg",
        model_name=spec.name,
        scaling_axis=scaling_axis,
        scale_index=scale_index,
        resolution=resolution,
        batch_size=args.segmentation_batch_size,
        epochs=args.epochs,
        train_samples=args.segmentation_train_samples,
        parameter_count=parameter_count,
        forward_flops=forward_flops,
        activation_memory_mb=activation_bytes / MIB,
        estimated_training_memory_mb=training_memory_mb,
        args=args,
        baseline_forward_flops=baseline_forward_flops,
        baseline_memory_mb=baseline_memory_mb,
        details=spec_details(spec),
    )


def vit_row(
    spec: ViTSpec,
    resolution: int,
    args: argparse.Namespace,
    scaling_axis: str,
    scale_index: int,
    baseline_forward_flops: float | None,
    baseline_memory_mb: float | None,
) -> dict[str, object]:
    if resolution % spec.patch_size != 0:
        raise ValueError(f"{spec.name} patch size {spec.patch_size} does not divide resolution {resolution}")

    patch_count = (resolution // spec.patch_size) ** 2
    forward_flops = flops_per_sequence(
        spec.patch_size,
        patch_count,
        spec.num_layers,
        spec.num_heads,
        spec.hidden_dim,
        spec.mlp_dim,
        args.num_classes,
    )[0]
    parameter_count = vit_parameter_count(spec, resolution, args.num_classes)
    activation_elements = total_memory_per_sequence(
        resolution,
        spec.num_layers,
        patch_count,
        spec.hidden_dim,
        spec.mlp_dim,
    )
    activation_bytes = float(activation_elements * args.activation_bytes)
    training_memory_mb = estimate_training_memory_mb(
        parameter_count,
        activation_bytes,
        args.batch_size,
        args.parameter_bytes,
        args.optimizer_memory,
        args.activation_training_multiplier,
    )
    return make_row(
        family="vit",
        model_name=spec.name,
        scaling_axis=scaling_axis,
        scale_index=scale_index,
        resolution=resolution,
        batch_size=args.batch_size,
        epochs=args.epochs,
        train_samples=args.train_samples,
        parameter_count=parameter_count,
        forward_flops=forward_flops,
        activation_memory_mb=activation_bytes / MIB,
        estimated_training_memory_mb=training_memory_mb,
        args=args,
        baseline_forward_flops=baseline_forward_flops,
        baseline_memory_mb=baseline_memory_mb,
        details=spec_details(spec),
    )


def make_row(
    family: str,
    model_name: str,
    scaling_axis: str,
    scale_index: int,
    resolution: int,
    batch_size: int,
    epochs: int,
    train_samples: int,
    parameter_count: int,
    forward_flops: float,
    activation_memory_mb: float,
    estimated_training_memory_mb: float,
    args: argparse.Namespace,
    baseline_forward_flops: float | None,
    baseline_memory_mb: float | None,
    details: str,
) -> dict[str, object]:
    train_flops_per_sample = args.backward_multiplier * forward_flops
    total_train_flops = train_flops_per_sample * train_samples * epochs
    flops_ratio = forward_flops / baseline_forward_flops if baseline_forward_flops else 1.0
    memory_ratio = estimated_training_memory_mb / baseline_memory_mb if baseline_memory_mb else 1.0
    return {
        "family": family,
        "model_name": model_name,
        "scaling_axis": scaling_axis,
        "scale_index": scale_index,
        "resolution": resolution,
        "batch_size": batch_size,
        "epochs": epochs,
        "train_samples": train_samples,
        "parameters": parameter_count,
        "activation_memory_mb_per_sample": round(activation_memory_mb, 6),
        "estimated_training_memory_mb": round(estimated_training_memory_mb, 6),
        "forward_flops_per_sample": int(forward_flops),
        "training_flops_per_sample": int(train_flops_per_sample),
        "total_training_flops": int(total_train_flops),
        "forward_flops_ratio": round(flops_ratio, 6),
        "training_memory_ratio": round(memory_ratio, 6),
        "details": details,
    }


def build_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    """Analytic cost rows for the shared scaling grid.

    The grid itself comes from ``scaling_specs.iter_scaling_points`` so that this table and the
    measured table produced by ``benchmark_training_step.py`` stay joinable on
    ``(family, model_name, resolution)``.
    """
    row_builders = {"resnet": resnet_row, "vit": vit_row, "regseg": regseg_row}
    rows = []

    for family in args.families:
        builder = row_builders[family]
        baseline_spec = baseline_spec_for_family(family, args.resnet_baseline_index, args.vit_baseline_index)
        baseline_resolution = (
            args.segmentation_baseline_resolution if family == "regseg" else args.baseline_resolution
        )
        baseline = builder(baseline_spec, baseline_resolution, args, "baseline", 0, None, None)
        baseline_forward_flops = baseline["forward_flops_per_sample"]
        baseline_memory_mb = baseline["estimated_training_memory_mb"]

        points = iter_scaling_points(
            families=(family,),
            resolutions=args.resolutions,
            baseline_resolution=args.baseline_resolution,
            resnet_baseline_index=args.resnet_baseline_index,
            vit_baseline_index=args.vit_baseline_index,
            vit_layer_values=args.vit_layer_values,
            vit_hidden_dim_values=args.vit_hidden_dim_values,
            vit_mlp_dim_values=args.vit_mlp_dim_values,
            include_resnet_depth=args.include_resnet_depth,
            segmentation_resolutions=args.segmentation_resolutions,
        )
        for point in points:
            rows.append(
                builder(
                    point.spec,
                    point.resolution,
                    args,
                    point.scaling_axis,
                    point.scale_index,
                    baseline_forward_flops,
                    baseline_memory_mb,
                )
            )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare model scaling and input-resolution scaling costs for ResNet and ViT."
    )
    parser.add_argument("--output", type=Path, default=Path("scaling_resources.csv"))
    parser.add_argument("--plot", type=Path, help="Optional path for a PNG plot of FLOPs and memory ratios")
    parser.add_argument(
        "--families", nargs="+", choices=("resnet", "vit", "regseg"), default=["resnet", "vit"]
    )
    parser.add_argument("--resolutions", type=parse_ints, default=parse_ints("112 144 176 224 256 288 320 384"))
    parser.add_argument("--baseline-resolution", type=int, default=224)
    parser.add_argument("--resnet-baseline-index", type=int, default=2, choices=range(len(RESNET_SPECS)))
    parser.add_argument("--vit-baseline-index", type=int, default=2, choices=range(len(VIT_SPECS)))
    parser.add_argument("--vit-layer-values", type=parse_ints, default=parse_ints("6 9 12 18 24"))
    parser.add_argument("--vit-hidden-dim-values", type=parse_ints, default=parse_ints("384 576 768 1024 1280"))
    parser.add_argument("--vit-mlp-dim-values", type=parse_ints, default=parse_ints("1536 2304 3072 4096 5120"))
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--train-samples", type=int, default=IMAGENET_TRAIN_SAMPLES)
    parser.add_argument("--backward-multiplier", type=float, default=3.0)
    parser.add_argument("--activation-training-multiplier", type=float, default=2.0)
    parser.add_argument("--activation-bytes", type=int, default=4)
    parser.add_argument("--parameter-bytes", type=int, default=4)
    parser.add_argument("--optimizer-memory", choices=("adamw", "sgd_momentum", "sgd"), default="adamw")
    parser.add_argument(
        "--include-resnet-depth",
        action="store_true",
        help="Add the ResNet depth-scaling axis. Keep this in sync with benchmark_training_step.py, "
        "otherwise the analytic and measured tables no longer cover the same points.",
    )
    parser.add_argument(
        "--segmentation-resolutions",
        type=parse_ints,
        default=list(DEFAULT_SEGMENTATION_RESOLUTIONS),
        help="Cityscapes training crop heights; inputs are R x 2R",
    )
    parser.add_argument("--segmentation-baseline-resolution", type=int, default=1024)
    parser.add_argument("--segmentation-batch-size", type=int, default=8)
    parser.add_argument("--segmentation-train-samples", type=int, default=CITYSCAPES_TRAIN_SAMPLES)
    return parser.parse_args()


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: list[dict[str, object]], output: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    families = [family for family in ("resnet", "vit") if any(row["family"] == family for row in rows)]
    figure, axes = plt.subplots(1, len(families), figsize=(6 * len(families), 5), squeeze=False)
    markers = {
        "model": "o",
        "model_layers": "o",
        "model_hidden_dim": "^",
        "model_mlp_dim": "D",
        "resolution": "s",
    }

    for col_index, family in enumerate(families):
        axis = axes[0][col_index]
        family_rows = [row for row in rows if row["family"] == family]
        scaling_axes = sorted({row["scaling_axis"] for row in family_rows})
        for scaling_axis in scaling_axes:
            selected = [row for row in family_rows if row["scaling_axis"] == scaling_axis]
            x_values = [row["training_flops_per_sample"] / 1e9 for row in selected]
            y_values = [row["estimated_training_memory_mb"] / 1024 for row in selected]
            axis.scatter(
                x_values,
                y_values,
                marker=markers.get(scaling_axis, "o"),
                s=48,
                alpha=0.85,
                label=scaling_axis,
            )
            axis.plot(x_values, y_values, linewidth=1, alpha=0.55)

        axis.set_title(f"{family.upper()} memory vs FLOPs")
        axis.set_xlabel("training FLOPs per sample (GFLOPs)")
        axis.set_ylabel("estimated training memory (GiB)")
        axis.grid(True, linewidth=0.5, alpha=0.35)
        axis.legend()

    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def print_summary(rows: list[dict[str, object]]) -> None:
    for family in sorted({row["family"] for row in rows}):
        model_rows = [
            row for row in rows if row["family"] == family and str(row["scaling_axis"]).startswith("model")
        ]
        resolution_rows = [row for row in rows if row["family"] == family and row["scaling_axis"] == "resolution"]
        max_model = max(model_rows, key=lambda row: row["forward_flops_ratio"])
        max_resolution = max(resolution_rows, key=lambda row: row["forward_flops_ratio"])
        print(
            f"{family}: max model scaling {max_model['model_name']} at {max_model['forward_flops_ratio']}x FLOPs, "
            f"{max_model['training_memory_ratio']}x memory"
        )
        print(
            f"{family}: max resolution scaling {max_resolution['resolution']}px at "
            f"{max_resolution['forward_flops_ratio']}x FLOPs, {max_resolution['training_memory_ratio']}x memory"
        )


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    write_csv(rows, args.output)
    if args.plot:
        write_plot(rows, args.plot)
    print_summary(rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    if args.plot:
        print(f"wrote plot to {args.plot}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
