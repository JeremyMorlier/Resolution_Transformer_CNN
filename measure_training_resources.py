import argparse
import json
from pathlib import Path


IMAGENET_TRAIN_SAMPLES = 1_281_167


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract training memory and FLOPs measurements from a TXT training log."
    )
    parser.add_argument("log", type=Path, help="TXT logger output produced by a training script")
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs for the training FLOPs estimate; defaults to the epochs value stored in the log header",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=IMAGENET_TRAIN_SAMPLES,
        help=f"Training samples per epoch for FLOPs estimate; defaults to ImageNet-1k ({IMAGENET_TRAIN_SAMPLES})",
    )
    parser.add_argument(
        "--backward-multiplier",
        type=float,
        default=3.0,
        help="Training FLOPs multiplier applied to forward FLOPs per sample",
    )
    return parser.parse_args()


def load_log(path):
    with path.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]

    if not records:
        raise ValueError(f"{path} is empty")

    return records[0], records[1:]


def latest_value(records, key):
    for record in reversed(records):
        if key in record:
            return record[key]
    return None


def first_value(records, key):
    for record in records:
        if key in record:
            return record[key]
    return None


def select_crop_value(values, crops, crop):
    if values is None:
        return None
    if crops is None or crop is None:
        return values[0] if isinstance(values, list) and values else values

    for index, measured_crop in enumerate(crops):
        if measured_crop == crop and index < len(values):
            return values[index]
    return values[0] if isinstance(values, list) and values else values


def format_number(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main():
    args = parse_args()
    header, records = load_log(args.log)
    logged_args = header.get("args", {})
    epochs = args.epochs if args.epochs is not None else logged_args.get("epochs")
    if epochs is None:
        raise ValueError("The epoch count is missing from the log header; pass it with --epochs")

    crop = logged_args.get("img_size") or logged_args.get("val_crop_size")
    measured_crops = first_value(records, "measured_crops")
    analytical_memory_values = first_value(records, "memory")
    analytical_total_memory_values = first_value(records, "total_memories")
    forward_flops_values = first_value(records, "model_ops")

    analytical_memory = select_crop_value(analytical_memory_values, measured_crops, crop)
    analytical_total_memory = select_crop_value(analytical_total_memory_values, measured_crops, crop)
    forward_flops = select_crop_value(forward_flops_values, measured_crops, crop)
    forward_flops_total = forward_flops[0] if isinstance(forward_flops, list) else forward_flops

    train_flops_per_sample = None
    estimated_training_flops = None
    if forward_flops_total is not None:
        train_flops_per_sample = args.backward_multiplier * forward_flops_total
        estimated_training_flops = train_flops_per_sample * args.train_samples * epochs

    measured_allocated = latest_value(records, "train_peak_cuda_memory_allocated")
    measured_allocated_mb = latest_value(records, "train_peak_cuda_memory_allocated_mb")
    measured_reserved = latest_value(records, "train_peak_cuda_memory_reserved")
    measured_reserved_mb = latest_value(records, "train_peak_cuda_memory_reserved_mb")

    print(f"log: {args.log}")
    print(f"crop_size: {crop}")
    print(f"epochs_for_estimate: {epochs}")
    print(f"train_samples_per_epoch: {args.train_samples}")
    print(f"backward_multiplier: {format_number(args.backward_multiplier)}")
    print(f"measured_peak_cuda_memory_allocated_bytes: {measured_allocated}")
    print(f"measured_peak_cuda_memory_allocated_mb: {format_number(measured_allocated_mb)}")
    print(f"measured_peak_cuda_memory_reserved_bytes: {measured_reserved}")
    print(f"measured_peak_cuda_memory_reserved_mb: {format_number(measured_reserved_mb)}")
    print(f"analytical_activation_memory: {analytical_memory}")
    print(f"analytical_total_memory: {analytical_total_memory}")
    print(f"forward_flops_per_sample: {forward_flops_total}")
    print(f"forward_flops_breakdown: {forward_flops}")
    print(f"estimated_training_flops_per_sample: {format_number(train_flops_per_sample)}")
    print(f"estimated_training_flops_total: {format_number(estimated_training_flops)}")


if __name__ == "__main__":
    main()
