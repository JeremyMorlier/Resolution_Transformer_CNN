"""Measure training step time and training memory against resolution and model scaling.

This is the measured counterpart of the analytic model in ``scaling_resources.py``. It runs the
real training step (forward, backward, optimizer) on synthetic batches, so the reported cost is
the model's cost and not the input pipeline's. Rows join to ``scaling_resources.csv`` on
``(family, model_name, resolution)``.

Why synthetic inputs: ``train_classification.train_one_epoch`` calls ``.item()`` three times per
step and runs JPEG decode plus RandAugment on the CPU, so a real epoch at 112px is dataloader
bound and its wall time understates the compute saved by lowering the resolution. Use
``--dataloader-mode`` on a few points to quantify that gap instead of inheriting it everywhere.
"""

import argparse
import csv
import gc
import json
import math
import os
import platform
import random
import resource
import statistics
import subprocess
import time
from pathlib import Path

import torch
from torch import nn

from scaling_specs import (
    DEFAULT_BASELINE_RESOLUTION,
    DEFAULT_RESOLUTIONS,
    DEFAULT_SEGMENTATION_RESOLUTIONS,
    DEFAULT_VIT_HIDDEN_DIM_VALUES,
    DEFAULT_VIT_LAYER_VALUES,
    DEFAULT_VIT_MLP_DIM_VALUES,
    SEGMENTATION_NUM_CLASSES,
    build_model,
    iter_scaling_points,
    parse_ints,
    spec_details,
    spec_signature,
)


IMAGENET_TRAIN_SAMPLES = 1_281_167
CITYSCAPES_TRAIN_SAMPLES = 2_975
BENCHMARK_VERSION = 1
MIB = 1024**2

FIELDNAMES = (
    # identity / join key against scaling_resources.csv
    "family", "model_name", "scaling_axis", "scale_index", "resolution", "input_height",
    "input_width", "batch_size", "num_classes", "details",
    # model size
    "parameters", "trainable_parameters",
    # run configuration
    "device", "device_name", "torch_version", "cuda_version", "precision", "channels_last",
    "cudnn_benchmark", "tf32", "matmul_precision", "optimizer", "zero_grad_set_to_none",
    "clip_grad_norm", "input_mode", "warmup_steps", "measure_steps", "repeat_index", "seed",
    "threads", "alloc_conf",
    # timing
    "step_time_ms_median", "step_time_ms_mean", "step_time_ms_std", "step_time_ms_min",
    "step_time_ms_p10", "step_time_ms_p90", "step_time_ms_batched_mean", "images_per_second",
    "forward_ms_median", "backward_ms_median", "optimizer_ms_median", "phase_sum_ms_median",
    "phase_residual_ms_median",
    # memory
    "param_bytes", "buffer_bytes", "grad_bytes", "optimizer_state_bytes", "input_batch_bytes",
    "static_bytes", "peak_allocated_bytes", "peak_activation_bytes", "peak_reserved_bytes",
    "fragmentation_bytes", "peak_allocated_mb", "peak_activation_mb",
    "peak_activation_mb_per_sample", "optimizer_bytes_per_parameter", "cpu_max_rss_bytes",
    # extrapolation to a full training run
    "train_samples", "epochs", "steps_per_epoch", "estimated_epoch_seconds",
    "estimated_total_train_hours",
    # dataloader validation mode
    "dataloader_step_ms_median", "loader_only_ms_median", "workers", "gpu_bound_fraction",
    "dataloader_bound",
    # provenance / health
    "sm_clock_mhz_start", "sm_clock_mhz_end", "gpu_temp_c_start", "gpu_temp_c_end",
    "status", "error", "timestamp_iso", "hostname", "run_id", "benchmark_version",
    "config_index",
)


# --------------------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------------------
def configure_backends(args, device):
    """Pin every numerics knob that changes measured time, and report what was pinned.

    TF32 matters more than it looks: torch defaults ``cuda.matmul.allow_tf32`` to False but
    ``cudnn.allow_tf32`` to True. ViT time is dominated by matmuls and ResNet time by
    convolutions, so leaving the defaults alone silently penalises ViT and corrupts exactly the
    CNN-versus-transformer comparison this benchmark exists to make.
    """
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cudnn_benchmark = args.cudnn_benchmark == "on"
    tf32 = args.tf32 == "on"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        torch.set_float32_matmul_precision("high" if tf32 else "highest")
        device_name = torch.cuda.get_device_name(device)
    else:
        torch.set_num_threads(args.threads)
        device_name = platform.processor() or platform.machine()

    return {
        "device": str(device),
        "device_name": device_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if device.type == "cuda" else "",
        "cudnn_benchmark": cudnn_benchmark if device.type == "cuda" else "",
        "tf32": tf32 if device.type == "cuda" else "",
        "matmul_precision": torch.get_float32_matmul_precision(),
        "threads": torch.get_num_threads(),
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "hostname": platform.node(),
    }


def gpu_health(device):
    """SM clock and temperature, so throttled rows can be spotted after the fact."""
    if device.type != "cuda":
        return {"sm_clock_mhz": "", "gpu_temp_c": ""}
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        output = subprocess.run(
            ["nvidia-smi", f"--id={index}", "--query-gpu=clocks.sm,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().split(",")
        return {"sm_clock_mhz": int(output[0]), "gpu_temp_c": int(output[1])}
    except Exception:
        return {"sm_clock_mhz": "", "gpu_temp_c": ""}


# --------------------------------------------------------------------------------------
# workload
# --------------------------------------------------------------------------------------
def input_shape(family, resolution):
    """RegSeg keeps the Cityscapes aspect ratio; classification models are square."""
    if family == "regseg":
        return resolution, 2 * resolution
    return resolution, resolution


def make_synthetic_batch(family, batch_size, resolution, num_classes, device, channels_last):
    height, width = input_shape(family, resolution)
    images = torch.randn(batch_size, 3, height, width, device=device)
    if channels_last:
        images = images.to(memory_format=torch.channels_last)
    if family == "regseg":
        targets = torch.randint(0, num_classes, (batch_size, height, width), device=device)
    else:
        targets = torch.randint(0, num_classes, (batch_size,), device=device)
    return images, targets


def make_criterion(family, label_smoothing):
    if family == "regseg":
        # Mirrors train_semantic.criterion: sum of per-output cross entropies, ignoring 255.
        def segmentation_criterion(outputs, target):
            return sum(nn.functional.cross_entropy(x, target, ignore_index=255) for x in outputs.values())

        return segmentation_criterion
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def forward_loss(model, images, targets, criterion, family):
    if family == "regseg":
        return criterion(model(images, shape=targets.shape[-2:]), targets)
    return criterion(model(images), targets)


def build_optimizer(model, name, lr, weight_decay):
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def autocast_context(device_type, precision):
    if precision == "fp32":
        return torch.autocast(device_type=device_type, enabled=False)
    dtype = torch.float16 if precision == "amp_fp16" else torch.bfloat16
    return torch.autocast(device_type=device_type, dtype=dtype)


def train_step(model, criterion, optimizer, images, targets, scaler, device_type, precision,
               clip_grad_norm, set_to_none, family):
    """The canonical measured step.

    Same operations and same order as ``train_classification.train_one_epoch``, minus the EMA
    update and the three ``.item()`` calls, each of which is a device-to-host sync that would
    serialise the pipeline and inflate every measurement.
    """
    with autocast_context(device_type, precision):
        loss = forward_loss(model, images, targets, criterion, family)

    optimizer.zero_grad(set_to_none=set_to_none)
    if scaler is not None:
        scaler.scale(loss).backward()
        if clip_grad_norm is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
    return loss


# --------------------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------------------
class CudaTimer:
    """Pair of CUDA events. Does not synchronise, so it can be used inside the measured loop."""

    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, *exc):
        self.end.record()
        return False

    def elapsed_ms(self):
        return self.start.elapsed_time(self.end)


class HostTimer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._end = time.perf_counter()
        return False

    def elapsed_ms(self):
        return (self._end - self._start) * 1000.0


def make_timer(device):
    return CudaTimer() if device.type == "cuda" else HostTimer()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def time_steps(step, device, warmup, measure):
    """Two passes: an instrumented one for the distribution, a bare one for throughput.

    The bare pass is what the epoch-time extrapolation uses, because it keeps the CPU-side
    kernel-launch pipelining that a real training loop gets. A large gap between the two means
    the config is launch-bound rather than compute-bound, which is itself a result worth having.
    """
    for _ in range(warmup):
        step()
    synchronize(device)

    timers = []
    for _ in range(measure):
        with make_timer(device) as timer:
            step()
        timers.append(timer)
    synchronize(device)
    samples = [timer.elapsed_ms() for timer in timers]

    synchronize(device)
    start = time.perf_counter()
    for _ in range(measure):
        step()
    synchronize(device)
    batched_mean = (time.perf_counter() - start) * 1000.0 / measure

    return {
        "step_time_ms_median": statistics.median(samples),
        "step_time_ms_mean": statistics.fmean(samples),
        "step_time_ms_std": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "step_time_ms_min": min(samples),
        "step_time_ms_p10": percentile(samples, 0.10),
        "step_time_ms_p90": percentile(samples, 0.90),
        "step_time_ms_batched_mean": batched_mean,
        "samples": samples,
    }


def time_phases(model, criterion, optimizer, images, targets, scaler, device, args, family):
    """Separate pass splitting the step into forward, backward and optimizer.

    Run apart from the total-time pass: the phase boundaries partially serialise the step, so
    the phase sum is not exactly the step time. The residual is reported rather than hidden.
    """
    device_type = device.type
    forward_ms, backward_ms, optimizer_ms = [], [], []

    for _ in range(args.measure_steps):
        with make_timer(device) as forward_timer:
            with autocast_context(device_type, args.precision):
                loss = forward_loss(model, images, targets, criterion, family)

        optimizer.zero_grad(set_to_none=args.zero_grad_set_to_none)
        with make_timer(device) as backward_timer:
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        with make_timer(device) as optimizer_timer:
            if scaler is not None:
                if args.clip_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()

        forward_ms.append(forward_timer)
        backward_ms.append(backward_timer)
        optimizer_ms.append(optimizer_timer)

    synchronize(device)
    forward = statistics.median([t.elapsed_ms() for t in forward_ms])
    backward = statistics.median([t.elapsed_ms() for t in backward_ms])
    optimizer_time = statistics.median([t.elapsed_ms() for t in optimizer_ms])
    return {
        "forward_ms_median": forward,
        "backward_ms_median": backward,
        "optimizer_ms_median": optimizer_time,
        "phase_sum_ms_median": forward + backward + optimizer_time,
    }


# --------------------------------------------------------------------------------------
# memory
# --------------------------------------------------------------------------------------
def measure_memory(model, criterion, optimizer, images, targets, scaler, device, args, family):
    """Decompose training memory into parameters, gradients, optimizer state and activations.

    A single peak number conflates all four, which is precisely the distinction the chapter
    argues about: resolution scaling moves only the activation term, while width scaling moves
    all of them. Two subtleties drive the ordering below:

    - AdamW creates ``exp_avg``/``exp_avg_sq`` lazily inside the first ``step()``, so the state
      must be read after at least one full step or it reads as zero.
    - ``optimizer.zero_grad()`` defaults to ``set_to_none=True``, which frees gradient storage
      between steps. The static accounting therefore forces ``set_to_none=False`` so gradients
      are resident and countable, while the peak pass uses the real training policy.

    Under AMP the split is approximate: autocast keeps a half-precision weight cast cache inside
    the autocast region, so part of what lands in ``peak_activation_bytes`` is really a parameter
    copy. Expect the activation term to shrink by somewhat less than 2x, not exactly 2x.
    """
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    input_batch_bytes = (images.numel() * images.element_size()
                         + targets.numel() * targets.element_size())

    def step(set_to_none):
        train_step(model, criterion, optimizer, images, targets, scaler, device.type,
                   args.precision, args.clip_grad_norm, set_to_none, family)

    # Two steps to reach steady state: the first allocates .grad, the first step() the AdamW state.
    step(False)
    step(False)
    optimizer.zero_grad(set_to_none=False)
    synchronize(device)

    grad_bytes = sum(p.grad.numel() * p.grad.element_size()
                     for p in model.parameters() if p.grad is not None)
    # ``state["step"]`` is a CPU scalar in torch 2.x; counting it as device bytes would be wrong.
    optimizer_state_bytes = sum(
        tensor.numel() * tensor.element_size()
        for state in optimizer.state.values()
        for tensor in state.values()
        if torch.is_tensor(tensor) and tensor.device.type == device.type
    )
    static_bytes = param_bytes + buffer_bytes + grad_bytes + optimizer_state_bytes + input_batch_bytes

    result = {
        "param_bytes": param_bytes,
        "buffer_bytes": buffer_bytes,
        "grad_bytes": grad_bytes,
        "optimizer_state_bytes": optimizer_state_bytes,
        "input_batch_bytes": input_batch_bytes,
        "static_bytes": static_bytes,
        "optimizer_bytes_per_parameter": round(
            optimizer_state_bytes / sum(p.numel() for p in model.parameters()), 4
        ),
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.memory_steps):
            step(args.zero_grad_set_to_none)
        synchronize(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        result.update({
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "peak_activation_bytes": peak_allocated - static_bytes,
            "fragmentation_bytes": peak_reserved - peak_allocated,
            "peak_allocated_mb": round(peak_allocated / MIB, 4),
            "peak_activation_mb": round((peak_allocated - static_bytes) / MIB, 4),
            "peak_activation_mb_per_sample": round(
                (peak_allocated - static_bytes) / MIB / args.batch_size, 6
            ),
            "cpu_max_rss_bytes": "",
        })
    else:
        # No allocator peak counter on CPU. RSS is an upper bound polluted by malloc arena
        # caching, so it is reported in its own column and never as peak_activation_bytes.
        for _ in range(args.memory_steps):
            step(args.zero_grad_set_to_none)
        result.update({
            "peak_allocated_bytes": "", "peak_reserved_bytes": "", "peak_activation_bytes": "",
            "fragmentation_bytes": "", "peak_allocated_mb": "", "peak_activation_mb": "",
            "peak_activation_mb_per_sample": "",
            "cpu_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        })
    return result


# --------------------------------------------------------------------------------------
# dataloader validation mode
# --------------------------------------------------------------------------------------
def measure_dataloader(point, args, device):
    """Compare the synthetic step against a real ImageNet loader at the same resolution.

    Establishes both that the synthetic numbers predict real training in the GPU-bound regime
    and where the low-resolution dataloader-bound regime starts.
    """
    from args import get_classification_argsparse
    from train_classification import load_data
    from references.classification.transforms import get_mixup_cutmix

    loader_args = get_classification_argsparse().parse_known_args([])[0]
    loader_args.data_path = args.data_path
    loader_args.train_crop_size = point.resolution
    loader_args.val_crop_size = point.resolution
    loader_args.val_resize_size = point.resolution + 8
    loader_args.batch_size = args.batch_size
    loader_args.workers = args.workers
    loader_args.auto_augment = args.auto_augment
    loader_args.mixup_alpha = args.mixup_alpha
    loader_args.cutmix_alpha = args.cutmix_alpha
    loader_args.distributed = False
    loader_args.weights = None
    loader_args.test_only = False

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, _, train_sampler, _ = load_data(train_dir, val_dir, loader_args)

    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=loader_args.mixup_alpha,
        cutmix_alpha=loader_args.cutmix_alpha,
        num_categories=args.num_classes,
        use_v2=loader_args.use_v2,
    )
    collate_fn = None
    if mixup_cutmix is not None:
        def collate_fn(batch):
            return mixup_cutmix(*torch.utils.data.default_collate(batch))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    model = build_model(point.spec, point.resolution, args.num_classes).to(device)
    model.train()
    criterion = make_criterion(point.family, args.label_smoothing)
    optimizer = build_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    scaler = torch.amp.GradScaler(device.type) if args.precision == "amp_fp16" else None

    full_steps, loader_only_steps = [], []
    iterator = iter(loader)
    total = args.dataloader_warmup_steps + args.dataloader_steps
    for index in range(total):
        start = time.perf_counter()
        try:
            images, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, targets = next(iterator)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        synchronize(device)
        fetched = time.perf_counter()

        train_step(model, criterion, optimizer, images, targets, scaler, device.type,
                   args.precision, args.clip_grad_norm, args.zero_grad_set_to_none, point.family)
        synchronize(device)
        done = time.perf_counter()

        if index >= args.dataloader_warmup_steps:
            loader_only_steps.append((fetched - start) * 1000.0)
            full_steps.append((done - start) * 1000.0)

    return {
        "dataloader_step_ms_median": statistics.median(full_steps),
        "loader_only_ms_median": statistics.median(loader_only_steps),
        "workers": args.workers,
    }


# --------------------------------------------------------------------------------------
# one grid point
# --------------------------------------------------------------------------------------
def empty_row(point, args, environment, config_index, status, error=""):
    height, width = input_shape(point.family, point.resolution)
    row = {name: "" for name in FIELDNAMES}
    row.update({
        "family": point.family,
        "model_name": point.model_name,
        "scaling_axis": point.scaling_axis,
        "scale_index": point.scale_index,
        "resolution": point.resolution,
        "input_height": height,
        "input_width": width,
        "batch_size": args.batch_size,
        "num_classes": num_classes_for(point.family, args),
        "details": spec_details(point.spec),
        "precision": args.precision,
        "channels_last": args.channels_last,
        "optimizer": args.optimizer,
        "zero_grad_set_to_none": args.zero_grad_set_to_none,
        "clip_grad_norm": args.clip_grad_norm if args.clip_grad_norm is not None else "",
        "input_mode": "dataloader" if args.dataloader_mode else "synthetic",
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "repeat_index": 0,
        "seed": args.seed,
        "status": status,
        "error": error,
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_id": args.run_id,
        "benchmark_version": BENCHMARK_VERSION,
        "config_index": config_index,
    })
    row.update({key: value for key, value in environment.items() if key in FIELDNAMES})
    return row


def num_classes_for(family, args):
    return SEGMENTATION_NUM_CLASSES if family == "regseg" else args.num_classes


def train_samples_for(family, args):
    return args.segmentation_train_samples if family == "regseg" else args.train_samples


def benchmark_point(point, args, environment, config_index, repeat_index=0):
    device = torch.device(args.device)
    num_classes = num_classes_for(point.family, args)
    row = empty_row(point, args, environment, config_index, "ok")
    row["repeat_index"] = repeat_index
    health_start = gpu_health(device)
    row["sm_clock_mhz_start"] = health_start["sm_clock_mhz"]
    row["gpu_temp_c_start"] = health_start["gpu_temp_c"]

    model = build_model(point.spec, point.resolution, num_classes).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.train()
    row["parameters"] = sum(p.numel() for p in model.parameters())
    row["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = make_criterion(point.family, args.label_smoothing)
    optimizer = build_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    scaler = torch.amp.GradScaler(device.type) if args.precision == "amp_fp16" else None
    images, targets = make_synthetic_batch(
        point.family, args.batch_size, point.resolution, num_classes, device, args.channels_last
    )

    def step():
        train_step(model, criterion, optimizer, images, targets, scaler, device.type,
                   args.precision, args.clip_grad_norm, args.zero_grad_set_to_none, point.family)

    timing = time_steps(step, device, args.warmup_steps, args.measure_steps)
    samples = timing.pop("samples")
    row.update({key: round(value, 4) for key, value in timing.items()})
    row["images_per_second"] = round(args.batch_size / (timing["step_time_ms_batched_mean"] / 1000.0), 3)

    if not args.no_phase_breakdown:
        phases = time_phases(model, criterion, optimizer, images, targets, scaler, device, args,
                             point.family)
        row.update({key: round(value, 4) for key, value in phases.items()})
        row["phase_residual_ms_median"] = round(
            timing["step_time_ms_median"] - phases["phase_sum_ms_median"], 4
        )

    if not args.skip_memory:
        row.update(measure_memory(model, criterion, optimizer, images, targets, scaler, device,
                                  args, point.family))

    train_samples = train_samples_for(point.family, args)
    steps_per_epoch = math.ceil(train_samples / args.batch_size)
    epoch_seconds = steps_per_epoch * timing["step_time_ms_batched_mean"] / 1000.0
    row.update({
        "train_samples": train_samples,
        "epochs": args.epochs,
        "steps_per_epoch": steps_per_epoch,
        "estimated_epoch_seconds": round(epoch_seconds, 3),
        "estimated_total_train_hours": round(args.epochs * epoch_seconds / 3600.0, 4),
    })

    if args.dataloader_mode:
        loader_result = measure_dataloader(point, args, device)
        row.update({key: round(value, 4) if isinstance(value, float) else value
                    for key, value in loader_result.items()})
        fraction = timing["step_time_ms_batched_mean"] / loader_result["dataloader_step_ms_median"]
        row["gpu_bound_fraction"] = round(fraction, 4)
        row["dataloader_bound"] = fraction < 0.9

    health_end = gpu_health(device)
    row["sm_clock_mhz_end"] = health_end["sm_clock_mhz"]
    row["gpu_temp_c_end"] = health_end["gpu_temp_c"]

    if args.dump_raw_steps:
        with Path(args.dump_raw_steps).open("a", encoding="utf-8") as file:
            file.write(json.dumps({
                "family": point.family, "model_name": point.model_name,
                "resolution": point.resolution, "batch_size": args.batch_size,
                "repeat_index": repeat_index, "step_ms": [round(value, 5) for value in samples],
            }) + "\n")

    del model, optimizer, images, targets, criterion, scaler
    release_memory(device)
    return row


def release_memory(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------
def iter_configs(args):
    points = iter_scaling_points(
        families=tuple(args.families),
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
    if args.config_index is not None:
        if not 0 <= args.config_index < len(points):
            raise SystemExit(f"--config-index {args.config_index} out of range (0..{len(points) - 1})")
        return [points[args.config_index]], [args.config_index]
    return points, list(range(len(points)))


def write_rows(rows, output, append):
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not (append and output.exists())
    with output.open("a" if append else "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(FIELDNAMES), extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure training step time and training memory across the resolution and "
                    "model-scaling grid."
    )
    grid = parser.add_argument_group("grid selection")
    grid.add_argument("--families", nargs="+", choices=("resnet", "vit", "regseg"),
                      default=["resnet", "vit"])
    grid.add_argument("--resolutions", type=parse_ints, default=list(DEFAULT_RESOLUTIONS))
    grid.add_argument("--baseline-resolution", type=int, default=DEFAULT_BASELINE_RESOLUTION)
    grid.add_argument("--resnet-baseline-index", type=int, default=2)
    grid.add_argument("--vit-baseline-index", type=int, default=2)
    grid.add_argument("--vit-layer-values", type=parse_ints, default=list(DEFAULT_VIT_LAYER_VALUES))
    grid.add_argument("--vit-hidden-dim-values", type=parse_ints,
                      default=list(DEFAULT_VIT_HIDDEN_DIM_VALUES))
    grid.add_argument("--vit-mlp-dim-values", type=parse_ints,
                      default=list(DEFAULT_VIT_MLP_DIM_VALUES))
    grid.add_argument("--include-resnet-depth", action="store_true",
                      help="Add the ResNet depth axis; keep in sync with scaling_resources.py")
    grid.add_argument("--segmentation-resolutions", type=parse_ints,
                      default=list(DEFAULT_SEGMENTATION_RESOLUTIONS))
    grid.add_argument("--config-index", type=int,
                      help="Benchmark a single grid point; this is the SLURM array form and gives "
                           "one process per point, so cuDNN autotune caches and allocator "
                           "fragmentation cannot leak between configs")
    grid.add_argument("--list-configs", action="store_true", help="Print the grid and exit")
    grid.add_argument("--shuffle-configs", action="store_true",
                      help="Randomise config order so any thermal drift shows up as noise rather "
                           "than as a trend correlated with model size")

    workload = parser.add_argument_group("workload")
    workload.add_argument("--batch-size", type=int, default=64)
    workload.add_argument("--num-classes", type=int, default=1000)
    workload.add_argument("--optimizer", choices=("adamw", "sgd_momentum", "sgd"), default="adamw")
    workload.add_argument("--lr", type=float, default=0.003)
    workload.add_argument("--weight-decay", type=float, default=0.3)
    workload.add_argument("--label-smoothing", type=float, default=0.11)
    workload.add_argument("--clip-grad-norm", type=float, default=1.0)
    workload.add_argument("--no-clip-grad-norm", action="store_const", const=None,
                          dest="clip_grad_norm")
    workload.add_argument("--zero-grad-set-to-none", action="store_true", default=True)
    workload.add_argument("--no-zero-grad-set-to-none", action="store_false",
                          dest="zero_grad_set_to_none")

    device_group = parser.add_argument_group("device and precision")
    device_group.add_argument("--device", default="cuda")
    device_group.add_argument("--precision", choices=("fp32", "amp_fp16", "amp_bf16"),
                              default="fp32")
    device_group.add_argument("--channels-last", action="store_true")
    device_group.add_argument("--tf32", choices=("on", "off"), default="on",
                              help="Sets BOTH matmul and cuDNN TF32. Torch's defaults differ "
                                   "between the two, which biases ViT against ResNet.")
    device_group.add_argument("--cudnn-benchmark", choices=("on", "off"), default="on")
    device_group.add_argument("--threads", type=int, default=os.cpu_count(),
                              help="CPU device only: torch.set_num_threads")

    timing = parser.add_argument_group("timing")
    timing.add_argument("--warmup-steps", type=int, default=10,
                        help="Absorbs cuDNN autotune, lazy allocator growth and optimizer state "
                             "creation; the first steps at a new shape can be 10x the steady state")
    timing.add_argument("--measure-steps", type=int, default=30)
    timing.add_argument("--repeats", type=int, default=1)
    timing.add_argument("--settle-seconds", type=float, default=0.0)
    timing.add_argument("--no-phase-breakdown", action="store_true")
    timing.add_argument("--dump-raw-steps", type=Path)

    memory = parser.add_argument_group("memory")
    memory.add_argument("--memory-steps", type=int, default=3)
    memory.add_argument("--skip-memory", action="store_true")

    loader = parser.add_argument_group("dataloader validation")
    loader.add_argument("--dataloader-mode", action="store_true")
    loader.add_argument("--data-path", default="/SCRATCH/datasets/imagenet")
    loader.add_argument("--workers", type=int, default=10)
    loader.add_argument("--dataloader-warmup-steps", type=int, default=50)
    loader.add_argument("--dataloader-steps", type=int, default=100)
    loader.add_argument("--auto-augment", default="ra")
    loader.add_argument("--mixup-alpha", type=float, default=0.2)
    loader.add_argument("--cutmix-alpha", type=float, default=1.0)

    output = parser.add_argument_group("extrapolation and output")
    output.add_argument("--train-samples", type=int, default=IMAGENET_TRAIN_SAMPLES)
    output.add_argument("--segmentation-train-samples", type=int, default=CITYSCAPES_TRAIN_SAMPLES)
    output.add_argument("--epochs", type=int, default=120)
    output.add_argument("--output", type=Path, default=Path("training_cost.csv"))
    output.add_argument("--append", action="store_true")
    output.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    output.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    points, indices = iter_configs(args)

    if args.list_configs:
        for index, point in zip(indices, points):
            print(f"{index:3d} {point.family:7s} {point.scaling_axis:17s} {point.model_name:18s} "
                  f"r={point.resolution:4d} {spec_details(point.spec)}")
        print(f"{len(points)} configs")
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    environment = configure_backends(args, device)

    order = list(zip(indices, points))
    if args.shuffle_configs:
        random.Random(args.seed).shuffle(order)

    # The grid emits the same network under several axes (vit_layers_12, vit_hidden_768,
    # vit_mlp_3072 and vit_b_16@224 are one model). Measure once, reuse for every axis.
    measured = {}
    rows = []
    for index, point in order:
        for repeat_index in range(args.repeats):
            if args.settle_seconds:
                time.sleep(args.settle_seconds)
            key = (spec_signature(point.spec, point.resolution), repeat_index)
            if key in measured:
                row = dict(measured[key])
                row.update({
                    "scaling_axis": point.scaling_axis,
                    "scale_index": point.scale_index,
                    "model_name": point.model_name,
                    "config_index": index,
                })
                rows.append(row)
                continue
            try:
                row = benchmark_point(point, args, environment, index, repeat_index)
            except torch.OutOfMemoryError as error:
                release_memory(device)
                row = empty_row(point, args, environment, index, "oom", str(error)[:200])
            except RuntimeError as error:
                release_memory(device)
                if "out of memory" not in str(error).lower():
                    raise
                row = empty_row(point, args, environment, index, "oom", str(error)[:200])
            measured[key] = row
            rows.append(row)
            print(f"[{index:3d}] {point.family:7s} {point.model_name:18s} r={point.resolution:4d} "
                  f"{row['status']:5s} step={row['step_time_ms_median'] or '-'} ms "
                  f"peak={row['peak_allocated_mb'] or '-'} MB", flush=True)

    write_rows(rows, args.output, args.append)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
