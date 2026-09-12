"""Join measured training cost with the analytic model and draw the chapter figures.

Inputs:
  - measured rows from ``benchmark_training_step.py`` (one CSV, or a directory of them)
  - the analytic table from ``scaling_resources.py`` (``scaling_resources.csv``)
  - optionally the real-epoch table from ``collect_scaling_training_times.py``

The join key is ``(family, model_name, resolution)``, which is why both grids are enumerated
from ``scaling_specs.iter_scaling_points``.
"""

import argparse
import csv
import os
import statistics
from pathlib import Path


MIB = 1024**2
AXIS_MARKERS = {
    "model": "o",
    "model_depth": "v",
    "model_layers": "o",
    "model_hidden_dim": "^",
    "model_mlp_dim": "D",
    "resolution": "s",
}
MEMORY_TERMS = (
    ("param_bytes", "parameters"),
    ("grad_bytes", "gradients"),
    ("optimizer_state_bytes", "optimizer state"),
    ("peak_activation_bytes", "activations"),
)


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value):
    value = to_float(value)
    return None if value is None else int(value)


def load_csv(path):
    with Path(path).open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_measured(paths):
    rows = []
    for path in paths:
        path = Path(path)
        files = sorted(path.glob("**/*.csv")) if path.is_dir() else [path]
        for file in files:
            rows.extend(load_csv(file))
    return [row for row in rows if row.get("status") == "ok"]


def join_key(row):
    return (row["family"], row["model_name"], to_int(row["resolution"]))


def reduce_repeats(rows):
    """Collapse repeated measurements of the same point to a median-of-medians."""
    grouped = {}
    for row in rows:
        key = (join_key(row), row["device"], row["precision"], row["batch_size"],
               row["scaling_axis"], row["input_mode"])
        grouped.setdefault(key, []).append(row)

    reduced = []
    for group in grouped.values():
        base = dict(group[0])
        if len(group) > 1:
            for column in ("step_time_ms_median", "step_time_ms_batched_mean",
                           "peak_allocated_bytes", "peak_activation_bytes"):
                values = [to_float(row[column]) for row in group if to_float(row[column]) is not None]
                if values:
                    base[column] = statistics.median(values)
            base["repeats_collapsed"] = len(group)
        reduced.append(base)
    return reduced


def join_analytic(rows, analytic_rows):
    """Attach the analytic FLOPs/memory estimate and the ratios derived from it."""
    analytic = {}
    for row in analytic_rows:
        analytic.setdefault(join_key(row), row)

    joined = []
    for row in rows:
        merged = dict(row)
        match = analytic.get(join_key(row))
        if match is None:
            merged["analytic_matched"] = False
            joined.append(merged)
            continue

        forward_flops = to_float(match["forward_flops_per_sample"])
        training_flops = to_float(match["training_flops_per_sample"])
        analytic_activation_mb = to_float(match["activation_memory_mb_per_sample"])
        merged.update({
            "analytic_matched": True,
            "forward_flops_per_sample": forward_flops,
            "training_flops_per_sample": training_flops,
            "analytic_activation_mb_per_sample": analytic_activation_mb,
            "analytic_training_memory_mb": to_float(match["estimated_training_memory_mb"]),
            "analytic_forward_flops_ratio": to_float(match["forward_flops_ratio"]),
            "analytic_training_memory_ratio": to_float(match["training_memory_ratio"]),
        })

        step_seconds = to_float(row["step_time_ms_batched_mean"])
        batch_size = to_int(row["batch_size"])
        if step_seconds and training_flops and batch_size:
            step_seconds /= 1000.0
            merged["achieved_tflops"] = round(training_flops * batch_size / step_seconds / 1e12, 4)
            merged["step_ms_per_training_gflop"] = round(
                to_float(row["step_time_ms_batched_mean"]) / (training_flops * batch_size / 1e9), 6
            )

        # The analytic model multiplies inference activations by a fixed
        # --activation-training-multiplier (default 2.0). This is the measured value of it.
        measured_activation = to_float(row.get("peak_activation_bytes"))
        if measured_activation and analytic_activation_mb and batch_size:
            merged["fitted_activation_training_multiplier"] = round(
                measured_activation / MIB / (analytic_activation_mb * batch_size), 4
            )
        joined.append(merged)
    return joined


def add_ratios(rows, baseline_resolution):
    """Normalise every axis against its family baseline, matching scaling_resources.csv."""
    baselines = {}
    for row in rows:
        if row["scaling_axis"] == "resolution" and to_int(row["resolution"]) == baseline_resolution:
            baselines[(row["family"], row["device"], row["precision"])] = row

    for row in rows:
        baseline = baselines.get((row["family"], row["device"], row["precision"]))
        if baseline is None:
            continue
        for column, name in (("step_time_ms_batched_mean", "measured_time_ratio"),
                             ("peak_allocated_bytes", "measured_memory_ratio"),
                             ("peak_activation_bytes", "measured_activation_ratio")):
            value, reference = to_float(row.get(column)), to_float(baseline.get(column))
            if value is not None and reference:
                row[name] = round(value / reference, 6)
    return rows


def write_csv(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------
def pyplot():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def families_in(rows):
    return [family for family in ("resnet", "vit", "regseg")
            if any(row["family"] == family for row in rows)]


def sorted_by(rows, column):
    return sorted(rows, key=lambda row: to_float(row.get(column)) or 0.0)


def figure_time_vs_flops(rows, output, device):
    """F1: does analytic FLOPs predict measured step time, and does it predict it the same way
    along the resolution axis as along the model axis?"""
    plt = pyplot()
    selected = [row for row in rows if row["device"].startswith(device) and row.get("analytic_matched")]
    families = families_in(selected)
    if not families:
        return None
    figure, axes = plt.subplots(1, len(families), figsize=(5.5 * len(families), 4.5), squeeze=False)

    for index, family in enumerate(families):
        axis = axes[0][index]
        family_rows = [row for row in selected if row["family"] == family]
        for scaling_axis in sorted({row["scaling_axis"] for row in family_rows}):
            points = sorted_by([row for row in family_rows if row["scaling_axis"] == scaling_axis],
                               "training_flops_per_sample")
            x = [to_float(row["training_flops_per_sample"]) / 1e9 for row in points]
            y = [to_float(row["step_time_ms_batched_mean"]) for row in points]
            axis.plot(x, y, marker=AXIS_MARKERS.get(scaling_axis, "o"), linewidth=1,
                      markersize=6, alpha=0.85, label=scaling_axis)
        axis.set_title(f"{family.upper()} step time vs analytic FLOPs")
        axis.set_xlabel("training FLOPs per sample (GFLOPs)")
        axis.set_ylabel("measured step time (ms)")
        axis.grid(True, linewidth=0.5, alpha=0.35)
        axis.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)
    return output


def figure_memory_breakdown(rows, output, device):
    """F2: the chapter's core claim. Resolution scaling moves only the activation term;
    width and depth scaling move parameters, gradients and optimizer state as well."""
    plt = pyplot()
    selected = [row for row in rows
                if row["device"].startswith(device) and to_float(row.get("peak_allocated_bytes"))]
    families = families_in(selected)
    if not families:
        return None

    panels = []
    for family in families:
        for scaling_axis in ("resolution", "model", "model_hidden_dim"):
            group = [row for row in selected
                     if row["family"] == family and row["scaling_axis"] == scaling_axis]
            if group:
                panels.append((family, scaling_axis, group))
                break
        model_axes = sorted({row["scaling_axis"] for row in selected
                             if row["family"] == family and row["scaling_axis"].startswith("model")})
        if model_axes:
            group = [row for row in selected
                     if row["family"] == family and row["scaling_axis"] == model_axes[0]]
            panels.append((family, model_axes[0], group))

    figure, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.5), squeeze=False)
    for index, (family, scaling_axis, group) in enumerate(panels):
        axis = axes[0][index]
        key = "resolution" if scaling_axis == "resolution" else "parameters"
        group = sorted_by(group, key)
        labels = [row["resolution"] if scaling_axis == "resolution" else row["model_name"]
                  for row in group]
        bottom = [0.0] * len(group)
        for column, name in MEMORY_TERMS:
            values = [(to_float(row.get(column)) or 0.0) / MIB for row in group]
            axis.bar(range(len(group)), values, bottom=bottom, label=name)
            bottom = [base + value for base, value in zip(bottom, values)]
        axis.set_xticks(range(len(group)))
        axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        axis.set_title(f"{family.upper()} — {scaling_axis}")
        axis.set_ylabel("training memory (MiB)")
        axis.grid(True, axis="y", linewidth=0.5, alpha=0.35)
        axis.margins(y=0.18)
        axis.legend(fontsize=8, loc="upper left")

    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)
    return output


def figure_measured_vs_analytic(rows, output, device):
    """F3: calibrates the analytic model that the chapter's estimates rest on."""
    plt = pyplot()
    selected = [row for row in rows
                if row["device"].startswith(device)
                and to_float(row.get("peak_activation_bytes"))
                and to_float(row.get("analytic_activation_mb_per_sample"))]
    if not selected:
        return None

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for family in families_in(selected):
        family_rows = [row for row in selected if row["family"] == family]
        batch = [to_int(row["batch_size"]) for row in family_rows]
        analytic = [to_float(row["analytic_activation_mb_per_sample"]) * size
                    for row, size in zip(family_rows, batch)]
        measured = [to_float(row["peak_activation_bytes"]) / MIB for row in family_rows]
        axes[0].scatter(analytic, measured, s=36, alpha=0.8, label=family)

        multipliers = [to_float(row.get("fitted_activation_training_multiplier"))
                       for row in family_rows]
        multipliers = [value for value in multipliers if value is not None]
        resolutions = [to_int(row["resolution"]) for row in family_rows
                       if to_float(row.get("fitted_activation_training_multiplier")) is not None]
        axes[1].scatter(resolutions, multipliers, s=36, alpha=0.8, label=family)
        if multipliers:
            print(f"{family}: fitted activation training multiplier "
                  f"median={statistics.median(multipliers):.3f} "
                  f"min={min(multipliers):.3f} max={max(multipliers):.3f}")

    limit = max(axes[0].get_xlim()[1], axes[0].get_ylim()[1])
    axes[0].plot([0, limit], [0, limit], linestyle="--", linewidth=1, color="grey", label="y = x")
    axes[0].set_xlabel("analytic inference activation memory per batch (MiB)")
    axes[0].set_ylabel("measured peak training activation memory (MiB)")
    axes[0].set_title("Measured vs analytic activation memory")
    axes[1].axhline(2.0, linestyle="--", linewidth=1, color="grey",
                    label="analytic default (2.0)")
    axes[1].set_xlabel("input resolution (px)")
    axes[1].set_ylabel("fitted activation training multiplier")
    axes[1].set_title("Calibration of the analytic multiplier")
    for axis in axes:
        axis.grid(True, linewidth=0.5, alpha=0.35)
        axis.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)
    return output


def figure_device_comparison(rows, output):
    """F4: FLOPs do not predict latency identically across hardware, the point the chapter makes
    with the MobileViT/MobileNetV2 CPU figure. Same models, GPU and CPU, this thesis's own data."""
    plt = pyplot()
    selected = [row for row in rows if row["scaling_axis"] == "resolution"]
    devices = sorted({row["device"].split(":")[0] for row in selected})
    if len(devices) < 2:
        return None
    families = families_in(selected)
    figure, axes = plt.subplots(1, len(families), figsize=(5.5 * len(families), 4.5), squeeze=False)

    for index, family in enumerate(families):
        axis = axes[0][index]
        for device in devices:
            points = sorted_by([row for row in selected if row["family"] == family
                                and row["device"].split(":")[0] == device], "resolution")
            if not points:
                continue
            reference = next((to_float(row["step_time_ms_batched_mean"]) for row in points
                              if to_int(row["resolution"]) == 224), None)
            if not reference:
                reference = to_float(points[-1]["step_time_ms_batched_mean"])
            axis.plot([to_int(row["resolution"]) for row in points],
                      [to_float(row["step_time_ms_batched_mean"]) / reference for row in points],
                      marker="o", linewidth=1, label=device)
        axis.set_title(f"{family.upper()} normalised step time")
        axis.set_xlabel("input resolution (px)")
        axis.set_ylabel("step time relative to 224px")
        axis.grid(True, linewidth=0.5, alpha=0.35)
        axis.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)
    return output


def figure_epoch_validation(rows, epoch_rows, output):
    """F5: does the synthetic step time predict the real epoch time, and where does the
    dataloader-bound regime begin?"""
    plt = pyplot()
    if not epoch_rows:
        return None
    predicted_by_key = {}
    for row in rows:
        if row["input_mode"] == "synthetic" and row["device"].startswith("cuda"):
            predicted_by_key[(row["family"], to_int(row["resolution"]))] = row

    x, y, labels = [], [], []
    for row in epoch_rows:
        key = (row.get("family"), to_int(row.get("image_size")))
        predicted = predicted_by_key.get(key)
        measured = to_float(row.get("measured_epoch_seconds"))
        if predicted is None or measured is None:
            continue
        x.append(to_float(predicted["estimated_epoch_seconds"]))
        y.append(measured)
        labels.append(f"{key[0]} {key[1]}px")
    if not x:
        return None

    figure, axis = plt.subplots(figsize=(5.5, 5))
    axis.scatter(x, y, s=48, alpha=0.85)
    for xi, yi, label in zip(x, y, labels):
        axis.annotate(label, (xi, yi), fontsize=7, xytext=(4, 4), textcoords="offset points")
    limit = max(max(x), max(y)) * 1.05
    axis.plot([0, limit], [0, limit], linestyle="--", linewidth=1, color="grey",
              label="perfect prediction")
    axis.set_xlabel("epoch time predicted from synthetic step time (s)")
    axis.set_ylabel("measured ImageNet epoch time (s)")
    axis.set_title("Points above the line are dataloader bound")
    axis.grid(True, linewidth=0.5, alpha=0.35)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)
    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Join measured training cost with the analytic model and draw the figures."
    )
    parser.add_argument("measured", nargs="+", type=Path,
                        help="CSV files or directories produced by benchmark_training_step.py")
    parser.add_argument("--analytic", type=Path, default=Path("scaling_resources.csv"))
    parser.add_argument("--epoch-times", type=Path,
                        help="CSV from collect_scaling_training_times.py, for the F5 validation")
    parser.add_argument("--output", type=Path, default=Path("results/training_cost_joined.csv"))
    parser.add_argument("--figure-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--figure-format", default="pdf", choices=("pdf", "png"))
    parser.add_argument("--baseline-resolution", type=int, default=224)
    parser.add_argument("--plot-device", default="cuda",
                        help="Device prefix used for the single-device figures")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = reduce_repeats(load_measured(args.measured))
    if not rows:
        raise SystemExit("no successful measured rows found")

    analytic_rows = load_csv(args.analytic) if args.analytic.exists() else []
    if not analytic_rows:
        print(f"warning: {args.analytic} not found, skipping the analytic join")
    rows = add_ratios(join_analytic(rows, analytic_rows), args.baseline_resolution)

    unmatched = [row for row in rows if not row.get("analytic_matched")]
    if unmatched:
        print(f"warning: {len(unmatched)} measured rows had no analytic match "
              f"(regenerate scaling_resources.csv with the same grid flags)")

    write_csv(rows, args.output)
    print(f"wrote {len(rows)} joined rows to {args.output}")

    if args.no_figures:
        return

    epoch_rows = load_csv(args.epoch_times) if args.epoch_times and args.epoch_times.exists() else []
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.figure_format
    written = [
        figure_time_vs_flops(rows, args.figure_dir / f"f1_step_time_vs_flops.{suffix}", args.plot_device),
        figure_memory_breakdown(rows, args.figure_dir / f"f2_memory_breakdown.{suffix}", args.plot_device),
        figure_measured_vs_analytic(rows, args.figure_dir / f"f3_measured_vs_analytic.{suffix}", args.plot_device),
        figure_device_comparison(rows, args.figure_dir / f"f4_device_comparison.{suffix}"),
        figure_epoch_validation(rows, epoch_rows, args.figure_dir / f"f5_epoch_validation.{suffix}"),
    ]
    for path in written:
        print(f"wrote {path}" if path else "skipped a figure (not enough data)")


if __name__ == "__main__":
    main()
