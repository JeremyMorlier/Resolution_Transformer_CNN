import argparse
import csv
from pathlib import Path

from measure_training_time import find_epoch_measurement, load_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect scaling training-time measurements into one CSV.")
    parser.add_argument("root", type=Path, help="OUTPUT_ROOT used by slurm_scripts/measure_scaling_training_time.sh")
    parser.add_argument("--output", type=Path, help="CSV output path; defaults to ROOT/scaling_training_times.csv")
    parser.add_argument("--epoch", type=int, default=0, help="Measured epoch to extract from each log")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, str]:
    config = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            config[key.strip()] = value.strip()
    return config


def collect_row(config_path: Path, epoch: int) -> dict[str, object]:
    config = load_config(config_path)
    log_file = Path(config["log_file"])
    header, records = load_log(log_file)
    measured_epoch, start_timestamp, end_timestamp = find_epoch_measurement(records, epoch)
    epoch_seconds = end_timestamp - start_timestamp
    estimated_epochs = int(config["estimated_epochs"])

    return {
        "family": config["family"],
        "scaling_axis": config["scaling_axis"],
        "config_name": config["config_name"],
        "image_size": int(config["image_size"]),
        "batch_size": int(config["batch_size"]),
        "workers": int(config["workers"]) if "workers" in config else "",
        "measured_epoch": measured_epoch,
        "measured_epoch_seconds": round(epoch_seconds, 6),
        "estimated_epochs": estimated_epochs,
        "estimated_total_seconds": round(epoch_seconds * estimated_epochs, 6),
        "run_name": header.get("run_name"),
        "log_file": str(log_file),
    }


def main() -> None:
    args = parse_args()
    output = args.output or args.root / "scaling_training_times.csv"
    rows = []

    for config_path in sorted(args.root.glob("**/scaling_config.txt")):
        rows.append(collect_row(config_path, args.epoch))

    if not rows:
        raise ValueError(f"No scaling_config.txt files found under {args.root}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
