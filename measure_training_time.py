import argparse
import datetime
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate total training time from one logged training and evaluation epoch."
    )
    parser.add_argument("log", type=Path, help="TXT logger output produced by a training script")
    parser.add_argument("--epoch", type=int, help="Epoch to use; defaults to the first complete logged epoch")
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs to estimate; defaults to the epochs value stored in the log header",
    )
    return parser.parse_args()


def load_log(path):
    with path.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]

    if not records:
        raise ValueError(f"{path} is empty")

    return records[0], records[1:]


def find_epoch_measurement(records, requested_epoch=None):
    starts = {}

    for record in records:
        event = record.get("event")
        epoch = record.get("epoch")
        timestamp = record.get("timestamp")

        if event == "epoch_start" and epoch is not None and timestamp is not None:
            starts[epoch] = timestamp
        elif event == "epoch_end" and epoch in starts and timestamp is not None:
            if requested_epoch is None or epoch == requested_epoch:
                return epoch, starts[epoch], timestamp

    epoch_description = requested_epoch if requested_epoch is not None else "any"
    raise ValueError(
        f"No complete epoch_start/epoch_end pair with timestamps was found for epoch {epoch_description}"
    )


def format_duration(seconds):
    return str(datetime.timedelta(seconds=round(seconds)))


def main():
    args = parse_args()
    header, records = load_log(args.log)
    epoch, start_timestamp, end_timestamp = find_epoch_measurement(records, args.epoch)

    epoch_duration = end_timestamp - start_timestamp
    if epoch_duration < 0:
        raise ValueError(f"Epoch {epoch} has an end timestamp before its start timestamp")

    epochs = args.epochs
    if epochs is None:
        epochs = header.get("args", {}).get("epochs")
    if epochs is None:
        raise ValueError("The epoch count is missing from the log header; pass it with --epochs")
    if epochs < 0:
        raise ValueError("--epochs must be non-negative")

    estimated_duration = epoch_duration * epochs
    print(f"Measured epoch: {epoch}")
    print(f"One epoch (training + evaluation): {format_duration(epoch_duration)} ({epoch_duration:.2f} s)")
    print(f"Estimated training time for {epochs} epochs: {format_duration(estimated_duration)}")


if __name__ == "__main__":
    main()
