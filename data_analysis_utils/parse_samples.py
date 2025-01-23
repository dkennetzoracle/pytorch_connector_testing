#!/usr/bin/env python3

import argparse
import re
from datetime import datetime
from collections import defaultdict
import sys

def get_args():
    parser = argparse.ArgumentParser(description="parse log file datetime / sample stats")
    parser.add_argument("-i", "--input_file", required=True, help="input log file to parse.")
    parser.add_argument("-l", "--logger", choices=["ocifs", "streaming"], default="streaming")
    return parser.parse_args()

# Parse log lines
def parse_logs(log_file, log_pattern):
    samples = []
    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        log_lines = f.readlines()
        for line in log_lines:
            line = line.strip()
            match = log_pattern.search(line)
            if match:
                samples.append({
                    "timestamp": datetime.strptime(match.group("timestamp"), "%Y-%m-%d %H:%M:%S"),
                    "sample_id": int(match.group("sample_id")),
                    "rank": int(match.group("rank")),
                    "sample_length": int(match.group("sample_length")),
                    "url_length": int(match.group("url_length")),
                    "timestamp_len": int(match.group("timestamp_len"))
                })
    if len(samples) == 0:
        sys.exit("No samples found in regex capture.")
    return samples


def analyze_samples(samples):
    total_data_transferred = 0
    time_deltas = []
    last_timestamp = None
    total_samples = 0
    rank_data = defaultdict(int)  # Store data transferred per rank

    for sample in samples:
        total_samples += 1
        total_data_transferred += sample["sample_length"]
        total_data_transferred += sample["url_length"]
        total_data_transferred += sample["timestamp_len"]
        rank_data[sample["rank"]] += sample["sample_length"]
        rank_data[sample["rank"]] += sample["url_length"]
        rank_data[sample["rank"]] += sample["timestamp_len"]

        if last_timestamp:
            time_deltas.append((sample["timestamp"] - last_timestamp).total_seconds())
        last_timestamp = sample["timestamp"]

    # Overall time range
    total_time = (samples[-1]["timestamp"] - samples[0]["timestamp"]).total_seconds()

    # Data transfer rate (bytes/sec)
    data_transfer_rate = total_data_transferred / total_time if total_time > 0 else 0

    # Average time between samples
    avg_time_between_samples = sum(time_deltas) / len(time_deltas) if time_deltas else 0

    return {
        "Total Samples": total_samples,
        "Total Data Transferred (bytes)": total_data_transferred,
        "Data Transfer Rate (bytes/sec)": data_transfer_rate,
        "Average Time Between Samples (sec)": avg_time_between_samples,
        "Rank Data Transferred (bytes)": rank_data,
    }

def main():
    args = get_args()
    LOG_PATTERN = re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - " + re.escape(args.logger) + r" - INFO - Sample: "
        r"(?P<sample_id>\d+), rank: (?P<rank>\d+), sample_length: (?P<sample_length>\d+), "
        r"url_length: (?P<url_length>\d+), timestamp_len: (?P<timestamp_len>\d+)"
    )
    samples = parse_logs(args.input_file, LOG_PATTERN)
    stats = analyze_samples(samples)
    print(f"{stats=}")

if __name__ == "__main__":
    main()