import argparse
from collections import defaultdict
import json
import os
from statistics import mean, median
import sys


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Get stats from torch profiler.")
    parser.add_argument("-i", "--input-file", help="input jsonl file for parsing.")
    parser.add_argument("-d", "--directory", help="input directory containing all jsonl outputs from profiler.")
    return parser.parse_args()

def process_events2(event_file):
    # Load the JSON data
    with open(event_file, 'r') as f:
        data = json.load(f)

    rank = data.get("distributedInfo", None).get("rank", None)
    if rank is None:
        sys.exit("No rank found. Cannot compute")
    trace_events = data.get("traceEvents", [])
    if not trace_events:
        sys.exit("No trace events found. Please run profiler.")

    # Initialize variables to track kernel wait time, active time, and sync overhead
    total_kernel_active_time = 0
    total_kernel_wait_time = 0
    total_sync_overhead = 0

    # Map external IDs to timestamps for easier analysis
    launch_timestamps = {}
    kernel_timestamps = {}

    # Process events
    event_names = set()
    for event in trace_events:
        event_type = event.get('ph')
        category = event.get('cat', '')
        name = event.get('name', '')
        event_names.add(name)
        timestamp = event.get('ts', 0)
        duration = event.get('dur', 0)
        external_id = external_id = event.get('args', {}).get('External id')

        # Track cudaLaunchKernel (kernel launch) timestamps
        if category == "cuda_runtime" and name == "cudaLaunchKernel" and external_id is not None:
            launch_timestamps[external_id] = timestamp

        # Track kernel execution timestamps and durations
        if category == "kernel" and external_id is not None:
            kernel_timestamps[external_id] = (timestamp, duration)
            total_kernel_active_time += duration

        # Track cudaEventSynchronize durations for synchronization overhead
        if category == "cuda_runtime" and name == "cudaEventSynchronize":
            total_sync_overhead += duration

    # Calculate wait times
    for external_id, launch_ts in launch_timestamps.items():
        if external_id in kernel_timestamps:
            kernel_ts, kernel_duration = kernel_timestamps[external_id]
            wait_time = kernel_ts - launch_ts
            if wait_time < 0:
                wait_time = 0
            total_kernel_wait_time += wait_time

    print(f"{event_names=}")
    return rank, {
        "Kernel Active Time (μs)": total_kernel_active_time,
        "Kernel Wait Time (μs)": total_kernel_wait_time,
        "Synchronization Overhead (μs)": total_sync_overhead
    }

def process_events(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    rank = data.get("distributedInfo", None).get("rank", None)
    if rank is None:
        sys.exit("No rank found. Cannot compute")
    trace_events = data.get("traceEvents", [])
    if not trace_events:
        sys.exit("No trace events found. Please run profiler.")


    kernel_exec_time = 0
    kernel_wait_time = 0
    kernel_count = 0
    sync_time = 0
    occupancies = []  # To collect "est. achieved occupancy %" values
    query_time = 0

    # Organize events by correlation ID
    sched_launch_start = {}
    sched_launch_end = {}
    kernel_external_ids = set()

    for event in trace_events:
        # Track synchronization events
        ph = event.get("ph", "")
        cat = event.get("cat", "")
        id = event.get("id", None)
        ts = event.get("ts")
        name = event.get("name", "")
        dur = event.get("dur", 0.0)
        args = event.get("args", None)
        if ph == "X" and cat == "kernel":
            kernel_exec_time += dur
            kernel_count += 1
            if args is not None:
                ex_id = args.get("External id")
                occ = args.get("est. achieved occupancy %")
                kernel_external_ids.add(ex_id)
                occupancies.append(occ)
        elif ph == "f" and cat == "ac2g" and id in kernel_external_ids:
            sched_launch_end[id] = ts
        elif ph == "s" and cat == "ac2g" and id in kernel_external_ids:
            sched_launch_start[id] = ts
        elif ph == "X" and cat == "cuda_runtime" and name == "cudaStreamSynchronize":
            sync_time += dur
        elif ph == "X" and cat == "cuda_runtime" and name in ("cudaEventSynchronize", "cudaStreamWaitEvent", "cudaStreamSynchronize", "cudaDeviceSynchronize"):
            query_time += dur

    # Compute schedule + launch overhead
    for id, end_ts in sched_launch_end.items():
        start_ts = sched_launch_start.get(id, 0)
        if start_ts != 0:
            kernel_wait_time += end_ts - start_ts

    # Calculate mean and median occupancy
    mean_occupancy = mean(occupancies) if occupancies else 0
    median_occupancy = median(occupancies) if occupancies else 0

    return rank, {
        "Kernel Active Time (μs)": kernel_exec_time,
        "Kernel Wait Time (μs)": kernel_wait_time,
        "Synchronization Overhead (μs)": sync_time,
        "Event Polling time (μs)": query_time,
        "Kernels Launched": kernel_count,
        "Mean Occupancy (%)": mean_occupancy,
        "Median Occupancy (%)": median_occupancy
    }


def main():
    args = get_args()
    
    if args.input_file and args.directory:
        sys.exit("Run only on single input file or directory, not both")
    if args.input_file:
        events = process_events(args.input_file)
        print(f"Rank: {events}")
        return
    if not args.directory:
        sys.exit("Must provide an input file or directory.")
    
    rank_stats = {}
    total_stats = {
        "Kernel Active Time (s)": 0,
        "Kernel Wait Time (s)": 0,
        "Synchronization Overhead (s)": 0,
        "Event Polling time (s)": 0,
        "Kernels Launched": 0,
        "Mean Occupancy (%)": 0,
        "Median Occupancy (%)": 0
    }
    for f in os.listdir(args.directory):
        if f.endswith('.json'):
            fpath = os.path.join(args.directory, f)
            rank, stats = process_events(fpath)

            rank_stats[rank] = stats
            total_stats["Kernel Active Time (s)"] += (stats["Kernel Active Time (μs)"] / 1e6)
            total_stats["Kernel Wait Time (s)"] += (stats["Kernel Wait Time (μs)"] / 1e6)
            total_stats["Synchronization Overhead (s)"] += (stats["Synchronization Overhead (μs)"] / 1e6)
            total_stats["Event Polling time (s)"] += (stats["Event Polling time (μs)"] / 1e6)
            total_stats["Kernels Launched"] += stats["Kernels Launched"]
    
    print("Per Rank Stats:")
    for rank, stats in rank_stats.items():
        print(f"Rank {rank}: {stats}")
    
    print("\nTotal Stats:")
    print(total_stats)


if __name__ == "__main__":
    main()