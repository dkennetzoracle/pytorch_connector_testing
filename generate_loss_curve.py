#!/usr/bin/env python3

import argparse
import re
import sys

import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description="Overlay loss from each run")
    parser.add_argument("-i", "--inputs", required=True, nargs='+', help="input files with loss info")
    parser.add_argument("-k", "--keys", required=True, nargs="+", help="keys for loss curves")
    parser.add_argument("-v", "--values", required=True, nargs="+", help="train_loss final values")
    parser.add_argument("-o", "--output", default="loss.png", help="output image file")
    return parser.parse_args()

def extract_loss(line):
    # Regular expression to match "loss": followed by a number
    match = re.search(r"'loss':\s*([\d\.eE+-]+)", line)
    if match:
        return float(match.group(1))
    return None

def extract_step(line):
    match = re.search(r'\b(\d+)/(\d+)\b', line)
    if match:
        current_step = int(match.group(1))
        total_steps = int(match.group(2))
        return current_step, total_steps
    return None, None

def parse_loss(infile, key: str) -> dict:
    loss_dict = {key: [[],[]]}
    last_step_found = 0
    with open(infile) as f:
        for line in f:
            loss = extract_loss(line)
            current_step, total_steps = extract_step(line)
            if current_step:
                last_step_found = current_step
            if loss:
                if current_step is None:
                    current_step = last_step_found
                loss_dict[key][0].append(current_step)
                loss_dict[key][1].append(loss)
    return loss_dict


def main():
    args = get_args()
    if len(args.keys) != len(args.inputs):
        sys.exit("Must have one key per input")

    final_dict = {}
    train_loss_dict = {}
    for k, v in zip(args.keys, args.values):
        train_loss_dict[k] = float(v)
    for k, infile in zip(args.keys, args.inputs):
        final_dict.update(parse_loss(infile, k))

    plt.figure(figsize=(12, 8))
    for run_name, loss_values in final_dict.items():
        x = loss_values[0]
        y = loss_values[1]
        run_name_with_train_loss = f"{run_name}={train_loss_dict[run_name]}"
        plt.plot(x, y, label=run_name_with_train_loss, linewidth=2)
    
    plt.xlabel("Loss steps")
    plt.ylabel("Loss values")
    plt.title("Loss Curves for Fine-tuning Llama-3.1-70B with allenai/c4 en Runs")
    plt.legend(fontsize=12, loc="upper right")
    plt.grid(True)
    plt.savefig(args.output)
    plt.show()

if __name__ == "__main__":
    main()