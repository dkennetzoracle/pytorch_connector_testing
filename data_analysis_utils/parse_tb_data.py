#!/usr/bin/env python3

import argparse
from tensorboard.backend.event_processing import event_accumulator

def get_args():
    parser = argparse.ArgumentParser(description="play with tensorboard data.")
    parser.add_argument("-i", "--input_dir", required=True,
                        help="path to tensorboard directory.")
    return parser.parse_args()

def main():
    args = get_args()
    ea = event_accumulator.EventAccumulator(
        args.input_dir,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    print(ea.Tags())

if __name__ == "__main__":
    main()