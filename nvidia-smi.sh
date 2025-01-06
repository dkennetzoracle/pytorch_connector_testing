#!/usr/bin/env bash

OUTFILE=$1

nvidia-smi --query-gpu=timestamp,index,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,clocks.current.sm,clocks.current.memory,power.draw,power.limit --format=csv -l 1 -f $OUTFILE
