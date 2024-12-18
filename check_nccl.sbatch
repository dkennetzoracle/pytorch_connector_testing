#!/bin/bash
#SBATCH --job-name=nccl-check
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96

#set -x -e

source ~/miniconda3/bin/activate
conda activate py310
source ~/pytorch_connector_testing/venv/bin/activate

#export NCCL_ASYNC_ERROR_HANDLING=1
#export NCCL_DEBUG=INFO

echo "START TIME: $(date)"

LOG_PATH="main_log.txt"

GPUS_PER_NODE=8
NNODES=2
NUM_PROCS=16

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

export LAUNCHER="accelerate launch \
  --config_file check_nccl.yaml \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --machine_rank \$SLURM_PROCID \
  --num_processes $NUM_PROCS \
  --num_machines $NNODES \
  "

export PROGRAM="check_nccl.py"

export CMD="$LAUNCHER $PROGRAM"

srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_PATH

echo "END TIME: $(date)"
