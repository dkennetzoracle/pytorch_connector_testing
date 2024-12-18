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

echo "START TIME: $(date)"

LOG_PATH="main_log.txt"

GPUS_PER_NODE=8
NNODES=2
NUM_PROCS=16

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

export LAUNCHER="accelerate launch \
  --config_file fsdp_config.yaml \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --machine_rank \$SLURM_PROCID \
  --num_processes $NUM_PROCS \
  --num_machines $NNODES \
  "

export PROGRAM="\
~/pytorch_connector_testing/dataloading/finetuning_from_bucket/mosaic_ml_streaming/finetune_mosaic.py \
    --bucket_name mosaic_ml_allenai_c4_en_compressed \
    --seed 100 \
    --world_size 16 \
    --local_world_size 8 \
    --rank \$SLURM_PROCID \
    --master_ip_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --model_path /nfs/cluster/models/meta-llama/Llama-3.1-70B/ \
    --use_flash_attn True \
    --use_peft_lora True \
    --use_reentrant False \
    --log_file $LOG_PATH \
    --local_cache_path /nfs/cluster/dataset_cache \
    --batch_size 8 \
    --packing True \
    --dataset_text_field text \
    --max_seq_length 2048 \
    --splits train \
    --oci_config_path ~/.oci/config \
    --oci_profile DEFAULT \
    --local_cache_max_size_gbs 25gb \
    --logging_steps 25 \
    --log_level "info" \
    --eval_steps 100 \
    --save_steps 250 \
    --logging_strategy "steps" \
    --save_strategy "steps" \
    --bf16 True \
    --learning_rate 5e-5 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --output_dir /nfs/cluster/finetune_mosaic_ml \
    --max_steps 3250 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --optim paged_adamw_32bit \
    --dataloader_num_workers 1 \
    --dataloader_persistent_workers True"

export CMD="$LAUNCHER $PROGRAM"

srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_PATH

echo "END TIME: $(date)"