#!/usr/bin/env bash

accelerate launch \
    --config_file configs/fsdp_config.yaml \
    train.py \
    --seed 100 \
    --model_name "meta-llama/Llama-2-70b-chat-hf" \
    --dataset_name "smangrul/code-chat-assistant-v1" \
    --chat_template_format "none" \
    --add_special_tokens False \
    --append_concat_token False \
    --splits "train,test" \
    --max_seq_len 2048 \
    --max_steps 500 \
    --logging_steps 25 \
    --log_level "info" \
    --eval_steps 100 \
    --save_steps 250 \
    --logging_strategy "steps" \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --bf16 True \
    --packing True \
    --learning_rate 5e-5 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --output_dir "/mnt/nvme/datasets/finetune_mosaic_ml" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --use_reentrant False \
    --dataset_text_field "content" \
    --use_flash_attn True \
    --ddp_timeout 5400 \
    --optim paged_adamw_32bit \
    --world_size 16 \
    --local_world_size 8 \
    --rank 0 \
    --master_ip_addr 10.140.19.72 \
    --master_port 12355
