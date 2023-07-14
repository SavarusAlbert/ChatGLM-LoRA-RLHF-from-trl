#!/bin/bash

LR=1e-4

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_train \
    --model_name_or_path THUDM/chatglm-6b \
    --output_dir output/chatglm-6b-lora-$PRE_SEQ_LEN-$LR \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_strategy "epoch" \
    --num_train_epochs 10 \
    --logging_steps 10 \
    --learning_rate $LR \
    --warmup_steps 20 \
    --training_args.weight_decay 0.0001