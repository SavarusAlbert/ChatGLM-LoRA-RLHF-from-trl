#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python rlhf.py \
    --do_train \
    --model_name None \
    --output_dir path_to_ppo_checkpoint \
    --mini_batch_size 1 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --early_stopping None \
    --target_kl 0.1 \
    --warmup_steps 5 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1.45e-5 \
    --ppo_epochs 1