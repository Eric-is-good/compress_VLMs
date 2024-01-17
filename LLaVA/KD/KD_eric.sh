#!/bin/bash

deepspeed KD/try_DS_KD.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ../ft/ \
    --version v1 \
    --data_path ../llava_v1_5_mix665k.json \
    --image_folder ../ \
    --vision_tower ../clip \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --output_dir ../save \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps  100 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    # --tf32 True \
    # --bf16 True \
    # --pretrain_mm_mlp_adapter /opt/data/private/eric/save_model/llava/pretrain/mm_projector.bin 
