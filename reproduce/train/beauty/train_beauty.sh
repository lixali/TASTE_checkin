#!/bin/zsh

#step=600
step=600
fusin=1
local_path="data_taste_synthetic_new_flow/beauty"
output_dir="checkpoint/synthetic_taste_last_item_finetune_T5_fusin_1/beauty_8_gpus_batch_160_save_step_600_fusin_${fusin}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py  \
    --output_dir "${output_dir}/name"  \
    --cache_dir "${output_dir}/cache"  \
    --model_name_or_path pretrained_model/t5-base  \
    --resume_from_checkpoint  "${output_dir}/name/" \
    --do_train \
    --save_steps ${step}  \
    --eval_steps ${step}  \
    --train_path "${local_path}/train_name_fusin_0.jsonl"  \
    --eval_path "${local_path}/valid_name_fusin_0.jsonl"  \
    --per_device_train_batch_size 160  \
    --per_device_eval_batch_size 160  \
    --train_n_passages 10  \
    --num_passages 2  \
    --learning_rate 2e-4  \
    --warmup_ratio 0.1 \
    --q_max_len 256  \
    --p_max_len 32  \
    --seed 2022  \
    --num_train_epochs 16  \
    --evaluation_strategy steps  \
    --logging_steps 100 \
    --logging_dir "${output_dir}/name-log"
    #&> ../TASTE/logs/beauty_t5_base.out

# eval examples: 22363

# --resume_from_checkpoint  "${output_dir}/name/checkpoint-70000" \x
