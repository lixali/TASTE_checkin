#!/bin/sh

experiment_name="/data/datasets/hf_cache/sample/synthetic_taste_last_item_finetune_T5_fusin_1/beauty_8_gpus_batch_160_save_step_600_fusin_1/name/"

echo $experiment_name
CUDA_VISIBLE_DEVICES=0 python evaluate.py  \
    --data_dir data \
    --data_name beauty  \
    --experiment_name name \
    --seed 2022  \
    --item_size 32  \
    --seq_size 256  \
    --num_passage 2  \
    --split_num 243  \
    --eval_batch_size 128 \
    --stopping_step 100  \
    --all_models_path $experiment_name \
    --checkpoint_dir $experiment_name


echo $experiment_name
