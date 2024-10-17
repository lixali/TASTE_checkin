#!/bin/sh

model_path="/home/lili2307338/meta/TASTE_benchmark_run/sports/name/best_dev/"

CUDA_VISIBLE_DEVICES=0 python3 build_prompt.py  \
    --data_name sports  \
    --experiment_name name  \
    --seed 2022  \
    --item_size 32  \
    --seq_size 256  \
    --num_passage 2  \
    --split_num 243  \
    --eval_batch_size 256  \
    --best_model_path   ${model_path} \
    --checkpoint_dir  ${model_path} \
