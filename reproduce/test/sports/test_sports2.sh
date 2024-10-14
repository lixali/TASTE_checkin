#!/bin/sh

model_dir="/data/datasets/hf_cache/sample/TASTE_benchmark_run/sports/name/best_dev/"

echo "${model_dir}"
CUDA_VISIBLE_DEVICES=0 python inference2.py  \
    --data_name sports  \
    --experiment_name name  \
    --seed 2022  \
    --item_size 32  \
    --seq_size 256  \
    --num_passage 2  \
    --split_num 243  \
    --eval_batch_size 128  \
    --best_model_path ${model_dir} \
    --checkpoint_dir ${model_dir}


