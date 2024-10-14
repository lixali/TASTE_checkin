#!/bin/sh

model_path="/data/datasets/hf_cache/sample/TASTE_benchmark_run/toys_4_gpus_batch_160_save_step_500_fusin_1/name/best_dev/"

CUDA_VISIBLE_DEVICES=0 python inference3.py  \
    --data_name toys  \
    --experiment_name name  \
    --seed 2022  \
    --item_size 32  \
    --seq_size 256  \
    --num_passage 2  \
    --split_num 243  \
    --eval_batch_size 256  \
    --best_model_path   ${model_path} \
    --checkpoint_dir  ${model_path}
