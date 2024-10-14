#!/bin/sh

data="ml-1m"
report_dir="ml-1m"

echo "data from ${data}"
echo "report to ${report_dir}"

CUDA_VISIBLE_DEVICES=0 python train.py  \
    --output_dir "../TASTE/checkpoint/${report_dir}/name"  \
    --model_name_or_path ../TASTE/pretrained_model/t5_base  \
    --resume_from_checkpoint ../TASTE/checkpoint/${report_dir}/name \
    --cache_dir "../TASTE/checkpoint/${report_dir}/cache"  \
    --do_train  \
    --save_steps 5000 \
    --eval_steps 5000 \
    --train_path ../TASTE/data/${data}/valid_name.jsonl  \
    --eval_path ../TASTE/data/${data}/valid_name.jsonl  \
    --per_device_train_batch_size 8  \
    --per_device_eval_batch_size 8  \
    --train_n_passages 5  \
    --num_passages 2  \
    --learning_rate 1e-4  \
    --q_max_len 256  \
    --p_max_len 32  \
    --seed 2022  \
    --num_train_epochs 30  \
    --evaluation_strategy steps  \
    --logging_dir "../TASTE/checkpoint/${report_dir}/name-log"
