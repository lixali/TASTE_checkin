#!/bin/sh

report_dir="books"
CUDA_VISIBLE_DEVICES=2,3,4,5 python train.py  \
    --output_dir "../TASTE/checkpoint/${report_dir}/name"  \
    --model_name_or_path ../TASTE/pretrained_model/t5_base  \
    --cache_dir "../TASTE/checkpoint/${report_dir}/cache"  \
    --do_train  \
    --save_steps 1000  \
    --eval_steps 1000  \
    --train_path ../TASTE/data/books/train_name.jsonl  \
    --eval_path ../TASTE/data/books/valid_name.jsonl  \
    --per_device_train_batch_size 64  \
    --per_device_eval_batch_size 64  \
    --train_n_passages 10  \
    --num_passages 2  \
    --learning_rate 1e-4  \
    --q_max_len 256  \
    --p_max_len 32  \
    --seed 2022  \
    --num_train_epochs 30  \
    --evaluation_strategy steps  \
    --logging_dir "../TASTE/checkpoint/${report_dir}/name-log"
