#!/bin/sh

CUDA_VISIBLE_DEVICES=1 python train.py  \
    --output_dir ../TASTE/checkpoint/yelp/address  \
    --model_name_or_path ../TASTE/pretrained_model/t5_base  \
    --do_train  \
    --save_steps 10000  \
    --eval_steps 10000  \
    --train_path ../TASTE/data/yelp/train_address.jsonl  \
    --eval_path ../TASTE/data/yelp/valid_address.jsonl  \
    --per_device_train_batch_size 8  \
    --per_device_eval_batch_size 8  \
    --train_n_passages 10  \
    --num_passages 2  \
    --learning_rate 1e-4  \
    --q_max_len 256  \
    --p_max_len 32  \
    --seed 2022  \
    --num_train_epochs 30  \
    --evaluation_strategy steps  \
    --logging_dir ../TASTE/checkpoint/yelp/address-log
