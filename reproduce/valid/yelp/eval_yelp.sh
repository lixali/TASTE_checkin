#!/bin/sh

CUDA_VISIBLE_DEVICES=2 python evaluate.py  \
    --data_name yelp  \
    --experiment_name address \
    --seed 2022  \
    --item_size 32  \
    --seq_size 256  \
    --num_passage 2  \
    --split_num 243  \
    --eval_batch_size 512  \
    --stopping_step 5  \
    --all_models_path ../TASTE/checkpoint/yelp/address
