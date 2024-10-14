#!/bin/bash



report_dir="toys"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py  \
    --output_dir "checkpoint/${report_dir}/name"  \
    --model_name_or_path pretrained_model/t5-base  \
    --do_train  \
    --save_steps 300  \
    --eval_steps 300  \
    --train_path data/toys/train_name.jsonl  \
    --eval_path data/toys/valid_name.jsonl  \
    --per_device_train_batch_size 256  \
    --per_device_eval_batch_size 256  \
    --train_n_passages 10  \
    --num_passages 2  \
    --learning_rate 2e-4  \
    --q_max_len 256  \
    --p_max_len 32  \
    --seed 2022  \
    --warmup_ratio 0.1 \
    --num_train_epochs 20  \
    --evaluation_strategy steps  \
    --report_to tensorboard \
    --logging_dir "checkpoint/${report_dir}/name-log" \
    --logging_steps 100
