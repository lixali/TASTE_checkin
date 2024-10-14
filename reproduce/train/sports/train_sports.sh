#!/bin/sh


step=600
fusin=1
report_dir="sports"
output_dir="TASTE_benchmark/sports_8_gpus_batch_160_save_step_${step}_fusin_${fusin}_50_percent"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py  \
    --output_dir checkpoint/${output_dir}/name  \
    --resume_from_checkpoint checkpoint/${output_dir}/name \
    --model_name_or_path pretrained_model/t5-base  \
    --do_train  \
    --save_steps ${step}  \
    --eval_steps ${step}  \
    --train_path data/${report_dir}/train_name_50_percent.jsonl  \
    --eval_path data/${report_dir}/valid_name.jsonl  \
    --per_device_train_batch_size 160  \
    --per_device_eval_batch_size 160  \
    --train_n_passages 10  \
    --num_passages 2  \
    --learning_rate 2e-4  \
    --q_max_len 256  \
    --p_max_len 32  \
    --seed 2022  \
    --num_train_epochs 19  \
    --evaluation_strategy steps  \
    --logging_steps 100 \
    --logging_dir checkpoint/${output_dir}/name-log
    # >> ../TASTE/logs/rep_sports_t5base.out


# bash reproduce/train/sports/train_sports.sh
