#!/bin/bash

#SBATCH --job-name=lixiang_taste_toys
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err # I put this in directory `outputs`, if the directory doesn't exists, job will fail immediately

#SBATCH --partition=long # check the partitions available and switch if you need a longer job/ different resources 

#SBATCH --mail-type=ALL
#SBATCH --mail-user=lixiangl@andrew.cmu.edu

#SBATCH --gres=gpu:v100:8
#SBATCH --time=23:50:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task 16
#SBATCH --nodes=1

# enter a config env
eval "$(conda shell.bash hook)"
conda activate taste_run




experiment_run="toys"
report_dir="toys_proposed_pretrain_0_test"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py  \
    --output_dir "checkpoint/${report_dir}/name"  \
    --model_name_or_path "t5_toys_best_dev"  \
    --do_train  \
    --save_steps 200  \
    --eval_steps 200  \
    --train_path "data/${experiment_run}/train_name.jsonl"  \
    --eval_path "data/${experiment_run}/valid_name.jsonl"  \
    --per_device_train_batch_size 256  \
    --per_device_eval_batch_size 256  \
    --train_n_passages 10  \
    --num_passages 2  \
    --learning_rate 2e-4  \
    --q_max_len 64  \
    --p_max_len 64  \
    --seed 2022  \
    --warmup_ratio 0.1 \
    --num_train_epochs 16  \
    --evaluation_strategy steps  \
    --report_to tensorboard \
    --logging_dir "checkpoint/${report_dir}/name-log" \
    --logging_steps 100
