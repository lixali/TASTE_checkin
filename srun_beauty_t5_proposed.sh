#!/bin/bash

#SBATCH --job-name=lixiang_taste_toys
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err # I put this in directory `outputs`, if the directory doesn't exists, job will fail immediately

#SBATCH --partition=general # check the partitions available and switch if you need a longer job/ different resources 

#SBATCH --mail-type=ALL
#SBATCH --mail-user=lixiangl@andrew.cmu.edu

#SBATCH --gres=gpu:A6000:4
#SBATCH --time=4:50:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task 16
#SBATCH --nodes=1

# enter a config env
eval "$(conda shell.bash hook)"
conda activate taste_run


batchsize=160
step=300
fusin=0
sequence_max_length=3
local_path="data_proposed_multiple_lengths/beauty"
#local_path="data_proposed_with_pool_negatives/beauty_new_best/"
#output_dir="checkpoint/pretrain_T5/beauty_new_best_with_pool_negatives_proposed_pretrain_4_gpus_batch_160_save_step_${step}_fusin_${fusin}_10_percent"
output_dir="checkpoint/pretrain_T5/beauty_new_best_sequence_length_${sequence_max_length}_proposed_pretrain_4_gpus_batch_${batchsize}_save_step_${step}_fusin_${fusin}_total_data_points"

CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py  \
    --output_dir "${output_dir}/name"  \
    --resume_from_checkpoint "${output_dir}/name"  \
    --cache_dir "${output_dir}/cache"  \
    --model_name_or_path "pretrained_model/t5-base"  \
    --do_train \
    --save_steps ${step}  \
    --eval_steps ${step}  \
    --train_path "${local_path}/train_name_length_${sequence_max_length}_total.jsonl"  \
    --eval_path "${local_path}/valid_name_length_${sequence_max_length}_total.jsonl"  \
    --per_device_train_batch_size ${batchsize}  \
    --per_device_eval_batch_size ${batchsize}  \
    --train_n_passages 10  \
    --num_passages 1  \
    --learning_rate 2e-4  \
    --warmup_ratio 0.1 \
    --q_max_len 64  \
    --p_max_len 64  \
    --seed 2022  \
    --num_train_epochs 19  \
    --evaluation_strategy steps  \
    --logging_steps 100 \
    --logging_dir "${output_dir}/name-log"
    #&> ../TASTE/logs/beauty_t5_base.out
