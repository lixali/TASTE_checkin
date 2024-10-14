#!/bin/bash

#SBATCH --job-name=lixiang_taste_beauty_finetune_pretrain_best_dev
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err # I put this in directory `outputs`, if the directory doesn't exists, job will fail immediately

#SBATCH --partition=general # check the partitions available and switch if you need a longer job/ different resources 

#SBATCH --mail-type=ALL
#SBATCH --mail-user=lixiangl@andrew.cmu.edu

#SBATCH --gres=gpu:4
#SBATCH --time=8:50:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task 16
#SBATCH --nodes=1

# enter a config env
eval "$(conda shell.bash hook)"
conda activate taste_run

fusin=1
step=600
batchsize=160
local_path="data/beauty"
maxsequencelength=3
#local_path="data_taste_synthetic/beauty"
#output_dir="checkpoint/proposed_with_finetune_T5_with_pool_negatives/beauty_new_best_8_gpus_batch_${batchsize}_save_step_${step}_fusin_${fusin}/"
output_dir="checkpoint/proposed_with_finetune_T5/beauty_new_best_4_gpus_batch_${batchsize}_save_step_${step}_fusin_${fusin}_max_seq_length_${maxsequencelength}_from_total_data_points_from_best_valid_dev/"
#output_dir="checkpoint/synthetic_taste_last_item_finetune_T5_fusin_1/beauty/"

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py  \
    --output_dir "${output_dir}/name"  \
    --cache_dir "${output_dir}/cache"  \
    --do_train \
    --save_steps ${step}  \
    --eval_steps ${step}  \
   --per_device_train_batch_size ${batchsize}  \
    --per_device_eval_batch_size ${batchsize}  \
    --train_n_passages 10  \
    --num_passages 2  \
    --learning_rate 2e-4  \
    --warmup_ratio 0.1 \
    --q_max_len 256  \
    --p_max_len 32  \
    --seed 2022  \
    --num_train_epochs 16  \
    --evaluation_strategy steps  \
    --logging_steps 100 \
    --logging_dir "${output_dir}/name-log" \
    --train_path "${local_path}/train_name.jsonl"  \
    --eval_path "${local_path}/valid_name.jsonl"  \
   --model_name_or_path "/data/datasets/hf_cache/sample/pretrain_T5/beauty_new_best_sequence_length_3_proposed_pretrain_4_gpus_batch_160_save_step_300_fusin_0_total_data_points/name/best_valid_dev/pretrain_weight/"   \
    --resume_from_checkpoint "${output_dir}/name" \
    #--train_path "${local_path}/train_name_fusin_1.jsonl"  \
    #--eval_path "${local_path}/valid_name_fusin_1.jsonl"  \
    #--model_name_or_path "pretrained_model/t5-base"   \
    #--val_dataset2 "data/beauty/valid_name.jsonl"
       #&> ../TASTE/logs/beauty_t5_base.out
   #--model_name_or_path "checkpoint/pretrain_T5/beauty_new_best_with_pool_negatives_proposed_pretrain_8_gpus_batch_256_save_step_300_fusin_0/name/best_valid_dev/pretrain_weight"   \
   #--model_name_or_path "checkpoint/pretrain_T5/beauty_new_best_proposed_pretrain_8_gpus_batch_160_save_step_300_fusin_0_20_percent/name/best_dev/pretrain_weight/"   \

