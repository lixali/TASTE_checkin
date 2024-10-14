#!/bin/bash

#SBATCH --job-name=sports_lixiang_taste
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err # I put this in directory `outputs`, if the directory doesn't exists, job will fail immediately

#SBATCH --partition=general # check the partitions available and switch if you need a longer job/ different resources 

#SBATCH --mail-type=ALL
#SBATCH --mail-user=lixiangl@andrew.cmu.edu

#SBATCH --gres=gpu:A6000:4
#SBATCH --time=9:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task 16
#SBATCH --nodes=1

# enter a config env
eval "$(conda shell.bash hook)"
conda activate taste_run



fusin=1
experiment_run="sports"
step=1200
#step=2000
batchsize=160
#batchsize=64
report_dir="proposed_with_finetune_T5/sports_4_gpus_batch_${batchsize}_save_step_${step}_fusin_${fusin}_50_percent_delete2"
#report_dir="proposed_with_finetune_T5/sports_with_pool_negatives_4_gpus_batch_${batchsize}_save_step_${step}_fusin_${fusin}_iter_1"
#report_dir="proposed_with_finetune_T5/sports_4_gpus_batch_${batchsize}_save_step_${step}_fusin_${fusin}_iter_2_more_epochs_sbatch_delete"
#report_dir="synthetic_taste_last_item_finetune_T5_fusin_${fusin}/sports"
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py  \
    --output_dir "checkpoint/${report_dir}/name"  \
    --do_train  \
    --save_steps ${step}  \
    --eval_steps ${step}  \
    --per_device_train_batch_size ${batchsize}  \
    --per_device_eval_batch_size ${batchsize}  \
    --train_n_passages 10  \
    --num_passages 2  \
    --learning_rate 2e-4  \
    --q_max_len 256  \
    --p_max_len 32  \
    --seed 2022  \
    --warmup_ratio 0.1 \
    --num_train_epochs 2  \
    --evaluation_strategy steps  \
    --report_to tensorboard \
    --logging_dir "checkpoint/${report_dir}/name-log" \
    --logging_steps 100 \
    --resume_from_checkpoint  "checkpoint/${report_dir}/name" \
    --train_path "data/${experiment_run}/train_name.jsonl"  \
    --eval_path "data/${experiment_run}/valid_name.jsonl"  \
    --model_name_or_path "checkpoint/pretrain_T5/sports_proposed_pretrain_8_gpus_batch_160_save_step_300_fusin_0_50_percent/name/best_dev/pretrain_weight/"  \
    #--train_path "data_taste_synthetic/${experiment_run}/train_name_fusin_${fusin}.jsonl"  \
    #--eval_path "data_taste_synthetic/${experiment_run}/valid_name_fusin_${fusin}.jsonl"  \
    #--model_name_or_path "checkpoint/pretrain_T5/sports_with_pool_negatives_proposed_pretrain_4_gpus_batch_160_save_step_300_fusin_0/name/best_valid_dev/pretrain_weight/"  \



 
