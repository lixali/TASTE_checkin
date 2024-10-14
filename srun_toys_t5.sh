#!/bin/bash

#SBATCH --job-name=lixiang_taste_toys
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err # I put this in directory `outputs`, if the directory doesn't exists, job will fail immediately

#SBATCH --partition=general # check the partitions available and switch if you need a longer job/ different resources 

#SBATCH --mail-type=ALL
#SBATCH --mail-user=lixiangl@andrew.cmu.edu

#SBATCH --gres=gpu:L40:4
#SBATCH --time=23:50:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task 16
#SBATCH --nodes=1

# enter a config env
eval "$(conda shell.bash hook)"
conda activate taste_run




experiment_run="toys"
fusin=1
step=500
batchsize=160
maxseqlength=3
#report_dir="synthetic_taste_last_item_finetune_T5_fusin_${fusin}/toys/"
#report_dir="checkpoint/toys_new_best_4_gpus_batch_${batchsize}_save_step_${step}_fusin_${fusin}"
report_dir="proposed_with_finetune_T5/toys_new_best_4_gpus_batch_${batchsize}_save_step_${step}_fusin_${fusin}_max_seq_length_${maxseqlength}_total_data_points_from_best_valid_dev"
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py  \
    --output_dir "checkpoint/${report_dir}/name"  \
    --resume_from_checkpoint "checkpoint/${report_dir}/name" \
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
    --num_train_epochs 16  \
    --evaluation_strategy steps  \
    --report_to tensorboard \
    --logging_dir "checkpoint/${report_dir}/name-log" \
    --logging_steps 100 \
    --train_path "data/${experiment_run}/train_name.jsonl"  \
    --eval_path "data/${experiment_run}/valid_name.jsonl"  \
    --model_name_or_path "/data/datasets/hf_cache/sample/pretrain_T5/toys_new_best_sequence_length_3_proposed_pretrain_4_gpus_batch_200_save_step_200_fusin_0_total_data_points/name/best_valid_dev/pretrain_weight"  \
#    --model_name_or_path "checkpoint/pretrain_T5/toys_new_best_proposed_pretrain_with_pool_negatives/name/best_valid_dev/pretrain_weight/"  \
#    --train_path "data_taste_synthetic/${experiment_run}/train_name_fusin_${fusin}.jsonl"  \
#    --eval_path "data_taste_synthetic/${experiment_run}/valid_name_fusin_${fusin}.jsonl"  \
  
