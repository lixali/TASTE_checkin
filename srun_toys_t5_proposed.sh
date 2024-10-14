#!/bin/bash

#SBATCH --job-name=lixiang_taste_toys
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err # I put this in directory `outputs`, if the directory doesn't exists, job will fail immediately

#SBATCH --partition=general # check the partitions available and switch if you need a longer job/ different resources 

#SBATCH --mail-type=ALL
#SBATCH --mail-user=lixiangl@andrew.cmu.edu

#SBATCH --gres=gpu:A6000:4
#SBATCH --time=1-02:50:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task 16
#SBATCH --nodes=1

# enter a config env
eval "$(conda shell.bash hook)"
conda activate taste_run




experiment_run="toys"
steps=200
batchsize=200
sequence_max_length=3
fusin=0
#report_dir="pretrain_T5/toys_new_best_proposed_pretrain_10_percent"

#report_dir="pretrain_T5/toys_new_best_sequence_length_3_proposed_pretrain_4_gpus_batch_200_save_step__fusin__80K_data_points"

report_dir="pretrain_T5/toys_new_best_sequence_length_${sequence_max_length}_proposed_pretrain_4_gpus_batch_${batchsize}_save_step_${steps}_fusin_${fusin}_total_data_points"

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py  \
    --output_dir "checkpoint/${report_dir}/name"  \
    --resume_from_checkpoint "checkpoint/${report_dir}/name"  \
    --model_name_or_path "pretrained_model/t5-base"  \
    --do_train  \
    --save_steps ${steps}  \
    --eval_steps ${steps} \
   --per_device_train_batch_size ${batchsize}  \
    --per_device_eval_batch_size ${batchsize}  \
    --train_n_passages 10  \
    --num_passages 1  \
    --learning_rate 2e-4  \
    --q_max_len 64  \
    --p_max_len 64  \
    --seed 2022  \
    --warmup_ratio 0.1 \
    --num_train_epochs 15  \
    --evaluation_strategy steps  \
    --report_to tensorboard \
    --logging_dir "checkpoint/${report_dir}/name-log" \
    --logging_steps 100 \
    --train_path "data_proposed_multiple_lengths/${experiment_run}/train_name_length_3_total.jsonl"  \
    --eval_path "data_proposed_multiple_lengths/${experiment_run}/valid_name_length_3_total.jsonl"  \
    #--train_path "data_proposed_with_pool_negatives/${experiment_run}/train_name.jsonl"  \
    #--eval_path "data_proposed_with_pool_negatives/${experiment_run}/valid_name.jsonl"  \
  
