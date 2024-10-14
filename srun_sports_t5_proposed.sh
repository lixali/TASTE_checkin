#!/bin/bash

#SBATCH --job-name=sports_lixiang_taste
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err # I put this in directory `outputs`, if the directory doesn't exists, job will fail immediately

#SBATCH --partition=general # check the partitions available and switch if you need a longer job/ different resources 

#SBATCH --mail-type=ALL
#SBATCH --mail-user=lixiangl@andrew.cmu.edu

#SBATCH --gres=gpu:8000:4
#SBATCH --time=23:50:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task 16
#SBATCH --nodes=1

# enter a config env
eval "$(conda shell.bash hook)"
conda activate taste_run




experiment_run="sports"
#output_dir="pretrain_T5/sports_proposed_pretrain_iter_2/"

batchsize=160
steps=300
fusin=0
output_dir="pretrain_T5/sports_proposed_pretrain_8_gpus_batch_${batchsize}_save_step_${steps}_fusin_${fusin}_50_percent/"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py  \
    --output_dir "checkpoint/${output_dir}/name"  \
    --resume_from_checkpoint "checkpoint/${output_dir}/name"  \
    --model_name_or_path "pretrained_model/t5-base"  \
    --do_train  \
    --save_steps ${steps}  \
    --eval_steps ${steps}  \
    --train_path "data_proposed/${experiment_run}/train_name_50_percent.jsonl"  \
    --eval_path "data_proposed/${experiment_run}/valid_name.jsonl"  \
    --per_device_train_batch_size ${batchsize}  \
    --per_device_eval_batch_size ${batchsize}  \
    --train_n_passages 10  \
    --num_passages 1  \
    --learning_rate 2e-4  \
    --q_max_len 64  \
    --p_max_len 64  \
    --seed 2022  \
    --warmup_ratio 0.1 \
    --num_train_epochs 20  \
    --evaluation_strategy steps  \
    --report_to tensorboard \
    --logging_dir "checkpoint/${output_dir}/name-log" \
    --logging_steps 100
