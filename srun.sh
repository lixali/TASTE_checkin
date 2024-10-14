#!/bin/bash
#SBATCH --job-name=lixiang_taste
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err # I put this in directory `outputs`, if the directory doesn't exists, job will fail immediately
#SBATCH --nodes=1
#SBATCH --partition=general # check the partitions available and switch if you need a longer job/ different resources 

#SBATCH --mem=64G
#SBATCH --gres=gpu:A6000

#SBATCH --ntasks-per-node=1

# enter a config env
eval "$(conda shell.bash hook)"
conda activate taste_run


srun CUDA_VISIBLE_DEVICES=0 python train.py  \
    --output_dir checkpoint/beauty/name  \
    --model_name_or_path pretrained_model/t5-base  \
    --do_train  \
    --save_steps 5000  \
    --eval_steps 5000  \
    --train_path data/beauty/train_name.jsonl  \
    --eval_path  data/beauty/valid_name.jsonl  \
    --per_device_train_batch_size 8  \
    --per_device_eval_batch_size 8  \
    --train_n_passages 10  \
    --num_passages 2  \
    --learning_rate 1e-4  \
    --q_max_len 256  \
    --p_max_len 32  \
    --seed 2022  \
    --num_train_epochs 30  \
    --evaluation_strategy steps  \
    --logging_dir checkpoint/beauty/name-log
