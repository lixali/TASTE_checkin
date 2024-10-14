#!/bin/bash
#SBATCH --job-name=lixiang_run       # Name of your job
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=4              # Number of CPU cores per task
#SBATCH --gres=gpu:1                   # Request 2 GPUs
#SBATCH --time=02:00:00                # Time limit hrs:min:sec
#SBATCH --output=output_%j.log         # Standard output and error log

# Load any necessary modules (if required)
# module load python/3.x.x

# Activate your Python environment (if using virtual environments)
# source /path/to/your/venv/bin/activate

# Run your Python script using srun


opensource_path="./data/beauty"
local_path="./data/"
output_dir="./checkpoint/test_beauty_srun/"

srun CUDA_VISIBLE_DEVICES=0 python train.py  \
    --output_dir "${output_dir}/name"  \
    --cache_dir "${output_dir}/cache"  \
    --model_name_or_path ../TASTE/pretrained_model/t5_base  \
    --do_train \
    --save_steps 5000  \
    --eval_steps 5000  \
    --train_path "${local_path}/train_name.jsonl"  \
    --eval_path "${local_path}/valid_name.jsonl"  \
    --per_device_train_batch_size 4  \
    --per_device_eval_batch_size 4  \
    --train_n_passages 10  \
    --num_passages 2  \
    --learning_rate 1e-4  \
    --warmup_ratio 0.1 \
    --q_max_len 256  \
    --p_max_len 32  \
    --seed 2022  \
    --num_train_epochs 30  \
    --evaluation_strategy steps  \
    --logging_dir "${output_dir}/name-log"

