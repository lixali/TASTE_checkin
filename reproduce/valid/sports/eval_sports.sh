#!/bin/bash

#SBATCH --job-name=lixiang_taste_sport_eval
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err # I put this in directory `outputs`, if the directory doesn't exists, job will fail immediately

#SBATCH --partition=general # check the partitions available and switch if you need a longer job/ different resources

#SBATCH --mail-type=ALL
#SBATCH --mail-user=lixiangl@andrew.cmu.edu

#SBATCH --gres=gpu:8000:1
#SBATCH --time=23:50:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task 16
#SBATCH --nodes=1

# enter a config env
eval "$(conda shell.bash hook)"
conda activate taste_run

model_dir="checkpoint/proposed_with_finetune_T5/beauty_new_best_4_gpus_batch_160_save_step_600_fusin_1_max_seq_length_3_from_80K_data_points/name/"

echo "${model_dir}"
CUDA_VISIBLE_DEVICES=1 python evaluate.py  \
    --data_name sports  \
    --experiment_name name \
    --seed 2022  \
    --item_size 32  \
    --seq_size 256  \
    --num_passage 2  \
    --split_num 243  \
    --eval_batch_size 256  \
    --stopping_step 100  \
    --all_models_path ${model_dir} \
    --checkpoint_dir ${model_dir}  


echo "${model_dir}"


#model_dir="checkpoint/TASTE_benchmark/sports_8_gpus_batch_160_save_step_600_fusin_1_20_percent/name"
#
#echo "${model_dir}"
#CUDA_VISIBLE_DEVICES=0 python evaluate.py  \
#    --data_name sports  \
#    --experiment_name name \
#    --seed 2022  \
#    --item_size 32  \
#    --seq_size 256  \
#    --num_passage 2  \
#    --split_num 243  \
#    --eval_batch_size 256  \
#    --stopping_step 100  \
#    --all_models_path ${model_dir} \
#    --checkpoint_dir ${model_dir}  &
#
#
#echo "${model_dir}"
#
#
