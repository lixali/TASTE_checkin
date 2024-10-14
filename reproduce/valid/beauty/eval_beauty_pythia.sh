#!/bin/sh

experiment_name="checkpoint_proposed/beauty_pythia_item_seq_1e-5_1024_64_proposed_pretrain/name/for_quick_validation/"

echo $experiment_name
CUDA_VISIBLE_DEVICES=3 python evaluate.py  \
    --data_dir data/pythia_410m \
    --data_name beauty_pythia_amazonReview_data/  \
    --experiment_name $experiment_name \
    --seed 2022  \
    --item_size 64  \
    --seq_size 1024  \
    --num_passage 2  \
    --split_num 243  \
    --eval_batch_size 64  \
    --stopping_step 5  \
    --all_models_path $experiment_name


echo $experiment_name
