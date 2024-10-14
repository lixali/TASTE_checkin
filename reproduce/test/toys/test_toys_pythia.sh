#!/bin/sh
checkpoint_dir="../TASTE_checkin/checkpoint/pythia_toys_1e-5_1024_64/name/checkpoint-2400/"
echo $checkpoint_dir
CUDA_VISIBLE_DEVICES=2 python inference.py  \
    --data_name toys  \
    --experiment_name name  \
    --seed 2022  \
    --item_size 64  \
    --seq_size 1024  \
    --num_passage 2  \
    --split_num 243  \
    --eval_batch_size 64  \
    --best_model_path ${checkpoint_dir} \
    --checkpoint_dir ${checkpoint_dir}

echo $checkpoint_dir
echo "inference test done"
