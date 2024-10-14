#!/bin/sh


checkpoint_dir="/home/li1/coin/pretrain_recommendation/checkpoint_benchmark_from_devserver/pythia_beauty_1e-5_1024_64/name/checkpoint-3300"
echo $checkpoint_dir

CUDA_VISIBLE_DEVICES=5 python inference.py  \
    --data_name pythia_410m/beauty_pythia_amazonReview_data/  \
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
