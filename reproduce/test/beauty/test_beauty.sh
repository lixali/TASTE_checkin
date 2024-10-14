#!/bin/sh

checkpoint_dir="/data/datasets/hf_cache/sample/synthetic_taste_last_item_finetune_T5_fusin_1/beauty_8_gpus_batch_160_save_step_600_fusin_1/name/best_dev"
echo $checkpoint_dir

CUDA_VISIBLE_DEVICES=0 python inference3.py  \
    --data_name beauty  \
    --experiment_name name  \
    --seed 2022  \
    --item_size 32  \
    --seq_size 256  \
    --num_passage 2  \
    --split_num 243  \
    --eval_batch_size 128  \
    --best_model_path ${checkpoint_dir} \
    --checkpoint_dir ${checkpoint_dir}

echo $checkpoint_dir
echo "inference test done"

#
#wait 
#
#
#checkpoint_dir="/data/datasets/hf_cache/sample/proposed_with_finetune_T5/beauty_new_best_8_gpus_batch_160_save_step_600_fusin_1_20_percent/name/best_dev"
#echo $checkpoint_dir
#
#CUDA_VISIBLE_DEVICES=0 python inference.py  \
#    --data_name beauty  \
#    --experiment_name name  \
#    --seed 2022  \
#    --item_size 32  \
#    --seq_size 256  \
#    --num_passage 2  \
#    --split_num 243  \
#    --eval_batch_size 128  \
#    --best_model_path ${checkpoint_dir} \
#    --checkpoint_dir ${checkpoint_dir}
#
#echo $checkpoint_dir
#echo "inference test done"
#
#
