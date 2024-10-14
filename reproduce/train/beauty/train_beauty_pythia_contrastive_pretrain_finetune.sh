#!/bin/sh

model="pythia_410m"
data_set="beauty"
checkpoint="checkpoint_pythia_contrastive_pretrain_finetune"
lr=1e-5
bz=64
q_max_len=1024
p_max_len=64

report_dir="second_round_item_length_256_during_pretrain_pythia_beauty_item_seq_${lr}_${q_max_len}_${p_max_len}_pythia_contrastive_pretrain_finetune"

# torchrun --nproc-per-node=1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py  \
    --output_dir "../TASTE_checkin/checkpoint_proposed/${report_dir}/name"  \
    --model_name_or_path ../TASTE_checkin/checkpoint_proposed/beauty_pythia_item_seq_1e-5_1024_256_proposed_pretrain/name/checkpoint-600/ \
    --do_train  \
    --save_steps 100  \
    --eval_steps 100  \
    --train_path ../TASTE_checkin/data/${model}/beauty_pythia_amazonReview_data//train_name.jsonl  \
    --eval_path ../TASTE_checkin/data/${model}/beauty_pythia_amazonReview_data//valid_name.jsonl  \
    --per_device_train_batch_size $bz  \
    --per_device_eval_batch_size $bz  \
    --train_n_passages 10  \
    --num_passages 2  \
    --learning_rate $lr \
    --q_max_len $q_max_len  \
    --p_max_len $p_max_len  \
    --warmup_ratio 0.01 \
    --seed 2022  \
    --num_train_epochs 30  \
    --evaluation_strategy steps  \
    --logging_dir "../TASTE_checkin/checkpoint_proposed/${report_dir}/name-log" \
    --report_to tensorboard \
    --resume_from_checkpoint "../TASTE_checkin/checkpoint_proposed/${report_dir}/name"  \
