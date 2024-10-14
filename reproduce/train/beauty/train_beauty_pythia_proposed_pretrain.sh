#!/bin/sh

model="pythia_410m"
data_set="beauty_pythia_item_seq"

lr=1e-5
bz=32
q_max_len=1024
p_max_len=256

report_dir="${data_set}_${lr}_${q_max_len}_${p_max_len}_proposed_pretrain"

# torchrun --nproc-per-node=1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py  \
    --output_dir "../TASTE_checkin/checkpoint_proposed/${report_dir}/name"  \
    --model_name_or_path ../TASTE/pretrained_model/${model}  \
    --do_train  \
    --save_steps 100  \
    --eval_steps 100  \
    --train_path ../TASTE_checkin/proposed_train_data/${data_set}/train_anchor_pos_neg.jsonl  \
    --eval_path ../TASTE_checkin/proposed_train_data/${data_set}/valid_anchor_pos_neg.jsonl  \
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
    # --resume_from_checkpoint "../TASTE/checkpoint/${report_dir}/name"  \
