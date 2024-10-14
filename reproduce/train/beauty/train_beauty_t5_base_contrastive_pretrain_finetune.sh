#!/bin/sh

data_set="beauty_t5_base_download_from_manifold_should_be_correct"

lr=1e-4
bz=512
q_max_len=256
p_max_len=32
negative_sample_per_row=100

report_dir="t5_base_beauty_negative_sample_per_row_${negative_sample_per_row}_${lr}_${q_max_len}_${p_max_len}_bz_${bz}_contrastive_pretrain_finetune"

# torchrun --nproc-per-node=1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py  \
    --output_dir "../TASTE_checkin/checkpoint_proposed/${report_dir}/name"  \
    --model_name_or_path checkpoint_proposed/t5_base_beauty_2.7e-4_256_32_proposed_pretrain/name/checkpoint-1300  \
    --do_train  \
    --save_steps 100  \
    --eval_steps 100  \
    --train_path ../TASTE_checkin/data/${data_set}/train_name.jsonl  \
    --eval_path ../TASTE_checkin/data/${data_set}/valid_name.jsonl  \
    --per_device_train_batch_size $bz  \
    --per_device_eval_batch_size $bz  \
    --train_n_passages 10  \
    --num_passages 2  \
    --learning_rate $lr \
    --q_max_len $q_max_len  \
    --p_max_len $p_max_len  \
    --warmup_ratio 0.1 \
    --seed 2022  \
    --num_train_epochs 30  \
    --evaluation_strategy steps  \
    --logging_dir "../TASTE_checkin/checkpoint_proposed/${report_dir}/name-log" \
    --report_to tensorboard \
    --logging_steps 100 \

    # --resume_from_checkpoint "../TASTE_checkin/checkpoint_proposed/${report_dir}/name"  \
