#!/bin/sh


# # Pythia Family
model="pythia_410m"
data_set="toys"

# dataset format switch
if [ $data_set = "yelp" ]
then
    set="yelp"
    folder="address"
else
    set="Amazon"
    folder="name"
fi

base_dir="data/${model}/${data_set}"
tokenizer_path="pretrained_model/${model}"

echo $base_dir

python build_train.py  \
    --data_name $set  \
    --sample_num 100 \
    --train_file "${base_dir}/train.txt"  \
    --item_file "${base_dir}/item.txt"  \
    --item_ids_file "${base_dir}/item_${folder}.jsonl"  \
    --output train_${folder}.jsonl  \
    --output_dir ${base_dir}  \
    --seed 2022  \
    --max_len 1024 \
    --tokenizer $tokenizer_path \


python build_train.py  \
    --data_name $set  \
    --sample_num 100 \
    --train_file "${base_dir}/valid.txt"  \
    --item_file "${base_dir}/item.txt"  \
    --item_ids_file "${base_dir}/item_${folder}.jsonl"  \
    --output valid_${folder}.jsonl  \
    --output_dir ${base_dir}  \
    --seed 2022  \
    --max_len 1024 \
    --tokenizer $tokenizer_path \
