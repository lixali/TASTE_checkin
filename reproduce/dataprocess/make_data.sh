#!/bin/sh


# T5 Family
model="t5-base"
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

base_dir="data/${data_set}"
tokenizer_path="pretrained_model/${model}"

echo $base_dir

python3 make_data.py  \
    --data_name $set  \
    --sample_num 100 \
    --train_file "${base_dir}/train.txt"  \
    --test_file "${base_dir}/test.txt"  \
    --item_file "${base_dir}/item.txt"  \
    --item_ids_file "${base_dir}/item_${folder}.jsonl"  \
    --output train_${folder}.jsonl  \
    --output_dir ${base_dir}  \
    --seed 2022  \
    --tokenizer $tokenizer_path \
    --split_num 499 \
    --num_passages 1 \
    --t5 \
    --experiment_run ${data_set}

