#!/bin/sh


# T5 Family
model="t5_large"
data_set="toy"

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

python gen_all_items.py \
    --data_name $set \
    --item_file "${base_dir}/item.txt" \
    --output item_${folder}.jsonl \
    --output_dir $base_dir \
    --tokenizer $tokenizer_path \
    --item_size 32 \
    --t5
