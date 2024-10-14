import sys
import re
sys.path.append("/data1/meisen/TASTE-main")
import json
import os.path
from argparse import ArgumentParser

import jsonlines
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, T5Tokenizer

from utils.data_loader import list_split, load_item_address, load_item_name
import collections




def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_name", type=str, default="Amazon", help="choose Amazon or yelp"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="/data1/meisen/TASTE-main/Data/beauty/valid.txt",
        help="Path of the train/valid.txt file",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="/data1/meisen/TASTE-main/Data/beauty/test.txt",
        help="Path of the train/test.txt file",
    )
    parser.add_argument(
        "--item_file",
        type=str,
        default="/data1/meisen/TASTE-main/Data/beauty/item.txt",
        help="Path of the item.txt file",
    )
    parser.add_argument(
        "--experiment_run",
        type=str,
        default="/data1/meisen/TASTE-main/Data/beauty/item_name.jsonl",
        help="Path of the item token file",
    )
    parser.add_argument(
        "--item_ids_file",
        type=str,
        default="/data1/meisen/TASTE-main/Data/beauty/item_name.jsonl",
        help="Path of the item token file",
    )
    parser.add_argument("--output", type=str, default="valid_name.jsonl")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data1/meisen/TASTE-main/Data/beauty",
        help="Output data path.",
    )
    parser.add_argument(
        "--split_num",
        type=int,
        default=243,
        help="token num of seq text without prompt, total num equals to 256",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=256,
        help="token num of seq text allowed for the model",
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=100,
        help="the sample num of random negatives ",  # might need to adjust for ml-1m -> might have out of bounds issur for items (3417)
    )
    parser.add_argument(
        "--num_passages",
        type=int,
        default=2,
        help="number of passages for query",
    )
    parser.add_argument("--seed", type=int, default=2022, help="random seed")
    parser.add_argument(
        "--tokenizer", type=str, default="/data1/meisen/pretrained_model/t5-base"
    )
    parser.add_argument("--t5", action="store_true")
    args = parser.parse_args()
    return args


def load_data(filename, item_desc):
    data = []
    data_ids = []
    lines = open(filename, "r").readlines()
    temp_lines = [] 
    total = []
    fourth_line = []
    for line in lines[1:]:
        example = list()
        example2 = list()
        line = line.strip().split("\t")
        #breakpoint()
        #### temporarily added by Lixiang #### 
        #breakpoint()
        #if line[2] != "0": continue
        if line[3] != "0" and line[4] == "0":
            if len(temp_lines) == 2: 
                for cl in temp_lines:
                    total.append(cl)
                total.append(line)
            temp_lines = []
            #breakpoint() 
        elif line[3] == "0":
    
            temp_lines.append(line)
        else: continue

        #####finished temporarily added by Lixiang #######


        target = int(line[-1])  # target item at this step
        seq_id = line[1:-1]  # the history of the user
        user = line[0]
        text_list = []
        # convert the item ids to the item text
        for id in seq_id:
            id = int(id)
            if id == 0:  # get the actions till current step
                break
            text_list.append(item_desc[id])
            example2.append(id)
        text_list.reverse()  # most recent to the oldest
        seq_text = ", ".join(text_list)
        example.append(seq_text)
        
        #example.append(target)
        
        ##### temporaily added by Lixiang ######
        example.append(item_desc[int(target)])
        pattern = r'id: \d+ title: ' 
        example = [re.sub(pattern, '', s) for s in example]
        #####finished temporarily added by Lixiang ########        

        example2.append(target)
        data.append(example)  # input txt list
        

        data_ids.append(example2)  # input id list

    with open("three_lines_delete.txt", 'w') as f:
        f.write("user_id seq     target" + "\n") 
        for idx, inner_list in enumerate(total):
            f.write('\t'.join(map(str, inner_list)) + '\n')
            if idx == 998: break

    with open("third_lines_delete.txt", "w") as f:
        f.write("user_id seq     target" + "\n") 
        for idx, inner_list in enumerate(total):
            if (idx + 1) % 3 == 0:
                f.write('\t'.join(map(str, inner_list)) + '\n')
            if idx == 998: break

    
    return data, data_ids




def load_data_train_last_line_test(train_file, test_file, item_desc, args):

    # Step 1: Read the last occurrence of each user_id from train.txt
    train_last_lines = collections.defaultdict(str)
    output_file = f"{args.experiment_run}_train_last_test_first.txt"
    output_file2 = f"{args.experiment_run}_test_first.txt"
    with open(train_file, 'r') as f:
        next(f)
        for line in f:
            # Skip empty lines
            if line.strip():
                # Extract user_id (first field)
                user_id = line.split()[0]
                # Store the line as the last seen occurrence of the user_id
                train_last_lines[user_id] = line.strip()
    
    # Step 2: Read the only occurrence of each user_id from test.txt
    test_lines = {}
    
    with open(test_file, 'r') as f:
        next(f)
        for line in f:
            # Skip empty lines
            if line.strip():
                # Extract user_id (first field)
                user_id = line.split()[0]
                test_lines[user_id] = line.strip()
    
    # Step 3: Write the output to combined.txt with pairs of lines
    with open(output_file, 'w') as f:
        f.write("user_id seq     target" + "\n")
        for user_id in train_last_lines:
            if user_id in test_lines:
                # Write the last occurrence from train.txt followed by the test.txt line
                f.write(train_last_lines[user_id] + '\n')
                f.write(test_lines[user_id] + '\n')
    
    with open(output_file2, 'w') as f:
        f.write("user_id seq     target" + "\n")
        breakpoint()
        for user_id in test_lines:
            # Write the last occurrence from train.txt followed by the test.txt line
            #print(user_id)
            f.write(test_lines[user_id] + '\n')
    
     

def main():
    args = get_args()
    item_desc = load_item_name(args.item_file)
    #train_data, train_data_ids = load_data(args.train_file, item_desc)
    load_data_train_last_line_test(args.train_file, args.test_file, item_desc, args)
    
#   
#    with open('output.txt', 'w') as f:
#        for line in train_data:
#            f.write(line[0] + line[1] + '\n') 
#    


    print("-----finish------")


if __name__ == "__main__":
    main()
