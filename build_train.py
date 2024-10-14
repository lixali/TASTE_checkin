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
        "--item_file",
        type=str,
        default="/data1/meisen/TASTE-main/Data/beauty/item.txt",
        help="Path of the item.txt file",
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


def load_item_input_ids(filename):
    item_input_ids_dict = dict()
    with open(filename, "r", encoding="utf-8") as f:
        for example in jsonlines.Reader(f):
            id = example["id"]
            item_ids = example["item_ids"]
            item_input_ids_dict[id] = item_ids
    return item_input_ids_dict


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

#    with open("first_last_item.jsonl", 'w') as mf:
#        for inner_list in data:
#            # Convert each inner list into a dictionary (JSON object) with custom keys
#            json_obj = {"item1": inner_list[0], "item2": inner_list[1]}
#            # Write each JSON object as a line in the JSONL file
#            mf.write(json.dumps(json_obj) + '\n')
#            #json.dump(json_obj, mf)
#            #mf.write('\n')
#            #mf.write(json.dumps(inner_list) + '\n')        
#
#            #breakpoint()
# 
#

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


def load_random_neagtive_items(args, item_num, data_num, train_data_ids):
    np.random.seed(args.seed)
    negative_samples = {}
    print(
        "going to sample: %d negatives for each training datapoint"
        % int(args.sample_num)
    )
    for i in range(data_num):
        samples = []
        for _ in range(args.sample_num):
            item = np.random.choice(item_num) + 1  # one-indexing
            while (
                item in train_data_ids[i]
                or item in samples
                or item == item_num  # extra check for keyerror in small datasets
            ):  # hash to the next one
                item = np.random.choice(item_num) + 1
                if len(train_data_ids[i]) + len(samples) == item_num:  # saturated
                    breakpoint()
            samples.append(item)
        negative_samples[i] = samples
    print("length of negative samples is %d" % len(negative_samples))
    print("sample of the first data point: %d" % len(negative_samples[0]))
    return negative_samples


def main():
    args = get_args()
    if args.t5:
        print("T5 tokenizer..")
        tokenizer = T5Tokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    else:
        print("Fast tokenizer..")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    item_input_ids_dict = load_item_input_ids(args.item_ids_file)
    item_num = len(item_input_ids_dict)
    print("item num is %d" % item_num)
    # We only use the title attribute in the Amazon dataset, and the title and address attributes in the Yelp dataset.
    if args.data_name == "Amazon":
        item_desc = load_item_name(args.item_file)
    elif args.data_name == "yelp":
        item_desc = load_item_address(args.item_file)
    train_data, train_data_ids = load_data(args.train_file, item_desc)
    
    #print(train_data) 
    #train_data1 = [["sdfsdf", "hjkhj"], ["2342", "67978"]]
    #breakpoint()
   
    with open('output.txt', 'w') as f:
        for line in train_data:
            f.write(line[0] + line[1] + '\n') 
    

    exit()
    #breakpoint()
#    data_num = len(train_data)
#    print("data num is %d" % data_num)
#    breakpoint()
#    random_neg_dict = load_random_neagtive_items(
#        args, item_num, data_num, train_data_ids
#    )
#
#    breakpoint()
#    output_file = os.path.join(args.output_dir, args.output)
#    if not os.path.exists(args.output_dir):
#        os.makedirs(args.output_dir)
#    template1 = "Here is the visit history list of user: "
#    template2 = " recommend next item "
#    t1 = tokenizer.encode(template1, add_special_tokens=False, truncation=False)
#    t2 = tokenizer.encode(template2, add_special_tokens=False, truncation=False)
#    split_num = args.max_len - len(t1) - len(t2) - 1
#    # all "query" are the interaction history in text
#    with open(output_file, "w") as f:
#        for idx, data in enumerate(tqdm(train_data)):
#            pos_list = []
#            neg_list = []
#            query = data[0]
#            query = tokenizer.encode(
#                query, add_special_tokens=False, padding=False, truncation=False
#            )
#            query_list = list_split(
#                query, split_num
#            )  # cut the history into 2 pieces, the first within seq max_len limit
#            query_list[0] = t1 + query_list[0] + t2  # first seq fit into template
#            if not args.t5 or args.num_passages == 1:
#                # split into n_passages for t5, otherwise, only keep the first seq that fit into template
#                query_list = query_list[:1]
#            pos = data[1]
#            group = {}
#            pos_list.append(item_input_ids_dict[pos])
#            for id in random_neg_dict[idx]:
#                neg_list.append(item_input_ids_dict[id])
#            group["query"] = query_list  # a list of lists of query_ids
#            group["positives"] = pos_list
#            group["negatives"] = neg_list
#            f.write(json.dumps(group) + "\n")
#
    print("-----finish------")


if __name__ == "__main__":
    main()
