import os

import faiss
import numpy as np
import torch

from src.model import PythiaTASTEModel, TASTEModel
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer, T5Tokenizer
from utils.data_loader import (
    DItemDataset,
    DSequenceDataset,
    ItemDataset,
    load_data,
    load_item_address,
    load_item_data,
    load_item_name,
    SequenceDataset,
)
from utils.option import Options
from utils.rec_metrics import get_metrics_dict
from utils.util import init_logger, set_randomseed

import json
import re
import pdb

def evaluate(model, test_seq_dataloader, test_item_dataloader, device, Ks, logging, test_data = None, test_data_reverse = None, item_desc = None):
    logging.info("***** Running testing *****")
    model.eval()
    model = model.module if hasattr(model, "module") else model
    item_emb_list = []
    seq_emb_list = []
    target_item_list = []
    with torch.no_grad():
        for _i, batch in tqdm(
            enumerate(test_item_dataloader), total=len(test_item_dataloader)
        ):
            item_inputs = batch["item_ids"].to(device)
            item_masks = batch["item_masks"].to(device)
            _, item_emb = model(item_inputs, item_masks)
            item_emb_list.append(item_emb.cpu().numpy())
        item_emb_list = np.concatenate(item_emb_list, 0)

        #breakpoint()
        for _i, batch in tqdm(
            enumerate(test_seq_dataloader), total=len(test_seq_dataloader)
        ):
            seq_inputs = batch["seq_ids"].to(device)
            seq_masks = batch["seq_masks"].to(device)
            batch_target = batch["target_list"]
            _, seq_emb = model(seq_inputs, seq_masks)
            seq_emb_list.append(seq_emb.cpu().numpy())
            target_item_list.extend(batch_target)
        seq_emb_list = np.concatenate(seq_emb_list, 0)

        #breakpoint()

        faiss.omp_set_num_threads(16)
        cpu_index = faiss.IndexFlatIP(
            seq_emb_list.shape[1]
        )  # adapt for any embeddding hidden dim
        
        cpu_index.add(np.array(item_emb_list, dtype=np.float32))
        query_embeds = np.array(seq_emb_list, dtype=np.float32)
        

        D, I = cpu_index.search(query_embeds, max(Ks))
        n_item = item_emb_list.shape[0]
        n_seq = seq_emb_list.shape[0]
        prompt = ""
        #prompt = "Following is one example of the item sequence and recommended item. "
        pattern = r'id: \d+ title: ' 
         
        #breakpoint()
        with open("sports_prompt_0_shot.jsonl", "w") as f:
            for idx in range(len(test_data_reverse)):
                
#                if (idx + 1) % 2 != 0: 
#                    prompt += "Item sequence is '"
#                    prompt += test_data_reverse[idx][0] + "', "
#                    prompt += " recommended item is '"
#                    prompt += item_desc[test_data_reverse[idx][1]] + "'. " 
#                    #prompt = re.sub(pattern, '', prompt)
#    
                if (idx+1) % 2 == 0:
                    prompt = prompt + "Given item sequence :"
    
                    prompt = prompt + test_data_reverse[idx][0] + ", and following items: "
                    #prompt = re.sub(pattern, '', prompt)
                    #print(idx + 1) 
                    #print(I[idx])
                    for item_id in I[idx]:
                        print(item_id)
                        prompt += "" + item_desc[item_id] + ", "
                    #prompt +=  "rerank the 20 item based on above one example (from first recommendation preference to last). Output 20 item IDs only (output format is a string of 20 numbers separated by comma)"
                    prompt +=  "rerank the 20 items (from first recommendation preference to last). Output 20 item IDs only (output format is a string of 20 numbers separated by comma)"
                
                    json_line = {"prompt": prompt}
                    f.write(json.dumps(json_line) + "\n")

                    #breakpoint()
                    prompt = ""
                    #prompt = "Following is one examples of the item sequence and recommended item. "
                


 
      
        #breakpoint()
                
        metrics_dict = get_metrics_dict(I, n_seq, n_item, Ks, target_item_list)
        logging.info(
            "Test: Recall@10：{:.4f}, Recall@20:{:.4f}, NDCG@10: {:.4f}, NDCG@20:{:.4f}".format(
                metrics_dict[10]["recall"],
                metrics_dict[20]["recall"],
                metrics_dict[10]["ndcg"],
                metrics_dict[20]["ndcg"],
            )
        )

        logging.info("***** Finish test *****")


    
       
    

def load_data_reverse(filename, item_desc):
    data = []
    lines = open(filename, "r").readlines()
    for line in lines[1:]:
        example = list()
        line = line.strip().split("\t")
        target = int(line[-1])
        seq_id = line[1:-1]
        text_list = []
        for id in seq_id:
            id = int(id)
            if id == 0:
                break
            text_list.append(item_desc[id])
        #text_list.reverse()
        seq_text = ", ".join(text_list)
        example.append(seq_text)
        example.append(target)
        data.append(example)

    return data



def main():
    options = Options()
    opt = options.parse()
    set_randomseed(opt.seed)
    checkpoint_path = os.path.join(
        opt.checkpoint_dir, opt.data_name, opt.experiment_name, "test"
    )
    runlog_path = os.path.join(checkpoint_path, "log")
    os.makedirs(runlog_path, exist_ok=True)
    logging = init_logger(os.path.join(runlog_path, "runlog.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tokenizer: switch for T5 tokenizer or Fast Tokenizer
    if "pythia" in opt.best_model_path or "llama" in opt.best_model_path:
        tokenizer = AutoTokenizer.from_pretrained(opt.best_model_path)
    else:
        tokenizer = T5Tokenizer.from_pretrained(opt.best_model_path)

    # Model: switch for model class
    if "pythia" in opt.best_model_path:
        model_class = PythiaTASTEModel
    else:
        model_class = TASTEModel
    model = model_class.from_pretrained(opt.best_model_path)
    model.to(device)

    data_dir = os.path.join(opt.data_dir, opt.data_name)
    #test_file = os.path.join(data_dir, "test.txt")
    #test_file = "three_lines.txt"
    test_file = f"sports_train_last_test_first_600_lines.txt"
    item_file = os.path.join(data_dir, "item.txt")

    # We only use the title attribute in the Amazon dataset, and the title and address attributes in the Yelp dataset.
    data_suffix = opt.data_name.split("_")[-1]  # make data name as model_dataset format
    if "yelp" in opt.data_name:
        item_desc = load_item_address(item_file)
    else:
        item_desc = load_item_name(item_file)
        
    item_len = len(item_desc)
    
    logging.info(f"{opt}")

    logging.info(f"item len: {item_len}")

    #### changed by Lixiang for generating in-domain data for LLM filtering #######
    test_data = load_data(test_file, item_desc)
    test_data_reverse = load_data_reverse(test_file, item_desc)

    #####finished changing by Lixinag for generating in-domain data for LLM filtering ######


    logging.info(f"test len: {len(test_data)}")
    item_data = load_item_data(item_desc)
    
    #breakpoint()
    # Evaluation Datasets: switch for tokenizer and tokenizer fast
    if "pythia" in opt.best_model_path or "llama" in opt.best_model_path:
        test_seq_dataset = DSequenceDataset(test_data, tokenizer, opt)
        test_item_dataset = DItemDataset(item_data, tokenizer, opt)
    else:
        test_seq_dataset = SequenceDataset(test_data, tokenizer, opt)
        test_item_dataset = ItemDataset(item_data, tokenizer, opt)

    test_seq_sampler = SequentialSampler(test_seq_dataset)
    test_seq_dataloader = DataLoader(
        test_seq_dataset,
        sampler=test_seq_sampler,
        batch_size=opt.eval_batch_size,
        drop_last=False,
        num_workers=0,
        collate_fn=test_seq_dataset.collect_fn,
    )
    test_item_sampler = SequentialSampler(test_item_dataset)
    test_item_dataloader = DataLoader(
        test_item_dataset,
        sampler=test_item_sampler,
        batch_size=opt.eval_batch_size,
        drop_last=False,
        num_workers=0,
        collate_fn=test_item_dataset.collect_fn,
    )
    Ks = eval(opt.Ks)
    #breakpoint()
    evaluate(model, test_seq_dataloader, test_item_dataloader, device, Ks, logging, test_data, test_data_reverse, item_desc)


if __name__ == "__main__":
    main()
