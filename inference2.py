import os

import faiss
import numpy as np
import torch
import json
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
import time

def evaluate(model, test_seq_dataloader, test_item_dataloader, device, Ks, logging):
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
        faiss.omp_set_num_threads(16)
        cpu_index = faiss.IndexFlatIP(
            seq_emb_list.shape[1]
        )  # adapt for any embeddding hidden dim
        cpu_index.add(np.array(item_emb_list, dtype=np.float32))
        query_embeds = np.array(seq_emb_list, dtype=np.float32)
        D, I = cpu_index.search(query_embeds, max(Ks))
        n_item = item_emb_list.shape[0]
        n_seq = seq_emb_list.shape[0]
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

        output_file_path = "sports_gptoutput_0_shot.jsonl"
        
        # Check if the file exists before proceeding
        if os.path.exists(output_file_path):
            print(f"{output_file_path} exists. Proceeding to read the file...")
        
            # List to store the parsed numbers
            all_numbers = []
        
            # Open the JSONL file and read each line
            with open(output_file_path, "r") as f:
                for line in f:
                    json_line = json.loads(line)  # Parse each line as a JSON object
                    completion_dict = json_line.get('completion', {})  # Get the 'completion' dictionary
        
                    # Extract the 'content' value from the 'completion' dictionary
                    content = completion_dict.get('content', "")
                    # Assuming 'content' is a string of space-separated numbers, convert it into a list of integers
                    try:
                        numbers = list(map(int, content.split(",")))
                        if len(numbers) > 20: 
                            print(f"{numbers} len is larger than 20")
                            numbers = numbers[:20]
                        elif len(numbers) < 20:
                            print(f"{numbers} len is less than 20")
                            numbers.extend([0] * (20 - len(numbers)))
               
                            
                    except ValueError:
                        # Handle the case where content might not be a proper string of numbers
                        print(f"Invalid content found: {content}")
                        continue
        
                    # Store the list of numbers
                    all_numbers.append(np.array(numbers))

                print(f"len of the number of tested sequecne is {len(all_numbers)}")
                breakpoint()
                metrics_dict = get_metrics_dict(all_numbers, n_seq, n_item, Ks, target_item_list)
                logging.info(
                    "Test: Recall@10：{:.4f}, Recall@20:{:.4f}, NDCG@10: {:.4f}, NDCG@20:{:.4f}".format(
                        metrics_dict[10]["recall"],
                        metrics_dict[20]["recall"],
                        metrics_dict[10]["ndcg"],
                        metrics_dict[20]["ndcg"],
                    )
                )
         
            # Print all the lists of numbers
            #for numbers in all_numbers:
                #print(numbers)
        else:
            print(f"{output_file_path} does not exist. Please check the file path.")
 
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

    start_time = time.time()
    start_time_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    logging.info(f"Start time: {start_time_readable}")

    
    # Tokenizer: switch for T5 tokenizer or Fast Tokenizer
    if "pythia" in opt.best_model_path or "llama" in opt.best_model_path:
        tokenizer = AutoTokenizer.from_pretrained(opt.best_model_path)
    else:
        tokenizer = T5Tokenizer.from_pretrained(opt.best_model_path)
        
    if not tokenizer.pad_token: tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Model: switch for model class
    if "pythia" in opt.best_model_path:
        model_class = PythiaTASTEModel
    else:
        model_class = TASTEModel
    model = model_class.from_pretrained(opt.best_model_path)
    model.to(device)

    data_dir = os.path.join(opt.data_dir, opt.data_name)
    #test_file = os.path.join(data_dir, "test.txt")
    test_file = "sports_test_first_300_lines.txt"
    item_file = os.path.join(data_dir, "item.txt")

    # We only use the title attribute in the Amazon dataset, and the title and address attributes in the Yelp dataset.
    data_suffix = opt.data_name.split("/")[-1]  # make data name as model_dataset format
    if "yelp" in opt.data_name:   
        item_desc = load_item_address(item_file)
    else:
        item_desc = load_item_name(item_file)
    item_len = len(item_desc)
    logging.info(f"item len: {item_len}")
    test_data = load_data(test_file, item_desc)
    logging.info(f"test len: {len(test_data)}")
    item_data = load_item_data(item_desc)

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
    breakpoint()
    evaluate(model, test_seq_dataloader, test_item_dataloader, device, Ks, logging)

    end_time = time.time()
    end_time_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    logging.info(f"End time: {end_time_readable}")

    elapsed_time = end_time - start_time

    minutes, seconds = divmod(elapsed_time, 60)
    logging.info(f"Time taken: {int(minutes)} minutes and {seconds:.2f} seconds")



if __name__ == "__main__":
    main()
