import glob
import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from datasets import load_dataset
from openmatch.dataset.train_dataset import (
    MappingTrainDatasetMixin,
    StreamTrainDatasetMixin,
    TrainDatasetBase,
)
from openmatch.trainer import DRTrainer
from torch.utils.data import Dataset, IterableDataset
from transformers import (
    BatchEncoding,
    DataCollatorWithPadding,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


@dataclass
class DTasteCollator(DataCollatorWithPadding):
    """
    Taste Collator adapted for the decoder-only models' fast tokenizers.
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    max_q_len: int = 32
    max_p_len: int = 128
    len_seq: int = 2

    def __call__(self, features):
        qq = [f["query_"] for f in features]
        dd = [f["passages"] for f in features]

        if isinstance(dd[0], list):
            dd = sum(dd, [])
        q_collated_list = list()
        # perform padding here
        for seq in qq:
            q_collated = self.tokenizer.pad(
                seq,
                padding="max_length",
                max_length=self.max_q_len,
                return_tensors="pt",
            )
            q_collated_list.append(q_collated)
        seq_input_ids = []
        seq_attention_mask = []
        # list has shape of bz
        for q_collated in q_collated_list:
            item_input_ids = q_collated.data["input_ids"]
            item_attention_mask = q_collated.data["attention_mask"]
            # adapt input shape for Pythia model: n_passage * dim -> gpt input: bsz*seq_len
            seq_input_ids.append(item_input_ids)
            seq_attention_mask.append(item_attention_mask)
        seq_input_ids = torch.cat(seq_input_ids, dim=0)
        seq_attention_mask = torch.cat(seq_attention_mask, dim=0)
        query = (seq_input_ids, seq_attention_mask)

        # the possible items
        d_collated = self.tokenizer.pad(
            dd,
            padding="max_length",
            max_length=self.max_p_len,
            return_tensors="pt",
        )
        item_input_ids = d_collated.data["input_ids"]
        item_attention_mask = d_collated.data["attention_mask"]
        # the decody-only pythia doesn't need the extra dimension
        item = (item_input_ids, item_attention_mask)

        return query, item


@dataclass
class TasteCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    max_q_len: int = 32
    max_p_len: int = 128
    len_seq: int = 2

    def __call__(self, features):
        qq = [f["query_"] for f in features]
        dd = [f["passages"] for f in features]

        if isinstance(dd[0], list):
            dd = sum(dd, [])
        q_collated_list = list()
        # perform padding here
        for seq in qq:
            q_collated = self.tokenizer.pad(
                seq,
                padding="max_length",
                max_length=self.max_q_len,
                return_tensors="pt",
            )
            q_collated_list.append(q_collated)
        seq_input_ids = []
        seq_attention_mask = []
        # list has shape of bz
        for q_collated in q_collated_list:
            item_input_ids = q_collated.data["input_ids"]
            item_attention_mask = q_collated.data["attention_mask"]
            cur_item = item_input_ids.size(0)
            # pad extra dimension
            if cur_item < self.len_seq:
                b = self.len_seq - cur_item
                length = item_input_ids.size(1)
                pad = torch.zeros([b, length], dtype=item_input_ids.dtype)
                item_input_ids = torch.cat((item_input_ids, pad), dim=0)
                item_attention_mask = torch.cat((item_attention_mask, pad), dim=0)
            seq_input_ids.append(item_input_ids[None])  # 1 * n_passage * dim
            seq_attention_mask.append(item_attention_mask[None])
        seq_input_ids = torch.cat(seq_input_ids, dim=0)
        seq_attention_mask = torch.cat(seq_attention_mask, dim=0)
        query = (seq_input_ids, seq_attention_mask)

        # the possible items
        d_collated = self.tokenizer.pad(
            dd,
            padding="max_length",
            max_length=self.max_p_len,
            return_tensors="pt",
        )
        item_input_ids = d_collated.data["input_ids"]
        item_attention_mask = d_collated.data["attention_mask"]
        item_input_ids = torch.unsqueeze(item_input_ids, 1)  # 1 * passage_len
        item_attention_mask = torch.unsqueeze(item_attention_mask, 1)
        item = (item_input_ids, item_attention_mask)

        return query, item


class TasteTrainer(DRTrainer):
    def __init__(self, *args, val_dataset2=None, **kwargs):
        super(TasteTrainer, self).__init__(*args, **kwargs)
        self.val_dataset2 = val_dataset2

    def _prepare_inputs(
        self, inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], ...]
    ) -> List[Dict[str, Union[torch.Tensor, Any]]]:
        prepared = []
        for tup in inputs:
            x = tup[0].to(self.args.device)
            y = tup[1].to(self.args.device)
            prepared.append((x, y))
        return prepared


#    def evaluate(self, eval_dataset=None, **kwargs):
#        
#        metrics = super().evaluate(eval_dataset=self.eval_dataset, **kwargs)
#        
#        # Log to WandB
#        self.log(metrics)
#
#        if self.val_dataset2 is not None:
#            metrics2 = super().evaluate(eval_dataset=self.val_dataset2, **kwargs)
#            
#            # Customize logging for the second validation set
#            metrics2 = {f"val2_{k}": v for k, v in metrics2.items()}
#            self.log(metrics2)        
#
#        return metrics

class DTasteTrainDataset(TrainDatasetBase):
    """
    The training dataset for decoder-only TASTE with fast tokenizer
    """

    def create_one_example(
        self, text_encoding: List[int], is_query=False
    ) -> BatchEncoding:

        max_len = self.data_args.q_max_len if is_query else self.data_args.p_max_len

        # method 1: convert back to text
        text_encoding = self.tokenizer.convert_ids_to_tokens(text_encoding)
        text_encoding = self.tokenizer.convert_tokens_to_string(text_encoding)

        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation="only_first",
            max_length=max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        # add </eos> token
        if len(item["input_ids"]) < max_len:
            item["input_ids"] = item["input_ids"] + [self.tokenizer.eos_token_id]
        else:
            item["input_ids"][
                -1
            ] = self.tokenizer.eos_token_id  # further truncate to fit

        return item

    def get_process_fn(self, epoch, hashed_seed):

        def process_fn(example):
            encoded_query = []
            qry = example["query"]  # ids; text for tokenizer_fast
            for item in qry:
                encoded_query.append(self.create_one_example(item, True))
            encoded_passages = []
            group_positives = example["positives"]
            group_negatives = example["negatives"]

            if self.data_args.positive_passage_no_shuffle or hashed_seed is None:
                pos_psg = group_positives[0]
            else:
                pos_psg = group_positives[(hashed_seed + epoch) % len(group_positives)]
            encoded_passages.append(self.create_one_example(pos_psg))

            negative_size = self.data_args.train_n_passages - 1
            if len(group_negatives) < negative_size:
                if hashed_seed is not None:
                    negs = random.choices(group_negatives, k=negative_size)
                else:
                    negs = [x for x in group_negatives]
                    negs = negs * 2
                    negs = negs[:negative_size]
            elif self.data_args.train_n_passages == 1:
                negs = []
            elif self.data_args.negative_passage_no_shuffle:
                negs = group_negatives[:negative_size]
            else:
                _offset = epoch * negative_size % len(group_negatives)
                negs = [x for x in group_negatives]
                if hashed_seed is not None:
                    random.Random(hashed_seed).shuffle(negs)
                negs = negs * 2
                negs = negs[_offset : _offset + negative_size]

            for neg_psg in negs:
                encoded_passages.append(self.create_one_example(neg_psg))

            assert len(encoded_passages) == self.data_args.train_n_passages

            return {
                "query_": encoded_query,
                "passages": encoded_passages,
            }  # Avoid name conflict with query in the original dataset

        return process_fn


class TasteTrainDataset(TrainDatasetBase):
    """
    The training dataset for original TASTE with T5 tokenizer
    """

    def create_one_example(
        self, text_encoding: List[int], is_query=False
    ) -> BatchEncoding:
        # this shall add the <\s> token + truncate
        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation="only_first",
            max_length=(
                self.data_args.q_max_len if is_query else self.data_args.p_max_len
            ),
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def get_process_fn(self, epoch, hashed_seed):

        def process_fn(example):
            encoded_query = []
            qry = example["query"]  # ids; text for tokenizer_fast
            for item in qry:
                encoded_query.append(self.create_one_example(item, True))
            encoded_passages = []
            group_positives = example["positives"]
            group_negatives = example["negatives"]

            if self.data_args.positive_passage_no_shuffle or hashed_seed is None:
                pos_psg = group_positives[0]
            else:
                pos_psg = group_positives[(hashed_seed + epoch) % len(group_positives)]
            encoded_passages.append(self.create_one_example(pos_psg))

            negative_size = self.data_args.train_n_passages - 1
            if len(group_negatives) < negative_size:
                if hashed_seed is not None:
                    negs = random.choices(group_negatives, k=negative_size)
                else:
                    negs = [x for x in group_negatives]
                    negs = negs * 2
                    negs = negs[:negative_size]
            elif self.data_args.train_n_passages == 1:
                negs = []
            elif self.data_args.negative_passage_no_shuffle:
                negs = group_negatives[:negative_size]
            else:
                _offset = epoch * negative_size % len(group_negatives)
                negs = [x for x in group_negatives]
                if hashed_seed is not None:
                    random.Random(hashed_seed).shuffle(negs)
                negs = negs * 2
                negs = negs[_offset : _offset + negative_size]

            for neg_psg in negs:
                encoded_passages.append(self.create_one_example(neg_psg))

            assert len(encoded_passages) == self.data_args.train_n_passages

            return {
                "query_": encoded_query,
                "passages": encoded_passages,
            }  # Avoid name conflict with query in the original dataset

        return process_fn


class StreamDRDTrainDataset(StreamTrainDatasetMixin, DTasteTrainDataset):
    logging.info("using the stream dataset with fast tokenizer...")
    pass


class StreamDRTrainDataset(StreamTrainDatasetMixin, TasteTrainDataset):
    logging.info("using TASTE's original stream dataset with T5 tokenizer...")
    pass


class MappingDRDTrainDataset(MappingTrainDatasetMixin, DTasteTrainDataset):
    pass


class MappingDRTrainDataset(MappingTrainDatasetMixin, TasteTrainDataset):
    pass
