import copy
import json
import logging
import os
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from openmatch.arguments import (
    DataArguments,
    DRTrainingArguments as TrainingArguments,
    ModelArguments,
)
from openmatch.modeling import DROutput
from src.model import PythiaTASTEModel
from src.taste_argument import TASTEArguments
from torch import Tensor

from transformers import AutoModel, PreTrainedModel

logger = logging.getLogger(__name__)


class DRD4RecModel(nn.Module):
    def __init__(
        self,
        lm_q: PreTrainedModel,
        lm_p: PreTrainedModel,
        tied: bool = True,
        feature: str = "last_hidden_state",
        pooling: str = "first",
        head_q: nn.Module = None,
        head_p: nn.Module = None,
        normalize: bool = False,
        model_args: ModelArguments = None,
        data_args: DataArguments = None,
        train_args: TrainingArguments = None,
        taste_args: TASTEArguments = None,
    ):
        super().__init__()

        self.tied = tied
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.head_q = head_q
        self.head_p = head_p

        self.feature = feature
        self.pooling = pooling
        self.normalize = normalize

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args
        self.taste_args = taste_args

        self._keys_to_ignore_on_save = None  # for checkpoint resuming

        if train_args is not None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="mean")
            if train_args.negatives_x_device:
                if not dist.is_initialized():
                    raise ValueError(
                        "Distributed training has not been initialized for representation all gather."
                    )
                self.process_rank = dist.get_rank()
                self.world_size = dist.get_world_size()

    def _get_config_dict(self):
        config = {
            "tied": self.tied,
            "plm_backbone": {
                "type": type(self.lm_q).__name__,
                "feature": self.feature,
            },
            "pooling": self.pooling,
            "linear_head": bool(self.head_q),
            "normalize": self.normalize,
        }
        return config

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
        positive: Dict[str, Tensor] = None,
        negative: Dict[str, Tensor] = None,
        score: Tensor = None,
    ):

        # same -> last token representation through the whole model
        q_hidden, q_reps = self.encode_query(query)
        p_hidden, p_reps = self.encode_passage(passage)

        scores = torch.matmul(q_reps, p_reps.transpose(0, 1))

        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * self.data_args.train_n_passages

        loss = self.loss_fn(scores, target)

        if self.training and self.train_args.negatives_x_device:
            loss = loss * self.world_size  # counter average weight reduction
        return DROutput(loss=loss, scores=scores, q_reps=q_reps, p_reps=p_reps)

    def encode(self, items, model, head):
        if items is None:
            return None, None
        input_ids = items[0]
        attention_mask = items[1]

        hidden, reps = model(
            input_ids, attention_mask
        )  # hidden states, 1st token embedding

        if head is not None:
            reps = head(reps)  # D * d
        if self.normalize:
            reps = F.normalize(reps, dim=1)
        return hidden, reps

    def encode_passage(self, psg):
        return self.encode(psg, self.lm_p, self.head_p)

    def encode_query(self, qry):
        return self.encode(qry, self.lm_q, self.head_q)

    def load_state_dict(self, state_dict, *args, **kwargs):
        load_result = self.lm_q.load_state_dict(state_dict, *args, **kwargs)
        self.lm_p = self.lm_q  # copy weight also
        return load_result

    @classmethod
    def build(
        cls,
        model_args: ModelArguments,
        model_name_or_path: str = None,
        data_args: DataArguments = None,
        train_args: TrainingArguments = None,
        taste_args: TASTEArguments = None,
        **hf_kwargs,
    ):
        model_name_or_path = model_name_or_path or model_args.model_name_or_path
        # load local
        config = None
        head_q = head_p = None
        if os.path.exists(os.path.join(model_name_or_path, "openmatch_config.json")):
            with open(os.path.join(model_name_or_path, "openmatch_config.json")) as f:
                config = json.load(f)
        tied = not model_args.untie_encoder

        # cannot do encoder_only; no seq2seq, omit the embed_out weights
        base_model = AutoModel.from_pretrained(model_name_or_path, **hf_kwargs)
        # a pass on these encoders should go to the AutoModel, not the custom model -> no halving passages
        lm_q = PythiaTASTEModel(base_model.config)
        lm_q.load_weights(base_model.state_dict())  # update the weights
        lm_p = copy.deepcopy(lm_q) if not tied else lm_q  # tied == True
        del base_model

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            tied=tied,
            feature=(
                model_args.feature
                if config is None
                else config["plm_backbone"]["feature"]
            ),
            pooling=model_args.pooling if config is None else config["pooling"],
            head_q=head_q,
            head_p=head_p,
            normalize=model_args.normalize if config is None else config["normalize"],
            model_args=model_args,
            data_args=data_args,
            train_args=train_args,
            taste_args=taste_args,
        )
        return model

    def save(self, output_dir: str):
        if not self.tied:
            os.makedirs(os.path.join(output_dir, "query_model"))
            os.makedirs(os.path.join(output_dir, "passage_model"))
            self.lm_q.save_pretrained(os.path.join(output_dir, "query_model"))
            self.lm_p.save_pretrained(os.path.join(output_dir, "passage_model"))
            if self.head_q is not None:
                self.head_q.save(os.path.join(output_dir, "query_head"))
                self.head_p.save(os.path.join(output_dir, "passage_head"))
        else:
            self.lm_q.save_pretrained(output_dir)
            if self.head_q is not None:
                self.head_q.save(output_dir)

        with open(os.path.join(output_dir, "openmatch_config.json"), "w") as f:
            json.dump(self._get_config_dict(), f, indent=4)
