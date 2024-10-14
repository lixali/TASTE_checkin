import logging
import os
import sys
#import wandb

from openmatch.arguments import (
    DataArguments,
    DRTrainingArguments as TrainingArguments,
    ModelArguments,
)

from openmatch.trainer import GCDenseTrainer
from openmatch.utils import get_delta_model_class

from src.taste_argument import TASTEArguments
from src.taste_auto_model import DRD4RecModel
from src.taste_model import DR4RecModel
from src.trainer import (
    DTasteCollator,
    MappingDRDTrainDataset,
    MappingDRTrainDataset,
    StreamDRDTrainDataset,
    StreamDRTrainDataset,
    TasteCollator,
    TasteTrainer,
)

from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed

# no parallelism for tokenizer to avoid possible deadlock; TokenizerFast onlys
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    #os.environ["WANDB_DISABLED"] = "true"
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, TASTEArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, taste_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, taste_args = (
            parser.parse_args_into_dataclasses()
        )
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments
        taste_args: TASTEArguments


    if not os.path.exists(training_args.logging_dir): os.makedirs(training_args.logging_dir)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # Set the logger level
    fh = logging.FileHandler(os.path.join(training_args.logging_dir,'train.log'), mode='w')
    fh.setLevel(logging.INFO)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    fh.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(fh)


    # wandb_project_name = None
    # if training_args.output_dir:
    #     run_name = training_args.output_dir.split("/")[1]
    #     print(f"run_name is {run_name}")
    #     wandb_project_name = f"taste_{run_name}"
    #     logger.info(f"wandb_project_name is {wandb_project_name}")
    #     base_path = f"out/taste/{wandb_project_name}"
    #     wandb_project = wandb_project_name
    #     wandb_run_name =  f"{wandb_project_name}"
    #     out_dir = f"/data/user_data/lixiangl/ads_content_understanding/TASTE_checkin/{wandb_project_name}"

    #     wandb.init(project=wandb_project, name=wandb_run_name, dir=out_dir)

    val_dataset2 = None 
    checkpoint = None
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
        and not training_args.resume_from_checkpoint
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        # check the latest checkpoint
        checkpoints = sorted(os.listdir(training_args.output_dir), key=lambda x: os.path.getctime(os.path.join(training_args.output_dir, x)))
        if checkpoints[-1] == "best_dev":  # ignore any temporary evaluation result
            checkpoints = checkpoints[:-1]
        checkpoint = os.path.join(training_args.output_dir, checkpoints[-1])
        logger.info(
            f"---- try resuming from checkpoint: {training_args.resume_from_checkpoint}... ----"
        )
    logger.info(f"---- checkpoint: {checkpoint} ----")

    #breakpoint()
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)
    logger.info("data info %s", data_args)
    logger.info("taste args %s", taste_args)

    set_seed(training_args.seed)

    num_labels = 1
    config = AutoConfig.from_pretrained(
        (
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path
        ),
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )

    # handle for decoder structure difference
    if (
        "pythia" in model_args.model_name_or_path
        or "llama" in model_args.model_name_or_path
    ):
        use_fast_tokenizer = True  # tokenizer conifg
        target_taste_model = DRD4RecModel
    else:
        use_fast_tokenizer = False
        target_taste_model = DR4RecModel

    # intialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path
        ),
        cache_dir=model_args.cache_dir,
        use_fast=use_fast_tokenizer,
    )
    
    # specific for GPT-based and Llama
    if use_fast_tokenizer:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # initialize model
    model = target_taste_model.build(
        model_args=model_args,
        data_args=data_args,
        train_args=training_args,
        taste_args=taste_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    if model_args.param_efficient_method:
        model_class = get_delta_model_class(model_args.param_efficient_method)
        delta_model = model_class(model)
        logger.info(
            "Using param efficient method: %s", model_args.param_efficient_method
        )

    # separate training dataset format for PretrainedTokenizerFast and PretrainedTokenizerBase
    if use_fast_tokenizer:
        train_dataset_cls = (
            MappingDRDTrainDataset
            if training_args.use_mapping_dataset
            else StreamDRDTrainDataset
        )
    else:
        train_dataset_cls = (
            MappingDRTrainDataset
            if training_args.use_mapping_dataset
            else StreamDRTrainDataset
        )

    # separate data collator for PretrainedToeknizerFast and PretrainedTokenizerBase
    if use_fast_tokenizer:
        data_collator_cls = DTasteCollator(
            tokenizer,
            max_p_len=data_args.p_max_len,
            max_q_len=data_args.q_max_len,
            len_seq=taste_args.num_passages,
        )
    else:
        data_collator_cls = TasteCollator(
            tokenizer,
            max_p_len=data_args.p_max_len,
            max_q_len=data_args.q_max_len,
            len_seq=taste_args.num_passages,
        )
    training_args.report_to = ["wandb"]
    # training_args.report_to = ["tensorboard"]
    # training_args.logging_dir = training_args.output_dir + "/tensorboard_output/"
    # initialize datasets

    # if wandb_project_name:
    if True:
        os.environ["WANDB_PROJECT"]= "taste_run_self_rewarding"

        # do not save your trained model checkpoint to wandb
        os.environ["WANDB_LOG_MODEL"]="false"

        # turn off watch to log faster
        os.environ["WANDB_WATCH"]="false"


    train_dataset = train_dataset_cls(
        tokenizer,
        data_args,
        shuffle_seed=training_args.seed,
        cache_dir=data_args.data_cache_dir or model_args.cache_dir,
    )
    eval_dataset = (
        train_dataset_cls(
            tokenizer,
            data_args,
            is_eval=True,
            cache_dir=data_args.data_cache_dir or model_args.cache_dir,
        )
        if data_args.eval_path is not None
        else None
    )

    val_dataset2 = (
        train_dataset_cls(
            tokenizer,
            data_args,
            is_eval=True,
            cache_dir=data_args.data_cache_dir or model_args.cache_dir,
        )
        if taste_args.val_dataset2 is not None
        else None
    )
    
    eval_dict = None
    if val_dataset2 is not None:
        eval_dict = {"eval_synthetic": eval_dataset ,  "eval_amzonreview": val_dataset2}
    # os.environ["WANDB_DISABLED"] = "true"

    # initialize trainer
    trainer_cls = GCDenseTrainer if training_args.grad_cache else TasteTrainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dict if eval_dict is not None else eval_dataset,
        data_collator=data_collator_cls,
        delta_model=delta_model if model_args.param_efficient_method else None,
    )
    train_dataset.trainer = trainer
    logger.info(f"##### checkpoint is {checkpoint} ########")
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
