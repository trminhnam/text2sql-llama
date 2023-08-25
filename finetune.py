import re
import os
import sys
import glob
import math
import logging
import argparse
import numpy as np
from typing import Dict, List, Optional, Sequence
from dataclasses import dataclass, field

import torch
import datasets
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    Trainer,
    HfArgumentParser,
    TrainingArguments,
)
from datasets import load_dataset, concatenate_datasets, DatasetDict

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,
)

from utils.arguments import (
    ModelArguments,
    DataArguments,
    TextToSqlTrainingArguments,
)
from utils.load_model import load_model_with_peft_and_tokenizer
from utils.prompter import generate_prompt_sql

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def train():
    # HF parser
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            TextToSqlTrainingArguments,
        )
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # TODO: Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    if "wandb" in training_args.report_to:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project

    # TODO: load model with peft and tokenizer
    model, tokenizer = load_model_with_peft_and_tokenizer(
        model_args,
        training_args,
    )

    # TODO: Load dataset from HF Hub
    dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
    )

    # Determine model_max_length for truncation
    model_max_length = data_args.model_max_length

    if data_args.val_set_size > 0 and "validation" not in dataset.keys():
        if "test" not in dataset.keys():
            train_val_data = datasets["train"].train_test_split(
                test_size=data_args.val_set_size, shuffle=True, seed=42
            )
        train_val_data["validation"] = train_val_data["test"]
    else:
        raise ValueError("val_set_size must large than 0.")

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=model_max_length,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < model_max_length
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt_sql(
            data_point["input"],
            data_point["context"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        # if not train_on_inputs:
        #     raise NotImplementedError("not implemented yet")
        return tokenized_full_prompt

    # with training_args.main_process_first(desc="dataset map tokenization"):
    train_data = train_val_data["train"].map(
        generate_and_tokenize_prompt,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=next(iter(dataset.values())).column_names,
        desc="preprocess train data set",
    )
    val_data = train_val_data["validation"].map(
        generate_and_tokenize_prompt,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=next(iter(dataset.values())).column_names,
        desc="preprocess val data set",
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()

    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
