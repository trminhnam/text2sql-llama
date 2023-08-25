import re
import os
import sys
import glob
import math
import logging
import argparse
import numpy as np

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

from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple


@dataclass
class ModelArguments:
    # TODO: Base model parameters
    model_name_or_path: Optional[str] = field(default=None)

    # TODO: quantization parameters
    load_in_8bit: bool = field(
        default=False,
        metadata={
            "help": "Whether to convert the loaded model into mixed-8bit quantized model."
        },
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={
            "help": "Whether to convert the loaded model into mixed-4bit quantized model."
        },
    )
    bnb_4bit_quant_type: str = field(
        default="fp4",
        metadata={
            "help": "bnb_4bit_quant_type (`str`, {fp4, nf4}, defaults to `fp4`):"
            " This sets the quantization data type in the bnb.nn.Linear4Bit layers. Options are FP4 and NF4 data types"
            " which are specified by `fp4` or `nf4`."
        },
    )
    bnb_4bit_compute_dtype: str = field(
        default="float32",
        metadata={
            "help": "The compute dtype of the model. Can be float32, fp32, float16, fp16"
            " bfloat16, bf16."
        },
    )
    bnb_4bit_use_double_quant: bool = field(
        default=False,
        metadata={
            "help": "Whether to use double quantization for mixed-4bit quantized model."
        },
    )

    # TODO: LoRA parameters
    lora_r: int = field(default=8, metadata={"help": "Lora rank."})
    lora_alpha: int = field(default=16, metadata={"help": "Lora alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "Lora dropout."})
    lora_bias: str = field(
        default="none",
        metadata={
            "help": "Lora bias (`str`, {none, lora_only, all}, defaults to `none`):"
            " If 'all' or 'lora_only', the corresponding biases will be updated during training."
            " Be aware that this means that, even when disabling the adapters,"
            " the model will not produce the same output as the base model would have without adaptation."
        },
    )
    lora_target_modules: str = field(
        default="q_proj,v_proj",
        metadata={"help": "Names of the modules to apply Lora to."},
    )


@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )

    model_max_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    val_set_size: Optional[int] = field(
        default=2000, metadata={"help": "The validation set size. For loss checking."}
    )


@dataclass
class TextToSqlTrainingArguments(TrainingArguments):
    pass