from typing import Dict

import torch
import transformers
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    CodeLlamaTokenizer,
    LlamaTokenizer,
)

COMPUTE_DTYPE_MAPPING = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


# Copied from https://github.com/bofenghuang/stanford_alpaca/blob/eb5b171d9b103a12a8e14e0edca9cbc45fe1d512/train.py#L75-L95
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def has_attr_and_true(obj, attr):
    return hasattr(obj, attr) and getattr(obj, attr)


def load_model_with_peft_and_tokenizer(model_args, training_args):
    # TODO: prepare quantization config
    bnb_config = None
    # if model_args.load_in_8bit or model_args.load_in_4bit:
    if (
        has_attr_and_true(model_args, "load_in_8bit")
        or has_attr_and_true(model_args, "load_in_4bit")
    ) and not has_attr_and_true(model_args, "quant_mode"):
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=model_args.load_in_8bit,
            load_in_4bit=model_args.load_in_4bit,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=COMPUTE_DTYPE_MAPPING[
                model_args.bnb_4bit_compute_dtype
            ],
            bnb_4bit_use_double_quant=model_args.bnb_4bit_use_double_quant,
        )
        print(f"Quantization config: {bnb_config}")

    # TODO: load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        subfolder=model_args.model_name_or_path_subfolder,
        quantization_config=bnb_config,
        cache_dir=model_args.cache_dir,
        device_map="auto",
        offload_folder=".",
    )

    # TODO: load tokenizer
    if "codellama" in model_args.model_name_or_path:
        tokenizer_class = CodeLlamaTokenizer
    else:
        tokenizer_class = (
            LlamaTokenizer
            if "llama" in model_args.model_name_or_path
            else AutoTokenizer
        )
    tokenizer = tokenizer_class.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        use_fast=False,
    )

    # TODO: add special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id

    # TODO: prepare model for kbit training
    if not has_attr_and_true(model_args, "quant_mode"):
        if has_attr_and_true(model_args, "load_in_8bit") or has_attr_and_true(
            model_args, "load_in_4bit"
        ):
            model = prepare_model_for_kbit_training(
                model=model,
                use_gradient_checkpointing=training_args.gradient_checkpointing
                if training_args is not None
                else False,
            )
        else:
            model.enable_input_require_grads()

    # TODO attach lora to the model
    print("#" * 20)
    if model_args.peft_name_or_path is not None:
        peft_model_id = model_args.peft_name_or_path
        config = PeftConfig.from_pretrained(
            peft_model_id, subfolder=model_args.peft_name_or_path_subfolder
        )
        # model = (
        #     AutoModelForCausalLM.from_pretrained(
        #         config.base_model_name_or_path,
        #         quantization_config=bnb_config,
        #         cache_dir=model_args.cache_dir,
        #     )
        #     if config.base_model_name_or_path != model_args.model_name_or_path
        #     else model
        # )
        model = PeftModel.from_pretrained(
            model,
            peft_model_id,
            is_trainable=True,
            subfolder=model_args.peft_name_or_path_subfolder,
            device_map="auto",
            offload_dir="",
            offload_folder=".",
        )
        print(
            f"Loaded PEFT model from {peft_model_id}/{model_args.peft_name_or_path_subfolder}"
            if model_args.peft_name_or_path_subfolder
            else f"Loaded PEFT model from {peft_model_id}"
        )
        model.print_trainable_parameters()
    elif has_attr_and_true(model_args, "lora_r"):
        config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.lora_target_modules.split(","),
            lora_dropout=model_args.lora_dropout,
            bias=model_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        print("Loaded PEFT model from scratch")
        model.print_trainable_parameters()
    else:
        print("No PEFT model is loaded")
        print("Using default config")

    return model, tokenizer
