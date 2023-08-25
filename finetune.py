import logging
import math
import os
import sys

import datasets
import transformers
from datasets import load_dataset
from transformers import HfArgumentParser, Trainer
from transformers.trainer_utils import get_last_checkpoint

from utils.arguments import DataArguments, ModelArguments, TextToSqlTrainingArguments
from utils.load_model import load_model_with_peft_and_tokenizer
from utils.prompter import generate_prompt_sql
from utils.another_utils import set_seed

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

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
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
        cache_dir=model_args.cache_dir,
    )

    # Determine model_max_length for truncation
    model_max_length = data_args.model_max_length

    if data_args.val_set_size > 0 and "validation" not in dataset.keys():
        if "test" not in dataset.keys():
            train_val_data = dataset["train"].train_test_split(
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
            data_point["question"],
            data_point["context"],
            data_point["answer"],
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
            tokenizer, pad_to_multiple_of=1, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_data)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_data))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(val_data)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(val_data))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "text-generation",
        # "peft_type": PEFT_TYPE_MAPPING_CONFIG[peft_args.peft_type][1],
    }
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs[
                "dataset"
            ] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    # ).__get__(model, type(model))

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    # trainer.train()

    # model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
