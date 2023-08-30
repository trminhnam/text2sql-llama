import pandas as pd
import time
import os
import sys
import argparse
import torch

from tqdm.auto import tqdm

from transformers import HfArgumentParser


from utils.prompter import generate_prompt_sql
from utils.load_dataset import creating_schema, get_context_with_db_name
from utils.load_model import load_model_with_peft_and_tokenizer
from utils.arguments import (
    TextToSqlTrainingArguments,
    PredictArguments,
)


@torch.no_grad()
def predict(model, tokenizer, prompt, device="cuda", args={}):
    model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.batch_decode(
        outputs.detach().cpu().numpy(), skip_special_tokens=True
    )[0]


if __name__ == "__main__":
    parser = HfArgumentParser(
        (
            PredictArguments,
            TextToSqlTrainingArguments,
        )
    )
    predict_args, training_args = parser.parse_args_into_dataclasses()

    # make output dirs
    os.makedirs(os.path.dirname(predict_args.output_path), exist_ok=True)

    # load spider dataset: schema, primary key, foreign key
    spider_schema, spider_primary, spider_foreign = creating_schema(
        predict_args.dataset_dir + "tables.json"
    )

    # load dev dataset
    dev_dataset = pd.read_json(predict_args.dataset_dir + "dev.json")

    # load model and tokenizer with path
    model, tokenizer = load_model_with_peft_and_tokenizer(predict_args, training_args)
    device = "cuda" if torch.cuda.is_available() and not predict_args.no_cuda else "cpu"
    model.to(device)

    predictions = []
    for idx, row in tqdm(dev_dataset.iterrows(), total=len(dev_dataset)):
        context = get_context_with_db_name(
            row["db_id"], spider_schema, spider_primary, spider_foreign
        )
        question = row["question"]

        prompt = generate_prompt_sql(question, context)
        prediction = predict(model, tokenizer, prompt, device, args=predict_args)
        predictions.append(prediction)

        if idx % 100 == 0:
            print(f"Predicted {idx} questions")
            print(f"Question: {question}")
            print(f"Prediction: {prediction}")
            print(f"Correct: {row['query']}")
            print()

    with open(predict_args.output_path, "w") as f:
        for prediction in predictions:
            f.write(prediction + "\n")
