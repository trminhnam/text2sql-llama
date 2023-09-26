import argparse
import os
import sys
import time

import pandas as pd
import regex as re
import torch
from tqdm.auto import tqdm
from transformers import HfArgumentParser

from utils.arguments import (
    PredictArguments,
    TextToSqlTrainingArguments,
    LlamaCppArguments,
)
from utils.load_dataset import creating_schema, get_context_with_db_name
from utils.load_model import load_model_with_peft_and_tokenizer, load_llama_cpp_model
from utils.prompter import generate_llama_prompt_sql, generate_prompt_sql
from utils.timeout import Timeout, timeout

try:
    from llama_cpp import Llama

    imported_llama_cpp = True
except ImportError:
    print("Failed to import Llama. Please install it first.")
    imported_llama_cpp = False


@torch.no_grad()
def predict(model, tokenizer, prompt, device="cuda", args={}):
    model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        # num_return_sequences=1,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        # do_sample=args.do_sample,
        # num_beams=args.num_beams,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        max_time=args.max_time,
    )
    return tokenizer.batch_decode(
        outputs.detach().cpu().numpy(), skip_special_tokens=True
    )[0]


def llama_cpp_predict(model, prompt, args={}):
    if args.max_new_tokens is not None:
        prompt_n_tokens = len(model.tokenize(prompt.encode("utf-8")))
        max_tokens = args.max_new_tokens + prompt_n_tokens
    else:
        max_tokens = args.max_length

    model_output = model(
        prompt,
        echo=True,
        max_tokens=max_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
    )

    return model_output["choices"][0]["text"]


def preprocess_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text


def get_sql_statement(prediction):
    idx = prediction.find("### Response:\n")
    prediction = prediction[idx + len("### Response:\n") :].strip()
    # if "\n\n" in prediction:
    #     prediction = prediction.split("\n\n")[0].strip()
    # if ";" in prediction:
    #     prediction = prediction.split(";")[0].strip()
    prediction = preprocess_text(prediction)
    return prediction


def get_llama_sql_statement(prediction):
    idx = prediction.find("[/INST] ")
    prediction = prediction[idx + len("[/INST] ") :].strip()
    prediction = preprocess_text(prediction)
    return prediction


if __name__ == "__main__":
    parser = HfArgumentParser(
        (
            PredictArguments,
            LlamaCppArguments,
            TextToSqlTrainingArguments,
        )
    )
    predict_args, llama_cpp_args, training_args = parser.parse_args_into_dataclasses()
    print(f"Predict args: {predict_args}")
    print("#" * 50)
    print()

    print(f"Llama-cpp args: {llama_cpp_args}")
    print("#" * 50)
    print()

    # make output dirs
    base_dir = os.path.dirname(predict_args.output_path)
    file_name = os.path.basename(predict_args.output_path)
    save_dir = (
        base_dir
        if not predict_args.peft_name_or_path_subfolder
        else os.path.join(base_dir, predict_args.peft_name_or_path_subfolder)
    )
    if predict_args.num_beams > 1:
        save_dir = os.path.join(save_dir, f"beam_{predict_args.num_beams}")
    os.makedirs(save_dir, exist_ok=True)
    predict_args.output_path = os.path.join(save_dir, file_name)
    if llama_cpp_args.llama_cpp_model_path:
        model_filename = os.path.basename(llama_cpp_args.llama_cpp_model_path)
        predict_args.output_path = os.path.join(
            save_dir, model_filename + "." + file_name
        )

    print(f"Output path to: {predict_args.output_path}")

    # load spider dataset: schema, primary key, foreign key
    spider_schema, spider_primary, spider_foreign = creating_schema(
        os.path.join(predict_args.dataset_dir, "tables.json")
    )

    # load dev dataset
    dev_dataset = pd.read_json(os.path.join(predict_args.dataset_dir, "dev.json"))

    # load model and tokenizer with path
    if llama_cpp_args.llama_cpp_model_path:
        # load model and tokenizer instead of loading with default function
        model = Llama(
            model_path=llama_cpp_args.llama_cpp_model_path,
            n_ctx=llama_cpp_args.n_ctx,
            n_gpu_layers=llama_cpp_args.n_gpu_layers,
        )
        print(f"Use llama-cpp model from {llama_cpp_args.llama_cpp_model_path}")
    else:
        model, tokenizer = load_model_with_peft_and_tokenizer(
            predict_args, training_args
        )
        device = (
            "cuda"
            if (torch.cuda.is_available() and not training_args.no_cuda)
            else "cpu"
        )
        model.to(device)

    predictions = []
    for idx, row in tqdm(dev_dataset.iterrows(), total=len(dev_dataset)):
        context = get_context_with_db_name(
            row["db_id"], spider_schema, spider_primary, spider_foreign
        )
        question = row["question"]

        if predict_args.use_llama_prompt:
            prompt = generate_llama_prompt_sql(question, context)
        else:
            prompt = generate_prompt_sql(question, context)
        try:
            if llama_cpp_args.llama_cpp_model_path:
                prediction = llama_cpp_predict(model, prompt, args=predict_args)
            else:
                prediction = predict(
                    model, tokenizer, prompt, device, args=predict_args
                )

            if predict_args.use_llama_prompt:
                prediction = get_llama_sql_statement(prediction)
            else:
                prediction = get_sql_statement(prediction)
        except Exception as e:
            print(f"Failed to predict {idx}-th question")
            print(f"Question: {question}")
            print(f"Context: {context}")
            print(f"Error: {e}")
            print("*" * 50)
            print()
            prediction = ""

        predictions.append(prediction)

        if idx % 100 == 0:
            print(f"Predicted {idx}-th question")
            print(f"Question: {question}")
            print(f"Context: {context}")
            print(f"Prediction: {prediction}")
            print(f"Label: {row['query']}")
            print("*" * 50)
            print()

        with open(predict_args.output_path + ".log", "a", encoding="utf-8") as f:
            f.write(prediction + "\n")
        # exit()

    with open(predict_args.output_path, "w", encoding="utf-8") as f:
        for prediction in predictions:
            f.write(prediction + "\n")

    print(f"Saved predictions to {predict_args.output_path}")
