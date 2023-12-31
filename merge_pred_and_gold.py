import argparse
import json
import os
import sys
import time

import pandas as pd
import regex as re
import torch
from tqdm.auto import tqdm
from transformers import HfArgumentParser

from utils.load_dataset import creating_schema, get_context_with_db_name


def preprocess_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = text.replace(" ,  ", ", ")
    text = text.replace(" .  ", ". ")
    text = text.replace(" ,", ",")
    text = text.strip()
    return text


def is_identical(gold, pred):
    gold = gold.lower().strip()
    gold = preprocess_text(gold)

    pred = pred.lower().strip()
    pred = pred.replace('"', "'")

    return gold == pred


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file_path", type=str, default="./dev_predict.txt")
    parser.add_argument("--dataset_dir", type=str, default="./dataset/spider")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argparser()

    # load spider dataset: schema, primary key, foreign key
    spider_schema, spider_primary, spider_foreign = creating_schema(
        os.path.join(args.dataset_dir, "tables.json")
    )

    # load dev dataset
    dev_dataset = pd.read_json(os.path.join(args.dataset_dir, "dev.json"))

    data = []

    with open(args.pred_file_path, "r", encoding="utf-8") as f:
        pred_data = f.readlines()

    for idx, row in tqdm(dev_dataset.iterrows(), total=len(dev_dataset)):
        datapoint = {}

        context = get_context_with_db_name(
            row["db_id"], spider_schema, spider_primary, spider_foreign
        )
        question = row["question"]

        datapoint["context"] = context
        datapoint["question"] = question
        datapoint["gold"] = preprocess_text(row["query"])
        datapoint["pred"] = preprocess_text(pred_data[idx])
        datapoint["is_identical"] = is_identical(datapoint["gold"], datapoint["pred"])

        data.append(datapoint)

    with open(
        os.path.join(
            os.path.dirname(args.pred_file_path),
            os.path.basename(args.pred_file_path) + ".compare.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
