# importing necessary libraries
import os
import re

import joblib
import pandas as pd
import torch
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from transformers import HfArgumentParser

from utils.arguments import PredictArguments, TextToSqlTrainingArguments
from utils.load_dataset import get_context_with_db_name
from utils.load_model import load_model_with_peft_and_tokenizer
from utils.prompter import generate_llama_prompt_sql, generate_prompt_sql

load_dotenv()

PORT = os.environ.get("PORT", 5000)
DEBUG = os.environ.get("DEBUG", False)
JSON_CONFIG_PATH = os.environ.get("JSON_CONFIG_PATH", "config.json")

app = Flask(__name__)


@app.before_request
def log_request_info():
    app.logger.debug("Headers: %s", request.headers)
    app.logger.debug("Body: %s", request.get_data())


def preprocess_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text


def get_sql_statement(prediction):
    idx = prediction.find("### Response:\n")
    prediction = prediction[idx + len("### Response:\n") :].strip()
    prediction = preprocess_text(prediction)
    return prediction


def get_llama_sql_statement(prediction):
    idx = prediction.find("[/INST] ")
    prediction = prediction[idx + len("[/INST] ") :].strip()
    prediction = preprocess_text(prediction)
    return prediction


@torch.no_grad()
def predict(model, tokenizer, prompt, device="cuda", args={}):
    model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=args.max_new_tokens,
        # num_beams=5,
        # early_stopping=True,
        # num_return_sequences=1,
    )
    return tokenizer.batch_decode(
        outputs.detach().cpu().numpy(), skip_special_tokens=True
    )


@app.route("/api/text2sql/ask", methods=["POST"])
def ask_text2sql():
    # Get the data from the POST request.
    data = request.get_json(force=True)

    context = data["context"]
    question = data["question"]

    # log data to the console so we can see what we received

    if PREDICT_ARGS.use_llama_prompt:
        prompt = generate_llama_prompt_sql(question, context)
    else:
        prompt = generate_prompt_sql(question, context)

    predictions = predict(MODEL, TOKENIZER, prompt, DEVICE, args=PREDICT_ARGS)

    for ids, prediction in enumerate(predictions):
        if PREDICT_ARGS.use_llama_prompt:
            prediction = get_llama_sql_statement(prediction)
        else:
            prediction = get_sql_statement(prediction)
        predictions[ids] = prediction

    response = {"answers": predictions}

    return jsonify(response)


@app.route("/api/homepage", methods=["POST", "GET"])
def homepage():
    return jsonify({"message": "Hello World!"})


if __name__ == "__main__":
    global MODEL, TOKENIZER, DEVICE, PREDICT_ARGS

    parser = HfArgumentParser((PredictArguments))
    PREDICT_ARGS = parser.parse_json_file(json_file=JSON_CONFIG_PATH)[0]

    print(f"Loading app with arguments {PREDICT_ARGS}")

    MODEL, TOKENIZER = load_model_with_peft_and_tokenizer(PREDICT_ARGS, None)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=DEBUG,
    )
