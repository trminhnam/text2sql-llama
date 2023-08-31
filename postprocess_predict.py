import os
import json
import re

FILE_PATH = "./dev_predict.txt"


def postprocess(text):
    phrases = ["```sql", "```", "```sql\n", "```sql\r\n", "```sql\r"]
    text = text.strip()
    for phrase in phrases:
        if phrase in text:
            start_idx = text.find(phrase)
            start_idx = start_idx + len(phrase)
            text = text[start_idx:]

            end_idx = text.find("```")
            if end_idx != -1:
                text = text[:end_idx]

            text = text.strip()
            return text

    return text


with open(FILE_PATH, "r", encoding="utf-8") as f:
    data = f.readlines()
    for line in data:
        line = postprocess(line)

CORRECTED_FILE_PATH = os.path.join(
    os.path.dirname(FILE_PATH), "dev_predict_corrected.txt"
)
with open(CORRECTED_FILE_PATH, "w", encoding="utf-8") as f:
    for line in data:
        line = postprocess(line)
        f.write(line + "\n")
