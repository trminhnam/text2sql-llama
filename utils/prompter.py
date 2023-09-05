def generate_prompt_sql(question, context, answer=""):
    return f"""You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

You must output the SQL query that answers the question.

### Question:
{question}

### Context:
{context}

### Response:
{answer}"""


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SQL_PROMPT = """You are a powerful text-to-SQL model. You are given a question and context regarding one or more tables. Your job is output the SQL query that answers the question."""


def generate_llama_prompt_sql(question, context, answer=""):
    return f"""{B_INST} {B_SYS}{SQL_PROMPT.strip()}{E_SYS}### Question:
{question}

### Context:
{context} {E_INST} {answer}"""
