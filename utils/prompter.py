def generate_prompt_sql(question, context, answer=""):
    return f"""You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables. 

You must output the SQL query that answers the question.

### Question:
{question}

### Context:
{context}

### Response:
{answer}"""
