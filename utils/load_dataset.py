import json
import pandas as pd


def creating_schema(DATASET_JSON):
    schema_df = pd.read_json(DATASET_JSON)
    schema_df = schema_df.drop(["column_names", "table_names"], axis=1)
    schema = []
    f_keys = []
    p_keys = []
    for index, row in schema_df.iterrows():
        tables = row["table_names_original"]
        col_names = row["column_names_original"]
        col_types = row["column_types"]
        foreign_keys = row["foreign_keys"]
        primary_keys = row["primary_keys"]
        for col, col_type in zip(col_names, col_types):
            index, col_name = col
            if index == -1:
                for table in tables:
                    schema.append([row["db_id"], table, "*", "text"])
            else:
                schema.append([row["db_id"], tables[index], col_name, col_type])
        for primary_key in primary_keys:
            index, column = col_names[primary_key]
            p_keys.append([row["db_id"], tables[index], column])
        for foreign_key in foreign_keys:
            first, second = foreign_key
            first_index, first_column = col_names[first]
            second_index, second_column = col_names[second]
            f_keys.append(
                [
                    row["db_id"],
                    tables[first_index],
                    tables[second_index],
                    first_column,
                    second_column,
                ]
            )
    spider_schema = pd.DataFrame(
        schema, columns=["Database name", " Table Name", " Field Name", " Type"]
    )
    spider_primary = pd.DataFrame(
        p_keys, columns=["Database name", "Table Name", "Primary Key"]
    )
    spider_foreign = pd.DataFrame(
        f_keys,
        columns=[
            "Database name",
            "First Table Name",
            "Second Table Name",
            "First Table Foreign Key",
            "Second Table Foreign Key",
        ],
    )
    return spider_schema, spider_primary, spider_foreign


def convert_type_to_sql_type(type):
    if type == "text":
        return "VARCHAR"
    elif type == "integer" or type == "number" or type == "int":
        return "INTEGER"
    elif type == "time":
        return "DATETIME"
    elif type == "boolean":
        return "BOOLEAN"
    elif type == "real" or type == "float" or type == "double":
        return "FLOAT"
    elif type == "others":
        return "BOOLEAN"
    else:
        return "VARCHAR"


def get_context_with_db_name(db_name, spider_schema, spider_primary, spider_foreign):
    # find all tables related to db_name
    df = spider_schema[spider_schema["Database name"] == db_name]
    df = df.groupby(" Table Name")
    tables = []
    for name, group in df:
        table = {}
        table["name"] = name
        table["columns"] = []
        for index, row in group.iterrows():
            table["columns"].append(
                (row[" Field Name"], convert_type_to_sql_type(row[" Type"]))
            )
        tables.append(table)

    # for each table, create the "CREATE TABLE" statement and append it to context
    statements = []
    for table in tables:
        statement = "CREATE TABLE " + table["name"] + " ("
        for idx, column in enumerate(table["columns"]):
            col_name = column[0]
            col_type = column[1]
            if col_name == "*":
                continue
            if " " in col_name:
                col_name = '"' + col_name + '"'

            statement += col_name + " " + col_type
            if idx != len(table["columns"]) - 1:
                statement += ", "
        statement = statement + ")"
        statements.append(statement)

    # print("; ".join(statements))
    return "; ".join(statements)
