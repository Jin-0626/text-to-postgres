from pathlib import Path
import sqlite3
import re
import pandas as pd
import sqlparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from tabulate import tabulate

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_dir = "./pg_sql_rtx3060_adapter"
dataset_name = "jin-0626/postgresql-employees-assistant"

# 1. Load Tokenizer & Holdout Split
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

test_dataset = load_dataset("json", data_files=str(Path(__file__).resolve().parent / "converted" / "test.jsonl"), split="train")

# 2. Setup Quantized Base and Adapter
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_dir)

# 3. SQLite In-Memory Mock Database
def create_mock_db():
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    # Create SQLite-compatible version of your schema for execution checking
    cursor.executescript("""
        CREATE TABLE employee (id INTEGER PRIMARY KEY, birth_date TEXT, first_name TEXT, last_name TEXT, gender TEXT, hire_date TEXT);
        CREATE TABLE department (id TEXT PRIMARY KEY, dept_name TEXT UNIQUE);
        CREATE TABLE department_employee (employee_id INTEGER, department_id TEXT, from_date TEXT, to_date TEXT);
        CREATE TABLE department_manager (employee_id INTEGER, department_id TEXT, from_date TEXT, to_date TEXT);
        CREATE TABLE salary (employee_id INTEGER, amount INTEGER, from_date TEXT, to_date TEXT);
        CREATE TABLE title (employee_id INTEGER, title TEXT, from_date TEXT, to_date TEXT);

        -- Sample seed row for execution checks
        INSERT INTO employee VALUES (10001, '1953-09-02', 'Georgi', 'Facello', 'M', '1986-06-26');
        INSERT INTO salary VALUES (10001, 85000, '1986-06-26', '9999-01-01');
        INSERT INTO department VALUES ('d005', 'Development');
        INSERT INTO department_employee VALUES (10001, 'd005', '1986-06-26', '9999-01-01');
    """)
    conn.commit()
    return conn

# Helper: Standard Text Normalization for Exact Match
def normalize_sql(sql_str):
    sql_str = sql_str.strip().rstrip(";")
    formatted = sqlparse.format(sql_str, reindent=False, keyword_case="lower")
    return re.sub(r"\s+", " ", formatted)

# Helper: Execution Accuracy Checker
def execute_query(conn, query):
    try:
        # Strip schema prefixes like 'employees.' to execute in local mock tables
        clean_q = re.sub(r"employees\.", "", query, flags=re.IGNORECASE)
        cursor = conn.cursor()
        cursor.execute(clean_q)
        res = cursor.fetchall()
        return True, set(res)
    except Exception as e:
        return False, None

# Helper: Generate SQL
def generate_sql(active_model, schema, question):
    system_prompt = (
        "You are an expert PostgreSQL database assistant. "
        "Based on the provided PostgreSQL schema definitions, output only the valid SQL query that answers the user's question."
    )
    user_prompt = f"### PostgreSQL Schema:\n{schema}\n\n### Request:\n{question}\n\n### SQL Query:"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = active_model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.01,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
db_conn = create_mock_db()
records = []

print("Running benchmark evaluation on test split...")

for idx, sample in enumerate(test_dataset):
    q = sample["question"]
    schema = sample["schema"]
    gold_sql = sample["query"]

    # Base inference
    with model.disable_adapter():
        base_pred = generate_sql(model, schema, q)

    # Fine-tuned inference
    ft_pred = generate_sql(model, schema, q)

    # 1. Exact Match Check
    norm_gold = normalize_sql(gold_sql)
    base_em = int(normalize_sql(base_pred) == norm_gold)
    ft_em = int(normalize_sql(ft_pred) == norm_gold)

    # 2. Execution & Validity Check
    gold_valid, gold_res = execute_query(db_conn, gold_sql)
    base_valid, base_res = execute_query(db_conn, base_pred)
    ft_valid, ft_res = execute_query(db_conn, ft_pred)

    base_ex = int(gold_valid and base_valid and (base_res == gold_res))
    ft_ex = int(gold_valid and ft_valid and (ft_res == gold_res))

    records.append({
        "sample_id": idx,
        "question": q,
        "gold_sql": gold_sql,
        "base_pred": base_pred,
        "ft_pred": ft_pred,
        "base_valid": int(base_valid),
        "ft_valid": int(ft_valid),
        "base_em": base_em,
        "ft_em": ft_em,
        "base_ex": base_ex,
        "ft_ex": ft_ex,
    })

# Save granular results to CSV
df = pd.DataFrame(records)
df.to_csv("sql_benchmark_results.csv", index=False)
print("Saved granular outputs to 'sql_benchmark_results.csv'.")

total = len(df)
metrics_table = [
    [
        "Valid SQL Rate (VSR)",
        f"{df['base_valid'].sum()} / {total} ({df['base_valid'].mean() * 100:.1f}%)",
        f"{df['ft_valid'].sum()} / {total} ({df['ft_valid'].mean() * 100:.1f}%)"
    ],
    [
        "Exact Match (EM)",
        f"{df['base_em'].sum()} / {total} ({df['base_em'].mean() * 100:.1f}%)",
        f"{df['ft_em'].sum()} / {total} ({df['ft_em'].mean() * 100:.1f}%)"
    ],
    [
        "Execution Accuracy (EX)",
        f"{df['base_ex'].sum()} / {total} ({df['base_ex'].mean() * 100:.1f}%)",
        f"{df['ft_ex'].sum()} / {total} ({df['ft_ex'].mean() * 100:.1f}%)"
    ]
]

print("\n" + tabulate(metrics_table, headers=["Metric", "Base Model", "Fine-Tuned (QLoRA)"], tablefmt="fancy_grid"))
