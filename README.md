# PostgreSQL Text-to-SQL

A reproducible dataset pipeline and QLoRA training experiment for translating natural-language requests into PostgreSQL queries over an employees schema.

[Get started](#get-started) | [Train a model](#train-a-model) | [Project layout](#project-layout) | [Validation](#validation)

## Overview

- Generate 400 synthetic examples across 50 SQL and assistant-behavior families.
- Keep every family's parameter variants in one split to reduce leakage: 320 training, 40 validation, and 40 test examples.
- Validate response contracts, split isolation, and PostgreSQL syntax.
- Convert the 320 query examples to `question`, `query`, and `schema` records; export Parquet for model training.
- Fine-tune `Qwen/Qwen2.5-1.5B-Instruct` with 4-bit quantization and LoRA.

The full dataset includes clarification and unsupported-request responses. The SQL-only training pipeline excludes these behaviors, so the adapter does not learn the full response policy.

## Get started

Use Python 3.11 or 3.12. Docker is optional for database execution checks. Run commands from the project root.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python dataset_pipeline.py
```

On macOS/Linux, activate with `source .venv/bin/activate`.

### Optional PostgreSQL checks

Copy `.env.example` to `.env`, then set a local password and matching `DATABASE_URL`.

```powershell
Copy-Item .env.example .env
docker compose up -d --wait
python dataset_pipeline.py --execute
```

Compose creates an empty employees schema on first initialization. A local employee-data dump is intentionally excluded from Git; supply and import your own attributed data if you need realistic result checks. Initialization scripts run only when the database volume is new. Use a restricted read-only database role for validation against populated databases.

> [!NOTE]
> The validator executes canonical queries in read-only transactions with a 10-second timeout. Successful execution does not prove that query results answer the request correctly.

## Train a model

Install a CUDA-compatible PyTorch build for your platform, then install the ML dependencies. The script is configured for an RTX 3060 with 6 GB VRAM, but memory use and runtime depend on your environment. Training requires CUDA, 4-bit bitsandbytes support, and bfloat16 support; CPU-only training is not configured.

```powershell
python -m pip install -r requirements-ml.txt
python dataset_pipeline.py
python QLora.py
```

Training reads local converted splits, evaluates each epoch, and saves checkpoints to `pg_sql_rtx3060/` and the final adapter to `pg_sql_rtx3060_adapter/`. These artifacts are ignored by Git. The dependency ranges are not a tested lockfile; a full GPU run is required to confirm a particular ML environment.

## Dataset formats

`data/*.jsonl` contains `prompt` and `completion` message arrays. Assistant completions contain a JSON object:

```json
{"action": "query", "sql": "SELECT id FROM employees.employee WHERE id = 10001;", "message": null}
```

For `clarify` and `unsupported`, `sql` is null and `message` explains what is needed. `metadata/*.jsonl` records family, action, and generation parameters. Converted query records contain only `question`, `query`, and `schema`.

## Project layout

| Path | Purpose |
| --- | --- |
| `dataset_pipeline.py` | Generate, validate, convert, export Parquet, and upload to Hugging Face |
| `QLora.py` | Fine-tune the model using local converted training and validation splits |
| `ml_runtime.py` | Internal helper module for SQL prompts, tokenizer, and quantized model setup |
| `data/` / `metadata/` | Versioned synthetic examples and family metadata |
| `schema_context.txt` / `sql_data/employees_schema.sql` | Prompt schema and PostgreSQL DDL |
| `tests/` | Dataset pipeline and model-configuration regression tests |

## Validation Result 
![Results](<Screenshot 2026-10-02 060908.png>)
The result shows the improvement from the fine-tuning, the dataset have to be increased to work better.

Schema attribution and distribution terms are in [LICENSE.md](LICENSE.md).
