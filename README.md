# PostgreSQL Text-to-SQL

A reproducible dataset pipeline and QLoRA training experiment for translating natural-language requests into PostgreSQL queries over an employees schema.

[Get started](#get-started) Â· [Train a model](#train-a-model) Â· [Project layout](#project-layout) Â· [Validation](#validation)

## Overview

- Generate 400 synthetic examples across 50 SQL and assistant-behavior families.
- Keep every family's parameter variants in one split to reduce leakage: 320 training, 40 validation, and 40 test examples.
- Validate response contracts, split isolation, and PostgreSQL syntax.
- Convert the 320 query examples to `question`, `query`, and `schema` records; export Parquet for model training.
- Fine-tune `Qwen/Qwen2.5-1.5B-Instruct` with 4-bit quantization and LoRA, then compare base and adapted predictions.

The full dataset includes clarification and unsupported-request responses. The SQL-only training pipeline excludes these behaviors, so the adapter does not learn the full response policy.

## Get started

Use Python 3.11 or 3.12. Docker is optional for database execution checks. Run commands from the project root.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python build_dataset.py
python validate_dataset.py
python convert_dataset.py
python export_parquet.py
```

On macOS/Linux, activate with `source .venv/bin/activate`.

### Optional PostgreSQL checks

Copy `.env.example` to `.env`, then set a local password and matching `DATABASE_URL`.

```powershell
Copy-Item .env.example .env
docker compose up -d --wait
python validate_dataset.py --execute
```

Compose creates an empty employees schema on first initialization. A local employee-data dump is intentionally excluded from Git; supply and import your own attributed data if you need realistic result checks. Initialization scripts run only when the database volume is new. Use a restricted read-only database role for validation against populated databases.

> [!NOTE]
> The validator executes canonical queries in read-only transactions with a 10-second timeout. Successful execution does not prove that query results answer the request correctly.

## Train a model

Install a CUDA-compatible PyTorch build for your platform, then install the ML dependencies. The script is configured for an RTX 3060 with 6 GB VRAM, but memory use and runtime depend on your environment. Training requires CUDA, 4-bit bitsandbytes support, and bfloat16 support; CPU-only training is not configured.

```powershell
python -m pip install -r requirements-ml.txt
python convert_dataset.py
python QLora.py
python evaluation_comparison.py
```

Training reads local converted splits, evaluates each epoch, and saves checkpoints to `pg_sql_rtx3060/` and the final adapter to `pg_sql_rtx3060_adapter/`. Evaluation reads the local test split and writes `sql_benchmark_results.csv`. These artifacts are ignored by Git. The dependency ranges are not a tested lockfile; a full GPU run is required to confirm a particular ML environment.

> [!IMPORTANT]
> Evaluation uses a tiny SQLite mock database. PostgreSQL-specific expressions may fail, and comparing sets of rows ignores ordering and duplicates. Its execution score is a smoke check, not a PostgreSQL semantic benchmark. Review predictions and validate against PostgreSQL before relying on results.

## Dataset formats

`data/*.jsonl` contains `prompt` and `completion` message arrays. Assistant completions contain a JSON object:

```json
{"action": "query", "sql": "SELECT id FROM employees.employee WHERE id = 10001;", "message": null}
```

For `clarify` and `unsupported`, `sql` is null and `message` explains what is needed. `metadata/*.jsonl` records family, action, and generation parameters. Converted query records contain only `question`, `query`, and `schema`.

## Project layout

| Path | Purpose |
| --- | --- |
| `build_dataset.py` | Deterministic generation with seed 42 |
| `validate_dataset.py` | Contract, family isolation, SQL parser, optional execution checks |
| `convert_dataset.py` / `export_parquet.py` | SQL-only JSONL and Parquet exports |
| `data/` / `metadata/` | Versioned synthetic examples and family metadata |
| `schema_context.txt` / `sql_data/employees_schema.sql` | Prompt schema and PostgreSQL DDL |
| `QLora.py` / `evaluation_comparison.py` | GPU training and comparison experiment |
| `upload_huggingface.py` | Upload the original JSONL package to an explicit dataset repository |
| `check_parquet.py` | Check Parquet files in the author's Hugging Face repository |
| `clean_huggingface` | Legacy remote-replacement utility; removes other repository files |

## Validation

```powershell
python validate_dataset.py
```

GitHub Actions regenerates the dataset, validates it, exports Parquet, and checks that generation leaves versioned dataset files unchanged. GPU training and live PostgreSQL execution are outside CI.

The generated report records which checks ran; semantic result correctness remains unverified. Parameter variants are synthetic and do not establish real-world model quality.

## Hugging Face publishing

Authenticate through the Hugging Face CLI, then select your own destination:

```powershell
hf auth login
python upload_huggingface.py YOUR_USERNAME/postgresql-employees-assistant
```

The uploader defaults to a private repository and uploads original message-format JSONL, not SQL-only Parquet. Add `--public` only when you intend public publication. The legacy `clean_huggingface` script requires an explicit repository ID and `--confirm-replace`; it deletes unrelated remote files. Do not run it as part of setup.

## Push to GitHub

Model weights, virtual environments, credentials, generated exports, and the employee-data dump are excluded by `.gitignore`. Review the staged files before committing:

```powershell
git add .
git diff --cached --stat
git commit -m "Prepare PostgreSQL text-to-SQL project"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Schema attribution and distribution terms are in [LICENSE.md](LICENSE.md).
