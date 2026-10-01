from datasets import load_dataset

dataset = load_dataset(
    "parquet",
    data_files={
        split: (
            "hf://datasets/jin-0626/postgresql-employees-assistant/"
            f"data/{split}.parquet"
        )
        for split in ("train", "validation", "test")
    },
    token=True,
)

for split, rows in dataset.items():
    assert set(rows.column_names) == {"question", "query", "schema"}
    assert all(
        isinstance(row[column], str) and row[column].strip()
        for row in rows
        for column in ("question", "query", "schema")
    )
    print(f"{split}: {len(rows)} rows")

print(dataset["train"][0])