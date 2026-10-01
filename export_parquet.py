from pathlib import Path
from datasets import load_dataset

root = Path(__file__).resolve().parent
output = root / "parquet"
output.mkdir(exist_ok=True)

dataset = load_dataset(
    "json",
    data_files={
        split: str(root / "converted" / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    },
)

for split, rows in dataset.items():
    assert set(rows.column_names) == {"question", "query", "schema"}
    rows.to_parquet(str(output / f"{split}.parquet"))
    print(f"{split}: {len(rows)} rows exported")