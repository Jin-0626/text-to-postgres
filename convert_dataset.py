import json
from pathlib import Path

root = Path(__file__).resolve().parent
output_dir = root / "converted"
output_dir.mkdir(exist_ok=True)

behaviors = []

for split in ("train", "validation", "test"):
    converted = []

    source = root / "data" / f"{split}.jsonl"
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue

        row = json.loads(line)
        user_text = next(
            message["content"]
            for message in row["prompt"]
            if message["role"] == "user"
        )

        schema, separator, question = user_text.partition("\n\nRequest:")
        if not separator or not schema.startswith("Schema:\n"):
            raise ValueError(f"{split}, line {line_number}: unexpected format")

        schema = schema.removeprefix("Schema:\n").strip()
        answer = json.loads(row["completion"][0]["content"])

        if answer["action"] != "query":
            behaviors.append({
                "split": split,
                "question": question.strip(),
                "schema": schema,
                "action": answer["action"],
                "message": answer["message"],
            })
            continue

        converted.append({
            "question": question.strip(),
            "query": answer["sql"],
            "schema": schema,
        })

    target = output_dir / f"{split}.jsonl"
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in converted),
        encoding="utf-8",
    )
    print(f"{split}: {len(converted)} SQL examples")

(root / "behavior_examples.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in behaviors),
    encoding="utf-8",
)
print(f"Separate behavioral examples: {len(behaviors)}")