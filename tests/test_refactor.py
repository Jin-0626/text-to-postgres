"""Regression checks for prompting, model setup, and dataset/Parquet boundaries."""
import contextlib
import io
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import dataset_pipeline as dataset_io
import ml_runtime

BASELINES = {'format_sql_data': 'def format_sql_data(example):\n    system_prompt = (\n        "You are an expert PostgreSQL database assistant. "\n        "Based on the provided PostgreSQL schema definitions, output only the valid SQL query that answers the user\'s question."\n    )\n    user_prompt = (\n        f"### PostgreSQL Schema:\\n{example[\'schema\']}\\n\\n"\n        f"### Request:\\n{example[\'question\']}\\n\\n"\n        f"### SQL Query:"\n    )\n    messages = [\n        {"role": "system", "content": system_prompt},\n        {"role": "user", "content": user_prompt},\n        {"role": "assistant", "content": example[\'query\']}\n    ]\n    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}', 'generate_sql': 'def generate_sql(active_model, schema, question):\n    system_prompt = (\n        "You are an expert PostgreSQL database assistant. "\n        "Based on the provided PostgreSQL schema definitions, output only the valid SQL query that answers the user\'s question."\n    )\n    user_prompt = f"### PostgreSQL Schema:\\n{schema}\\n\\n### Request:\\n{question}\\n\\n### SQL Query:"\n    messages = [\n        {"role": "system", "content": system_prompt},\n        {"role": "user", "content": user_prompt}\n    ]\n    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")\n\n    with torch.no_grad():\n        outputs = active_model.generate(\n            **inputs,\n            max_new_tokens=128,\n            temperature=0.01,\n            pad_token_id=tokenizer.eos_token_id\n        )\n    return tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()'}


class RefactorTests(unittest.TestCase):
    def test_training_prompt_matches_original(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: (messages, kwargs)
        namespace = {"tokenizer": tokenizer}
        exec(BASELINES["format_sql_data"], namespace)
        example = {"schema": "employees.employee(id bigint)", "question": "Look up 1", "query": "SELECT 1;"}
        expected = namespace["format_sql_data"](example)
        actual = {"text": tokenizer.apply_chat_template(ml_runtime.sql_messages(example["schema"], example["question"], example["query"]), tokenize=False)}
        self.assertEqual(actual, expected)

    def test_inference_prompt_matches_original(self):
        class Captured(Exception):
            pass
        tokenizer = Mock()
        tokenizer.apply_chat_template.side_effect = Captured
        namespace = {"tokenizer": tokenizer}
        exec(BASELINES["generate_sql"], namespace)
        with self.assertRaises(Captured):
            namespace["generate_sql"](None, "schema", "question")
        expected = tokenizer.apply_chat_template.call_args
        tokenizer.apply_chat_template.reset_mock()
        with self.assertRaises(Captured):
            tokenizer.apply_chat_template(ml_runtime.sql_messages("schema", "question"), tokenize=False, add_generation_prompt=True)
        self.assertEqual(tokenizer.apply_chat_template.call_args, expected)

    def test_empty_assistant_query_is_preserved(self):
        self.assertEqual(ml_runtime.sql_messages("", "", "")[-1], {"role": "assistant", "content": ""})
        self.assertEqual(len(ml_runtime.sql_messages("", "")), 2)

    def test_tokenizer_padding(self):
        for pad in (None, "existing"):
            tokenizer = types.SimpleNamespace(pad_token=pad, eos_token="eos")
            factory = Mock(return_value=tokenizer)
            with patch.dict(sys.modules, {"transformers": types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=factory))}):
                self.assertIs(ml_runtime.load_tokenizer("model"), tokenizer)
            self.assertEqual(tokenizer.pad_token, "eos" if pad is None else pad)
            factory.assert_called_once_with("model")

    def test_quantized_model_configuration(self):
        config_factory, model_factory = Mock(), Mock()
        transformers = types.SimpleNamespace(BitsAndBytesConfig=config_factory, AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=model_factory))
        with patch.dict(sys.modules, {"torch": types.SimpleNamespace(bfloat16="bf16"), "transformers": transformers}):
            config = ml_runtime.quantization_config()
            ml_runtime.load_quantized_model("model", config)
        config_factory.assert_called_once_with(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="bf16", bnb_4bit_use_double_quant=True)
        model_factory.assert_called_once_with("model", quantization_config=config, dtype="bf16", device_map="auto")

    def test_local_split_loading(self):
        loader = Mock()
        with patch.dict(sys.modules, {"datasets": types.SimpleNamespace(load_dataset=loader)}):
            dataset_io.load_converted_dataset(Path("root"), ("test",))
        loader.assert_called_once_with("json", data_files={"test": str(Path("root/converted/test.jsonl"))})

    def test_sql_column_contract(self):
        dataset_io.assert_sql_columns(types.SimpleNamespace(column_names=["schema", "query", "question"]))
        with self.assertRaises(AssertionError):
            dataset_io.assert_sql_columns(types.SimpleNamespace(column_names=["schema", "query"]))

    def test_export_paths_and_stdout(self):
        from tempfile import TemporaryDirectory
        rows = Mock(column_names=list(dataset_io.SQL_COLUMNS))
        rows.__len__ = Mock(return_value=2)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = io.StringIO()
            dataset = {"train": rows}
            with patch.object(dataset_io, "load_converted_dataset", return_value=dataset), contextlib.redirect_stdout(output):
                self.assertIs(dataset_io.export_parquet(root), dataset)
            rows.to_parquet.assert_called_once_with(str(root / "parquet/train.parquet"))
            self.assertEqual(output.getvalue(), "train: 2 rows exported\n")



if __name__ == "__main__":
    unittest.main()
