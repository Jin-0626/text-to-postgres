"""Shared SQL prompting and quantized model setup for training and inference."""
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM_PROMPT = (
    "You are an expert PostgreSQL database assistant. "
    "Based on the provided PostgreSQL schema definitions, output only the valid SQL query that answers the user's question."
)


def sql_messages(schema: str, question: str, query: str | None = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"### PostgreSQL Schema:\n{schema}\n\n### Request:\n{question}\n\n### SQL Query:"},
    ]
    if query is not None:
        messages.append({"role": "assistant", "content": query})
    return messages


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def quantization_config():
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_quantized_model(model_id: str, config):
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=config,
        dtype=torch.bfloat16,
        device_map="auto",
    )
