from pathlib import Path
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from dataset_pipeline import load_converted_dataset
from ml_runtime import MODEL_ID, sql_messages, load_tokenizer, quantization_config, load_quantized_model

model_id = MODEL_ID

# 1. Tokenizer
tokenizer = load_tokenizer(model_id)

# 2. Dataset Preparation
root = Path(__file__).resolve().parent
dataset = load_converted_dataset(root, ("train", "validation"))
train_dataset = dataset["train"]
val_dataset = dataset["validation"]

def format_sql_data(example):
    messages = sql_messages(example["schema"], example["question"], example["query"])
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

train_formatted = train_dataset.map(format_sql_data)
val_formatted = val_dataset.map(format_sql_data)

# 3. 4-bit Quantization Config (Ampere natively thrives on bfloat16)
bnb_config = quantization_config()

model = load_quantized_model(model_id, bnb_config)
model = prepare_model_for_kbit_training(model)

# 4. LoRA Setup
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, peft_config)

# 5. Training Config (Tuned for 6GB RTX 3060)
training_args = SFTConfig(
    output_dir="./pg_sql_rtx3060",
    dataset_text_field="text",
    max_length=512,                  # 512 is plenty for this schema/query length and saves VRAM
    num_train_epochs=4,
    per_device_train_batch_size=1,   # Keep batch size 1 to prevent OOM
    gradient_accumulation_steps=8,  # Effective batch size = 8
    learning_rate=2e-4,
    bf16=True,                       # Hardware supported on RTX 30-series
    fp16=False,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    optim="paged_adamw_8bit"         # Keeps optimizer states lean
)

# 6. Trainer Execution
trainer = SFTTrainer(
    model=model,
    train_dataset=train_formatted,
    eval_dataset=val_formatted,
    args=training_args,
    processing_class=tokenizer,
)

trainer.train()

# 7. Save Adapter Locally
trainer.model.save_pretrained("./pg_sql_rtx3060_adapter")
tokenizer.save_pretrained("./pg_sql_rtx3060_adapter")
print("Training complete and adapter saved!")