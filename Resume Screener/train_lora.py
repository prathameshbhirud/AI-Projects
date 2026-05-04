from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import torch

# 🔹 Choose small model (important for laptop)
MODEL_NAME = "microsoft/phi-2"   # or mistralai/Mistral-7B-v0.1 (if GPU strong)

# Load dataset
dataset = load_dataset("json", data_files="train.json")["train"]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Format dataset
def format_example(example):
    return {
        "text": f"""### Instruction:
{example['prompt']}

### Response:
{example['response']}"""
    }

dataset = dataset.map(format_example)

# Load model (4-bit for low memory)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Training args
training_args = TrainingArguments(
    output_dir="./resume_lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit"
)

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_args,
)

trainer.train()

# Save model
trainer.model.save_pretrained("./resume_lora")
tokenizer.save_pretrained("./resume_lora")