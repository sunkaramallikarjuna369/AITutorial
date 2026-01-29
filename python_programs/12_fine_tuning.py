"""
Module 12: Fine-tuning & Transfer Learning
==========================================
Customize AI models for your specific needs!
"""

def explain_fine_tuning():
    print("=" * 60)
    print("Fine-tuning & Transfer Learning")
    print("=" * 60)
    print("""
Fine-tuning = Taking a pre-trained model and training it on your data!

Why Fine-tune?
- Pre-trained models already know language/images
- You just teach them YOUR specific task
- Much faster and cheaper than training from scratch
- Works with small datasets

Types:
1. Full Fine-tuning: Update all parameters
2. LoRA: Only update small adapter layers
3. QLoRA: LoRA + Quantization (even smaller!)
4. Prompt Tuning: Only learn prompt embeddings

When to use what:
- Lots of data + compute → Full fine-tuning
- Limited resources → LoRA/QLoRA
- Very limited data → Few-shot or prompt tuning
    """)

code_examples = '''
"""
Fine-tuning Examples
====================
"""

# ============================================
# 1. Fine-tune with Hugging Face
# ============================================

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

def fine_tune_classifier():
    # Load pre-trained model
    model_name = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load dataset
    dataset = load_dataset("imdb")
    
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)
    
    tokenized = dataset.map(tokenize, batched=True)
    
    # Training arguments
    args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch"
    )
    
    # Train
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"]
    )
    
    trainer.train()
    return model


# ============================================
# 2. LoRA Fine-tuning (Parameter Efficient)
# ============================================

from peft import LoraConfig, get_peft_model, TaskType

def fine_tune_with_lora():
    model_name = "meta-llama/Llama-2-7b-hf"
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,                    # Rank
        lora_alpha=32,           # Scaling
        target_modules=["q_proj", "v_proj"],  # Which layers
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Check trainable parameters
    model.print_trainable_parameters()
    # Output: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%
    
    return model


# ============================================
# 3. QLoRA (Quantized LoRA)
# ============================================

from transformers import BitsAndBytesConfig

def fine_tune_with_qlora():
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load quantized model
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Apply LoRA on top
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    return model


# ============================================
# 4. Fine-tune OpenAI Model
# ============================================

from openai import OpenAI

client = OpenAI()

def fine_tune_gpt():
    # Prepare training data (JSONL format)
    training_data = [
        {"messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "I'd be happy to help with weather info!"}
        ]},
        # ... more examples
    ]
    
    # Upload file
    file = client.files.create(
        file=open("training_data.jsonl", "rb"),
        purpose="fine-tune"
    )
    
    # Create fine-tuning job
    job = client.fine_tuning.jobs.create(
        training_file=file.id,
        model="gpt-3.5-turbo"
    )
    
    return job


# ============================================
# 5. Transfer Learning for Images
# ============================================

import torch
import torch.nn as nn
from torchvision import models

def transfer_learning_cnn(num_classes):
    # Load pre-trained ResNet
    model = models.resnet50(pretrained=True)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model


# ============================================
# 6. SFTTrainer for Instruction Tuning
# ============================================

from trl import SFTTrainer

def instruction_tune():
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        args=TrainingArguments(
            output_dir="./sft_output",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=True
        )
    )
    
    trainer.train()


print("Fine-tuning examples loaded!")
print("Install: pip install transformers peft trl bitsandbytes")
'''

def main():
    explain_fine_tuning()
    print("\nPython Code Examples:")
    print(code_examples)

if __name__ == "__main__":
    main()
