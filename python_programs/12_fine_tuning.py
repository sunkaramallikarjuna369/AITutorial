"""
=============================================================================
MODULE 12: FINE-TUNING & TRANSFER LEARNING - COMPREHENSIVE 360 DEGREE
=============================================================================

This module provides COMPLETE coverage of Fine-Tuning including:
- 4W+H Explanations (What, Why, When, Where, How)
- Transfer Learning Concepts
- Fine-Tuning Techniques (Full, LoRA, QLoRA)
- Practical Examples
- GenAI Model Integrations
- Interview Questions
- Common Pitfalls

SETUP:
------
pip install torch transformers peft datasets

For GenAI features:
pip install openai anthropic

=============================================================================
"""

import os
from typing import List, Dict, Optional

# =============================================================================
# GENAI INTEGRATION
# =============================================================================

class FineTuningAssistant:
    """Use GenAI models for fine-tuning explanations"""
    
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
    
    def explain_with_openai(self, concept: str) -> str:
        try:
            from openai import OpenAI
            if not self.openai_key:
                return "[Set OPENAI_API_KEY]"
            client = OpenAI(api_key=self.openai_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"Explain {concept} in fine-tuning with examples."}],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[Error: {e}]"
    
    def explain_with_ollama(self, concept: str, model: str = "llama2") -> str:
        try:
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": f"Explain {concept} in fine-tuning.", "stream": False},
                timeout=60
            )
            return response.json().get("response", "") if response.status_code == 200 else "[Ollama error]"
        except Exception as e:
            return f"[Start Ollama: ollama serve. Error: {e}]"


# =============================================================================
# SECTION 1: COMPREHENSIVE FINE-TUNING EXPLANATION (4W+H)
# =============================================================================

def explain_fine_tuning_comprehensive():
    """Comprehensive 360-degree explanation of Fine-Tuning"""
    print("=" * 70)
    print("FINE-TUNING & TRANSFER LEARNING - COMPREHENSIVE COVERAGE")
    print("=" * 70)
    
    print("""
WHAT IS FINE-TUNING?
====================

Fine-tuning is adapting a pre-trained model to a specific task or domain
by continuing training on task-specific data.

TRANSFER LEARNING CONCEPT:
    Pre-trained Model (general knowledge)
           |
           v
    Fine-tuning (task-specific data)
           |
           v
    Specialized Model (your task)

ANALOGY:
    Pre-training = Learning to read and write (general education)
    Fine-tuning = Learning medical terminology (specialized training)

TYPES OF FINE-TUNING:
    1. Full Fine-tuning: Update all parameters
    2. Feature Extraction: Freeze base, train new head
    3. Partial Fine-tuning: Freeze some layers, train others
    4. Parameter-Efficient (PEFT): LoRA, Adapters, Prefix-tuning

WHY FINE-TUNE?
==============

1. BETTER PERFORMANCE: Adapt to your specific task/domain
2. LESS DATA: Pre-trained knowledge reduces data requirements
3. FASTER TRAINING: Start from good initialization
4. COST-EFFECTIVE: Cheaper than training from scratch
5. CUSTOMIZATION: Add your style, format, knowledge

WHEN TO FINE-TUNE?
==================

FINE-TUNE WHEN:
    - Pre-trained model doesn't perform well on your task
    - You have task-specific data (100s to 1000s of examples)
    - You need consistent output format
    - Domain-specific vocabulary/knowledge needed

DON'T FINE-TUNE WHEN:
    - Prompt engineering works well enough
    - You have very little data (<100 examples)
    - Task is too different from pre-training
    - RAG can provide needed knowledge

WHERE IS FINE-TUNING USED?
==========================

APPLICATIONS:
    - Customer support chatbots (company-specific)
    - Medical diagnosis (domain expertise)
    - Legal document analysis (specialized vocabulary)
    - Code generation (company coding style)
    - Content moderation (specific policies)

PLATFORMS:
    - OpenAI Fine-tuning API
    - HuggingFace Transformers
    - Google Vertex AI
    - AWS SageMaker

HOW TO FINE-TUNE?
=================

FULL FINE-TUNING:
    1. Load pre-trained model
    2. Prepare dataset (input-output pairs)
    3. Train on your data (all parameters updated)
    4. Evaluate and iterate

PARAMETER-EFFICIENT (LoRA):
    1. Load pre-trained model (frozen)
    2. Add small trainable adapters
    3. Train only adapters (0.1-1% of parameters)
    4. Merge adapters with base model

DATA FORMAT (Chat models):
    {"messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language..."}
    ]}
    """)


# =============================================================================
# SECTION 2: FINE-TUNING TECHNIQUES
# =============================================================================

def fine_tuning_techniques():
    """Detailed explanation of fine-tuning techniques"""
    print("\n" + "=" * 70)
    print("FINE-TUNING TECHNIQUES")
    print("=" * 70)
    
    print("""
1. FULL FINE-TUNING
   =================
   Update ALL model parameters
   
   Pros: Best performance potential
   Cons: Expensive, needs lots of data, catastrophic forgetting risk
   
   Use when: Large dataset, significant compute budget

2. FEATURE EXTRACTION (Freeze Base)
   =================================
   Freeze pre-trained layers, only train new classification head
   
   Pros: Fast, prevents overfitting
   Cons: Limited adaptation
   
   Use when: Small dataset, similar to pre-training task

3. GRADUAL UNFREEZING
   ===================
   Start frozen, gradually unfreeze layers from top to bottom
   
   Epoch 1: Train only head
   Epoch 2: Unfreeze last layer
   Epoch 3: Unfreeze more layers
   ...
   
   Pros: Stable training, good for small datasets

4. LoRA (Low-Rank Adaptation)
   ==========================
   Add small trainable matrices to attention layers
   
   Original: W (frozen)
   LoRA: W + BA where B, A are small matrices
   
   Parameters: 0.1-1% of original
   Performance: ~95-99% of full fine-tuning
   
   Key hyperparameters:
   - r (rank): 4, 8, 16, 32 (higher = more capacity)
   - alpha: scaling factor (typically 16-32)
   - target_modules: which layers to adapt

5. QLoRA (Quantized LoRA)
   =======================
   LoRA + 4-bit quantization
   
   Benefits:
   - Fine-tune 65B model on single GPU
   - 4-bit base model + 16-bit LoRA adapters
   - Minimal quality loss

6. ADAPTERS
   =========
   Add small bottleneck layers between transformer layers
   
   Structure: Down-project -> Activation -> Up-project
   
   Similar to LoRA but different architecture

7. PREFIX TUNING
   ==============
   Add trainable "virtual tokens" to input
   
   Input: [PREFIX_1, PREFIX_2, ..., actual_tokens]
   
   Only PREFIX tokens are trainable

8. PROMPT TUNING
   ==============
   Learn soft prompts (continuous embeddings)
   
   Similar to prefix tuning but simpler
    """)


# =============================================================================
# SECTION 3: PRACTICAL EXAMPLES
# =============================================================================

def practical_examples():
    """Practical fine-tuning examples"""
    print("\n" + "=" * 70)
    print("PRACTICAL FINE-TUNING EXAMPLES")
    print("=" * 70)
    
    print("""
EXAMPLE 1: OpenAI Fine-Tuning API
=================================

# Prepare data (JSONL format)
{"messages": [{"role": "system", "content": "You are a customer support agent."},
              {"role": "user", "content": "How do I reset my password?"},
              {"role": "assistant", "content": "To reset your password: 1. Go to..."}]}

# Upload and fine-tune
from openai import OpenAI
client = OpenAI()

# Upload file
file = client.files.create(file=open("training.jsonl", "rb"), purpose="fine-tune")

# Create fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-3.5-turbo"
)

# Use fine-tuned model
response = client.chat.completions.create(
    model="ft:gpt-3.5-turbo:my-org::abc123",
    messages=[{"role": "user", "content": "How do I reset my password?"}]
)


EXAMPLE 2: HuggingFace + LoRA
=============================

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure LoRA
lora_config = LoraConfig(
    r=16,                    # Rank
    lora_alpha=32,           # Scaling
    target_modules=["q_proj", "v_proj"],  # Which layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
print(f"Trainable params: {model.print_trainable_parameters()}")

# Train
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()


EXAMPLE 3: QLoRA (4-bit)
========================

from transformers import BitsAndBytesConfig
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Apply LoRA on top
model = get_peft_model(model, lora_config)
# Now you can fine-tune 70B model on single GPU!
    """)


# =============================================================================
# SECTION 4: INTERVIEW QUESTIONS
# =============================================================================

def interview_questions():
    """Common fine-tuning interview questions"""
    print("\n" + "=" * 70)
    print("FINE-TUNING INTERVIEW QUESTIONS")
    print("=" * 70)
    
    questions = [
        ("What is catastrophic forgetting?",
         "When fine-tuning causes model to forget pre-trained knowledge. Mitigate with: lower learning rate, regularization, gradual unfreezing, or PEFT methods."),
        ("When would you use LoRA vs full fine-tuning?",
         "LoRA: Limited compute, want to preserve base model, multiple tasks (swap adapters). Full: Maximum performance needed, large dataset, sufficient compute."),
        ("How much data do you need for fine-tuning?",
         "Depends on task complexity. Minimum: 100-500 examples. Better: 1000-10000. More data generally helps but with diminishing returns."),
        ("What learning rate should you use?",
         "Much lower than pre-training. Typical: 1e-5 to 5e-5 for full fine-tuning, 1e-4 to 3e-4 for LoRA. Start low, increase if underfitting."),
        ("How do you evaluate fine-tuned models?",
         "Hold-out test set, task-specific metrics (accuracy, F1, BLEU), human evaluation for generation tasks, compare to base model."),
    ]
    
    for i, (q, a) in enumerate(questions, 1):
        print(f"\nQ{i}: {q}")
        print(f"A: {a}")


# =============================================================================
# SECTION 5: COMMON PITFALLS
# =============================================================================

def common_pitfalls():
    """Common fine-tuning mistakes"""
    print("\n" + "=" * 70)
    print("COMMON FINE-TUNING PITFALLS")
    print("=" * 70)
    
    pitfalls = [
        ("Learning rate too high", "Use 10-100x lower than pre-training; start with 1e-5"),
        ("Not enough data", "Need 100+ examples minimum; augment if needed"),
        ("Overfitting", "Use validation set, early stopping, dropout, weight decay"),
        ("Wrong data format", "Match the format model was pre-trained on"),
        ("Ignoring base model capabilities", "Try prompt engineering first; fine-tune only if needed"),
        ("Not evaluating properly", "Use held-out test set; compare to base model"),
    ]
    
    for mistake, solution in pitfalls:
        print(f"\nMistake: {mistake}")
        print(f"Solution: {solution}")


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main():
    print("=" * 70)
    print("   MODULE 12: FINE-TUNING - COMPREHENSIVE 360 DEGREE COVERAGE")
    print("=" * 70)
    
    assistant = FineTuningAssistant()
    
    while True:
        print("\n" + "-" * 70)
        print("Choose a topic:")
        print()
        print("FUNDAMENTALS:")
        print("  1. What is Fine-Tuning? (4W+H)")
        print("  2. Fine-Tuning Techniques")
        print("  3. Practical Examples")
        print()
        print("DEEP DIVE:")
        print("  4. Interview Questions")
        print("  5. Common Pitfalls")
        print()
        print("GENAI FEATURES:")
        print("  6. AI Explanation (OpenAI)")
        print("  7. AI Explanation (Ollama - FREE)")
        print()
        print("  8. Run ALL Topics")
        print("  0. Exit")
        print("-" * 70)
        
        choice = input("\nEnter choice (0-8): ").strip()
        
        if choice == "0":
            print("\nHappy learning!")
            break
        elif choice == "1":
            explain_fine_tuning_comprehensive()
        elif choice == "2":
            fine_tuning_techniques()
        elif choice == "3":
            practical_examples()
        elif choice == "4":
            interview_questions()
        elif choice == "5":
            common_pitfalls()
        elif choice == "6":
            concept = input("Enter concept to explain: ").strip()
            print(assistant.explain_with_openai(concept))
        elif choice == "7":
            concept = input("Enter concept to explain: ").strip()
            print(assistant.explain_with_ollama(concept))
        elif choice == "8":
            explain_fine_tuning_comprehensive()
            fine_tuning_techniques()
            practical_examples()
            interview_questions()
            common_pitfalls()
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
