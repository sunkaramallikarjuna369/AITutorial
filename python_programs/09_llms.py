"""
Module 09: Large Language Models (LLMs)
=======================================
ChatGPT, Claude, Gemini, and how they work!
"""

def explain_llms():
    print("=" * 60)
    print("Large Language Models (LLMs)")
    print("=" * 60)
    print("""
LLMs are AI models trained on massive amounts of text!

What makes them "Large"?
- GPT-3: 175 billion parameters
- GPT-4: ~1.7 trillion parameters (estimated)
- Claude: Hundreds of billions of parameters

How they work:
1. Pre-training: Learn from internet text (predict next word)
2. Fine-tuning: Learn to follow instructions
3. RLHF: Learn from human feedback

Key Capabilities:
- Text generation
- Question answering
- Code writing
- Translation
- Summarization
- Reasoning

Popular LLMs:
- OpenAI: GPT-4, GPT-4o
- Anthropic: Claude 3.5 Sonnet
- Google: Gemini
- Meta: Llama 3
- Mistral: Mixtral
    """)

code_examples = '''
"""
LLM Usage Examples
==================
"""

# ============================================
# 1. OpenAI API
# ============================================

from openai import OpenAI

client = OpenAI(api_key="your-api-key")

def chat_with_gpt(message):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    return response.choices[0].message.content


# ============================================
# 2. Anthropic Claude API
# ============================================

import anthropic

client = anthropic.Anthropic(api_key="your-api-key")

def chat_with_claude(message):
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": message}
        ]
    )
    return response.content[0].text


# ============================================
# 3. Local LLM with Ollama
# ============================================

import ollama

def chat_with_ollama(message, model="llama3"):
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": message}]
    )
    return response["message"]["content"]


# ============================================
# 4. Hugging Face Transformers
# ============================================

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_local_llm(model_name="meta-llama/Llama-2-7b-chat-hf"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=0.7,
        do_sample=True,
        top_p=0.95
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ============================================
# 5. Streaming Responses
# ============================================

def stream_gpt_response(message):
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": message}],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)


# ============================================
# 6. Function Calling
# ============================================

def chat_with_functions(message):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": message}],
        tools=tools,
        tool_choice="auto"
    )
    
    return response


# ============================================
# 7. Embeddings
# ============================================

def get_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]


print("LLM examples loaded!")
print("Install: pip install openai anthropic ollama transformers")
'''

def main():
    explain_llms()
    print("\nPython Code Examples:")
    print(code_examples)

if __name__ == "__main__":
    main()
