"""
=============================================================================
MODULE 9: LARGE LANGUAGE MODELS (LLMs) - 360 DEGREE COVERAGE
=============================================================================

This comprehensive module covers:
- What are LLMs and how they work
- Popular LLMs: GPT-4, Claude, Gemini, Llama, Mistral
- Using multiple GenAI APIs (OpenAI, Anthropic, Google, Ollama)
- Streaming, function calling, embeddings
- Local LLM deployment with Hugging Face

SETUP INSTRUCTIONS:
-------------------
pip install openai anthropic google-generativeai ollama transformers torch requests

API Keys:
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"

For FREE local models:
1. Install Ollama: https://ollama.ai
2. Run: ollama pull llama2 mistral codellama

=============================================================================
"""

import os
from typing import Optional, Generator

# =============================================================================
# LLM INTERFACE CLASS - Unified access to multiple LLMs
# =============================================================================

class LLMInterface:
    """Unified interface to work with multiple LLM providers"""
    
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_key = os.getenv("GOOGLE_API_KEY")
    
    # -------------------------------------------------------------------------
    # OpenAI GPT Models
    # -------------------------------------------------------------------------
    def chat_openai(self, message: str, model: str = "gpt-3.5-turbo", 
                    system: str = "You are a helpful assistant.") -> str:
        """Chat with OpenAI GPT models"""
        try:
            from openai import OpenAI
            if not self.openai_key:
                return "Set OPENAI_API_KEY first"
            
            client = OpenAI(api_key=self.openai_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": message}
                ],
                max_tokens=1000
            )
            return response.choices[0].message.content
        except ImportError:
            return "pip install openai"
        except Exception as e:
            return f"Error: {e}"
    
    def stream_openai(self, message: str, model: str = "gpt-3.5-turbo") -> Generator:
        """Stream responses from OpenAI"""
        try:
            from openai import OpenAI
            if not self.openai_key:
                yield "Set OPENAI_API_KEY first"
                return
            
            client = OpenAI(api_key=self.openai_key)
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": message}],
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"Error: {e}"
    
    # -------------------------------------------------------------------------
    # Anthropic Claude Models
    # -------------------------------------------------------------------------
    def chat_claude(self, message: str, model: str = "claude-3-sonnet-20240229") -> str:
        """Chat with Anthropic Claude"""
        try:
            import anthropic
            if not self.anthropic_key:
                return "Set ANTHROPIC_API_KEY first"
            
            client = anthropic.Anthropic(api_key=self.anthropic_key)
            response = client.messages.create(
                model=model,
                max_tokens=1000,
                messages=[{"role": "user", "content": message}]
            )
            return response.content[0].text
        except ImportError:
            return "pip install anthropic"
        except Exception as e:
            return f"Error: {e}"
    
    # -------------------------------------------------------------------------
    # Google Gemini Models
    # -------------------------------------------------------------------------
    def chat_gemini(self, message: str, model: str = "gemini-pro") -> str:
        """Chat with Google Gemini"""
        try:
            import google.generativeai as genai
            if not self.google_key:
                return "Set GOOGLE_API_KEY first"
            
            genai.configure(api_key=self.google_key)
            model_instance = genai.GenerativeModel(model)
            response = model_instance.generate_content(message)
            return response.text
        except ImportError:
            return "pip install google-generativeai"
        except Exception as e:
            return f"Error: {e}"
    
    # -------------------------------------------------------------------------
    # Ollama (FREE Local Models)
    # -------------------------------------------------------------------------
    def chat_ollama(self, message: str, model: str = "llama2") -> str:
        """Chat with local Ollama models (FREE!)"""
        try:
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": message, "stream": False},
                timeout=120
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            return f"Ollama error: {response.status_code}"
        except Exception as e:
            return f"Start Ollama: ollama serve. Error: {e}"
    
    # -------------------------------------------------------------------------
    # Hugging Face (Local Models)
    # -------------------------------------------------------------------------
    def chat_huggingface(self, message: str, model: str = "gpt2") -> str:
        """Use Hugging Face models locally"""
        try:
            from transformers import pipeline
            generator = pipeline("text-generation", model=model)
            result = generator(message, max_length=100, num_return_sequences=1)
            return result[0]["generated_text"]
        except ImportError:
            return "pip install transformers torch"
        except Exception as e:
            return f"Error: {e}"


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

def interactive_llm_demo():
    """Interactive demo to test different LLMs"""
    print("\n" + "=" * 60)
    print("Interactive LLM Demo - Test Multiple AI Models")
    print("=" * 60)
    
    llm = LLMInterface()
    
    print("""
Choose an LLM provider:
1. OpenAI GPT (requires OPENAI_API_KEY)
2. Anthropic Claude (requires ANTHROPIC_API_KEY)
3. Google Gemini (requires GOOGLE_API_KEY)
4. Ollama Local (FREE - no API key!)
5. Hugging Face Local (FREE - runs on your machine)
""")
    
    choice = input("Select provider (1-5): ").strip()
    prompt = input("Enter your message: ").strip()
    
    if not prompt:
        prompt = "Explain what a large language model is in simple terms."
    
    print("\nGenerating response...\n")
    
    if choice == "1":
        response = llm.chat_openai(prompt)
    elif choice == "2":
        response = llm.chat_claude(prompt)
    elif choice == "3":
        response = llm.chat_gemini(prompt)
    elif choice == "4":
        response = llm.chat_ollama(prompt)
    elif choice == "5":
        response = llm.chat_huggingface(prompt)
    else:
        response = "Invalid choice"
    
    print(f"Response:\n{response}")


def main():
    print("\n" + "=" * 60)
    print("   MODULE 9: LARGE LANGUAGE MODELS (LLMs)")
    print("   360-Degree Coverage with Multiple GenAI Providers")
    print("=" * 60)
    
    while True:
        print("\n" + "-" * 60)
        print("Choose an option:")
        print("1. Learn about LLMs")
        print("2. Interactive LLM Demo (Test different models)")
        print("3. View Code Examples")
        print("0. Exit")
        print("-" * 60)
        
        choice = input("\nEnter choice (0-3): ").strip()
        
        if choice == "0":
            print("\nGoodbye!")
            break
        elif choice == "1":
            explain_llms()
        elif choice == "2":
            interactive_llm_demo()
        elif choice == "3":
            print(code_examples)
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
