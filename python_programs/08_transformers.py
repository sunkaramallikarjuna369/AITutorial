"""
=============================================================================
MODULE 08: TRANSFORMERS - COMPREHENSIVE 360 DEGREE COVERAGE
=============================================================================

This module provides COMPLETE coverage of Transformers including:
- 4W+H Explanations (What, Why, When, Where, How)
- Self-Attention Mechanism
- Transformer Architecture
- BERT, GPT, and Variants
- GenAI Model Integrations
- Interview Questions
- Common Pitfalls

SETUP:
------
pip install torch transformers

For GenAI features:
pip install openai anthropic google-generativeai

=============================================================================
"""

import os
import math
from typing import List, Tuple, Dict, Optional

# Try to import optional libraries
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# =============================================================================
# GENAI INTEGRATION
# =============================================================================

class TransformerAssistant:
    """
    Use GenAI models for transformer explanations
    
    Supports: OpenAI, Anthropic Claude, Google Gemini, Ollama, HuggingFace
    """
    
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_key = os.getenv("GOOGLE_API_KEY")
    
    def explain_with_openai(self, concept: str) -> str:
        """Get AI explanation using OpenAI GPT"""
        try:
            from openai import OpenAI
            if not self.openai_key:
                return f"[Set OPENAI_API_KEY for AI explanations]"
            
            client = OpenAI(api_key=self.openai_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "user",
                    "content": f"Explain {concept} in transformers with a simple analogy and the math."
                }],
                max_tokens=500
            )
            return response.choices[0].message.content
        except ImportError:
            return "[Install openai: pip install openai]"
        except Exception as e:
            return f"[OpenAI Error: {e}]"
    
    def explain_with_ollama(self, concept: str, model: str = "llama2") -> str:
        """Get AI explanation using Ollama (FREE, local)"""
        try:
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": f"Explain {concept} in transformers with an example.",
                    "stream": False
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            return f"[Ollama error: {response.status_code}]"
        except Exception as e:
            return f"[Start Ollama: ollama serve. Error: {e}]"
    
    def use_huggingface_model(self, text: str, task: str = "sentiment") -> str:
        """Use HuggingFace transformers (FREE, local)"""
        if not HAS_TRANSFORMERS:
            return "[Install transformers: pip install transformers torch]"
        
        try:
            if task == "sentiment":
                classifier = pipeline("sentiment-analysis")
                result = classifier(text[:512])
                return f"Sentiment: {result[0]['label']} ({result[0]['score']:.2f})"
            elif task == "fill-mask":
                unmasker = pipeline("fill-mask")
                result = unmasker(text)
                return str(result[:3])
            elif task == "summarization":
                summarizer = pipeline("summarization")
                result = summarizer(text[:1024], max_length=100)
                return result[0]['summary_text']
            else:
                return f"[Task '{task}' not supported]"
        except Exception as e:
            return f"[HuggingFace Error: {e}]"


# =============================================================================
# SECTION 1: COMPREHENSIVE TRANSFORMER EXPLANATION (4W+H)
# =============================================================================

def explain_transformers_comprehensive():
    """Comprehensive 360-degree explanation of Transformers"""
    print("=" * 70)
    print("TRANSFORMERS - COMPREHENSIVE 360 DEGREE COVERAGE")
    print("=" * 70)
    
    print("""
WHAT IS A TRANSFORMER?
======================

A Transformer is a neural network architecture that uses self-attention
to process sequences in parallel, without recurrence or convolution.

KEY INNOVATION: ATTENTION IS ALL YOU NEED
    - No RNNs (sequential processing)
    - No CNNs (local patterns)
    - Just attention (global relationships)

ARCHITECTURE:
    Input -> Embedding -> [Encoder] x N -> [Decoder] x N -> Output
    
    Encoder: Self-Attention -> Feed-Forward
    Decoder: Masked Self-Attention -> Cross-Attention -> Feed-Forward

WHY ARE TRANSFORMERS IMPORTANT?
===============================

1. PARALLELIZATION: Process entire sequence at once (vs RNN's sequential)
2. LONG-RANGE DEPENDENCIES: Attention connects any two positions directly
3. SCALABILITY: Performance improves with more data and compute
4. VERSATILITY: Works for text, images, audio, video, code

WHEN TO USE TRANSFORMERS?
=========================

USE FOR:
    - Natural language processing (all tasks)
    - Machine translation
    - Text generation
    - Image classification (ViT)
    - Speech recognition
    - Code generation

WHERE ARE TRANSFORMERS USED?
============================

MODELS:
    - GPT-4, ChatGPT (OpenAI)
    - Claude (Anthropic)
    - Gemini (Google)
    - BERT, T5 (Google)
    - LLaMA (Meta)
    - Stable Diffusion (text encoder)

PRODUCTS:
    - ChatGPT, Claude, Gemini
    - Google Search, Google Translate
    - GitHub Copilot
    - Grammarly

HOW DO TRANSFORMERS WORK?
=========================

SELF-ATTENTION:
    For each position, compute attention to all other positions.
    
    Q = X @ W_Q  (Query: What am I looking for?)
    K = X @ W_K  (Key: What do I contain?)
    V = X @ W_V  (Value: What information do I have?)
    
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

MULTI-HEAD ATTENTION:
    Run multiple attention heads in parallel, concatenate results.
    Allows model to attend to different aspects simultaneously.

POSITIONAL ENCODING:
    Since attention is permutation-invariant, add position information.
    PE(pos, 2i) = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

FEED-FORWARD NETWORK:
    FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2
    Applied to each position independently.

LAYER NORMALIZATION + RESIDUAL CONNECTIONS:
    output = LayerNorm(x + Sublayer(x))
    """)


# =============================================================================
# SECTION 2: HISTORY AND EVOLUTION
# =============================================================================

def transformer_history():
    """History and evolution of Transformers"""
    print("\n" + "=" * 70)
    print("HISTORY AND EVOLUTION OF TRANSFORMERS")
    print("=" * 70)
    
    timeline = [
        ("2014", "Seq2Seq + Attention", "Attention for machine translation"),
        ("2017", "Transformer", "'Attention Is All You Need' paper"),
        ("2018", "GPT-1", "Generative Pre-trained Transformer"),
        ("2018", "BERT", "Bidirectional Encoder Representations"),
        ("2019", "GPT-2", "1.5B parameters, text generation"),
        ("2019", "T5", "Text-to-Text Transfer Transformer"),
        ("2020", "GPT-3", "175B parameters, few-shot learning"),
        ("2020", "Vision Transformer", "Transformers for images"),
        ("2021", "CLIP", "Vision-language model"),
        ("2022", "ChatGPT", "Conversational AI"),
        ("2022", "InstructGPT", "RLHF for alignment"),
        ("2023", "GPT-4", "Multimodal, reasoning"),
        ("2023", "LLaMA", "Open-source LLMs"),
        ("2024", "Mixture of Experts", "Efficient scaling"),
    ]
    
    print("\nTIMELINE:")
    print("-" * 70)
    for year, event, description in timeline:
        print(f"{year}: {event} - {description}")


# =============================================================================
# SECTION 3: SELF-ATTENTION MECHANISM
# =============================================================================

def attention_explanation():
    """Detailed explanation of self-attention"""
    print("\n" + "=" * 70)
    print("SELF-ATTENTION MECHANISM")
    print("=" * 70)
    
    print("""
INTUITION:
    For each word, look at all other words to understand context.
    
    Example: "The animal didn't cross the street because it was too tired"
    What does "it" refer to? Attention helps figure this out!

STEP-BY-STEP:

1. CREATE Q, K, V:
   For input X (seq_len x d_model):
   Q = X @ W_Q  (seq_len x d_k)
   K = X @ W_K  (seq_len x d_k)
   V = X @ W_V  (seq_len x d_v)

2. COMPUTE ATTENTION SCORES:
   scores = Q @ K^T  (seq_len x seq_len)
   
   Each position has a score with every other position.

3. SCALE:
   scores = scores / sqrt(d_k)
   
   Prevents softmax from having extremely small gradients.

4. SOFTMAX:
   attention_weights = softmax(scores, dim=-1)
   
   Weights sum to 1 for each position.

5. WEIGHTED SUM:
   output = attention_weights @ V
   
   Each position is a weighted combination of all values.

MULTI-HEAD ATTENTION:
    head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
    MultiHead = Concat(head_1, ..., head_h) @ W_O
    
    Multiple heads attend to different aspects (syntax, semantics, etc.)

MASKED ATTENTION (for decoder):
    Prevent attending to future positions during training.
    mask = upper triangular matrix of -infinity
    scores = scores + mask
    """)


def attention_demo():
    """Demo self-attention calculation"""
    print("\n" + "=" * 70)
    print("SELF-ATTENTION DEMO")
    print("=" * 70)
    
    if not HAS_NUMPY:
        print("\nInstall numpy: pip install numpy")
        return
    
    # Simple attention calculation
    print("\nSimple Self-Attention Example:")
    print("-" * 40)
    
    # Input: 3 words, 4 dimensions each
    X = np.array([
        [1.0, 0.0, 1.0, 0.0],  # Word 1
        [0.0, 1.0, 0.0, 1.0],  # Word 2
        [1.0, 1.0, 0.0, 0.0],  # Word 3
    ])
    
    # Simple: Q = K = V = X (no learned weights)
    Q = K = V = X
    d_k = X.shape[1]
    
    # Attention scores
    scores = Q @ K.T / np.sqrt(d_k)
    print(f"Input X shape: {X.shape}")
    print(f"Attention scores (Q @ K^T / sqrt(d_k)):\n{scores}")
    
    # Softmax
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    attention_weights = softmax(scores)
    print(f"\nAttention weights (softmax):\n{attention_weights}")
    
    # Output
    output = attention_weights @ V
    print(f"\nOutput (weights @ V):\n{output}")
    
    print("\nInterpretation:")
    print("Each row shows how much each word attends to other words.")
    print("Higher values = more attention.")


# =============================================================================
# SECTION 4: TRANSFORMER ARCHITECTURE
# =============================================================================

def transformer_architecture():
    """Detailed transformer architecture"""
    print("\n" + "=" * 70)
    print("TRANSFORMER ARCHITECTURE")
    print("=" * 70)
    
    print("""
ENCODER-DECODER ARCHITECTURE (Original):

ENCODER (processes input):
    for each layer:
        x = LayerNorm(x + MultiHeadAttention(x, x, x))
        x = LayerNorm(x + FeedForward(x))

DECODER (generates output):
    for each layer:
        x = LayerNorm(x + MaskedMultiHeadAttention(x, x, x))
        x = LayerNorm(x + MultiHeadAttention(x, encoder_output, encoder_output))
        x = LayerNorm(x + FeedForward(x))

COMPONENTS:

1. INPUT EMBEDDING:
   token_ids -> embedding_matrix -> vectors
   
2. POSITIONAL ENCODING:
   Add position information to embeddings
   
3. MULTI-HEAD ATTENTION:
   Multiple attention heads in parallel
   
4. FEED-FORWARD NETWORK:
   Two linear layers with ReLU
   FFN(x) = max(0, xW1 + b1)W2 + b2
   
5. LAYER NORMALIZATION:
   Normalize across features (not batch)
   
6. RESIDUAL CONNECTIONS:
   Add input to output of each sublayer

HYPERPARAMETERS:
    d_model = 512 (embedding dimension)
    n_heads = 8 (attention heads)
    d_ff = 2048 (feed-forward dimension)
    n_layers = 6 (encoder/decoder layers)
    dropout = 0.1
    """)


# =============================================================================
# SECTION 5: BERT vs GPT
# =============================================================================

def bert_vs_gpt():
    """Compare BERT and GPT architectures"""
    print("\n" + "=" * 70)
    print("BERT vs GPT")
    print("=" * 70)
    
    print("""
BERT (Bidirectional Encoder Representations from Transformers):
    - Architecture: Encoder only
    - Direction: Bidirectional (sees all tokens)
    - Pre-training: Masked Language Modeling (MLM) + Next Sentence Prediction
    - Use case: Understanding tasks (classification, NER, QA)
    - Example: "The [MASK] sat on the mat" -> predict "cat"

GPT (Generative Pre-trained Transformer):
    - Architecture: Decoder only
    - Direction: Unidirectional (left-to-right)
    - Pre-training: Causal Language Modeling (predict next token)
    - Use case: Generation tasks (text completion, chatbots)
    - Example: "The cat sat on" -> predict "the"

COMPARISON:
    
    Feature          | BERT              | GPT
    -----------------|-------------------|------------------
    Architecture     | Encoder           | Decoder
    Attention        | Bidirectional     | Causal (masked)
    Pre-training     | MLM + NSP         | Next token
    Best for         | Understanding     | Generation
    Fine-tuning      | Task-specific     | Prompt-based
    
T5 (Text-to-Text Transfer Transformer):
    - Architecture: Encoder-Decoder
    - Approach: Everything is text-to-text
    - Example: "translate English to French: Hello" -> "Bonjour"
    - Unified framework for all NLP tasks
    """)


# =============================================================================
# SECTION 6: PYTORCH IMPLEMENTATION
# =============================================================================

def pytorch_transformer_demo():
    """Demo transformer implementation"""
    print("\n" + "=" * 70)
    print("PYTORCH TRANSFORMER DEMO")
    print("=" * 70)
    
    if not HAS_TORCH:
        print("\nInstall PyTorch: pip install torch")
        print("""
Example code:

import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.W_o(out)
        """)
        return
    
    # Simple self-attention implementation
    class SimpleSelfAttention(nn.Module):
        def __init__(self, d_model, n_heads):
            super().__init__()
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)
        
        def forward(self, x):
            batch_size, seq_len, d_model = x.shape
            
            Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            attention = F.softmax(scores, dim=-1)
            out = torch.matmul(attention, V)
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
            return self.W_o(out)
    
    # Test
    d_model = 64
    n_heads = 4
    seq_len = 10
    batch_size = 2
    
    attention = SimpleSelfAttention(d_model, n_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    output = attention(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in attention.parameters()):,}")


# =============================================================================
# SECTION 7: INTERVIEW QUESTIONS
# =============================================================================

def interview_questions():
    """Common Transformer interview questions"""
    print("\n" + "=" * 70)
    print("TRANSFORMER INTERVIEW QUESTIONS")
    print("=" * 70)
    
    questions = [
        ("Why do we scale attention scores by sqrt(d_k)?",
         "Dot products grow with dimension, pushing softmax into regions with tiny gradients. Scaling keeps variance stable."),
        ("What is the purpose of positional encoding?",
         "Self-attention is permutation-invariant. Positional encoding adds position information so the model knows word order."),
        ("Why use multi-head attention instead of single head?",
         "Multiple heads can attend to different aspects (syntax, semantics, coreference) simultaneously. More expressive."),
        ("What is the difference between encoder and decoder attention?",
         "Encoder: bidirectional self-attention. Decoder: masked self-attention (can't see future) + cross-attention to encoder."),
        ("How does BERT's MLM training work?",
         "Randomly mask 15% of tokens. Model predicts masked tokens using bidirectional context. Forces deep understanding."),
        ("What is the computational complexity of self-attention?",
         "O(n^2 * d) where n is sequence length, d is dimension. Quadratic in sequence length is the main bottleneck."),
    ]
    
    for i, (q, a) in enumerate(questions, 1):
        print(f"\nQ{i}: {q}")
        print(f"A: {a}")


# =============================================================================
# SECTION 8: COMMON PITFALLS
# =============================================================================

def common_pitfalls():
    """Common mistakes with Transformers"""
    print("\n" + "=" * 70)
    print("COMMON TRANSFORMER PITFALLS")
    print("=" * 70)
    
    pitfalls = [
        ("Forgetting positional encoding", "Without it, model can't distinguish word order"),
        ("Wrong attention mask", "Decoder needs causal mask; padding needs attention mask"),
        ("Not scaling attention scores", "Leads to vanishing gradients in softmax"),
        ("Sequence length limits", "Attention is O(n^2); use efficient attention for long sequences"),
        ("Not using pre-trained models", "Fine-tuning almost always beats training from scratch"),
        ("Wrong tokenizer", "Must use same tokenizer as pre-trained model"),
    ]
    
    for mistake, solution in pitfalls:
        print(f"\nMistake: {mistake}")
        print(f"Solution: {solution}")


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main():
    print("=" * 70)
    print("   MODULE 08: TRANSFORMERS - COMPREHENSIVE 360 DEGREE COVERAGE")
    print("=" * 70)
    
    assistant = TransformerAssistant()
    
    while True:
        print("\n" + "-" * 70)
        print("Choose a topic:")
        print()
        print("FUNDAMENTALS:")
        print("  1. What is a Transformer? (4W+H)")
        print("  2. History and Evolution")
        print()
        print("ARCHITECTURE:")
        print("  3. Self-Attention Explained")
        print("  4. Self-Attention Demo")
        print("  5. Transformer Architecture")
        print("  6. BERT vs GPT")
        print("  7. PyTorch Implementation")
        print()
        print("DEEP DIVE:")
        print("  8. Interview Questions")
        print("  9. Common Pitfalls")
        print()
        print("GENAI FEATURES:")
        print("  10. AI Explanation (OpenAI)")
        print("  11. AI Explanation (Ollama - FREE)")
        print("  12. Use HuggingFace Model (FREE)")
        print()
        print("  13. Run ALL Topics")
        print("  0. Exit")
        print("-" * 70)
        
        choice = input("\nEnter choice (0-13): ").strip()
        
        if choice == "0":
            print("\nHappy learning!")
            break
        elif choice == "1":
            explain_transformers_comprehensive()
        elif choice == "2":
            transformer_history()
        elif choice == "3":
            attention_explanation()
        elif choice == "4":
            attention_demo()
        elif choice == "5":
            transformer_architecture()
        elif choice == "6":
            bert_vs_gpt()
        elif choice == "7":
            pytorch_transformer_demo()
        elif choice == "8":
            interview_questions()
        elif choice == "9":
            common_pitfalls()
        elif choice == "10":
            concept = input("Enter concept to explain: ").strip()
            print(assistant.explain_with_openai(concept))
        elif choice == "11":
            concept = input("Enter concept to explain: ").strip()
            print(assistant.explain_with_ollama(concept))
        elif choice == "12":
            text = input("Enter text for sentiment analysis: ").strip()
            print(assistant.use_huggingface_model(text, "sentiment"))
        elif choice == "13":
            explain_transformers_comprehensive()
            transformer_history()
            attention_explanation()
            attention_demo()
            transformer_architecture()
            bert_vs_gpt()
            pytorch_transformer_demo()
            interview_questions()
            common_pitfalls()
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
