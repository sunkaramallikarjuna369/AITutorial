"""
=============================================================================
MODULE 13: AI FRAMEWORKS - COMPREHENSIVE 360 DEGREE COVERAGE
=============================================================================

This module provides COMPLETE coverage of AI Frameworks including:
- 4W+H Explanations (What, Why, When, Where, How)
- PyTorch, TensorFlow, JAX
- LangChain, LlamaIndex
- Comparison and Selection Guide
- Interview Questions
- Common Pitfalls

SETUP:
------
pip install torch tensorflow jax langchain llama-index

=============================================================================
"""

import os
from typing import List, Dict, Optional


# =============================================================================
# GENAI INTEGRATION
# =============================================================================

class FrameworkAssistant:
    """Use GenAI models for framework explanations"""
    
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
                messages=[{"role": "user", "content": f"Compare {concept} in AI frameworks with code examples."}],
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
                json={"model": model, "prompt": f"Explain {concept} in AI frameworks.", "stream": False},
                timeout=60
            )
            return response.json().get("response", "") if response.status_code == 200 else "[Ollama error]"
        except Exception as e:
            return f"[Start Ollama: ollama serve. Error: {e}]"


# =============================================================================
# SECTION 1: COMPREHENSIVE FRAMEWORKS OVERVIEW
# =============================================================================

def explain_frameworks_comprehensive():
    """Comprehensive 360-degree explanation of AI Frameworks"""
    print("=" * 70)
    print("AI FRAMEWORKS - COMPREHENSIVE 360 DEGREE COVERAGE")
    print("=" * 70)
    
    print("""
WHAT ARE AI FRAMEWORKS?
=======================

AI frameworks are software libraries that provide tools, APIs, and
abstractions for building, training, and deploying AI/ML models.

CATEGORIES:

1. DEEP LEARNING FRAMEWORKS:
   - PyTorch, TensorFlow, JAX
   - Build and train neural networks

2. LLM APPLICATION FRAMEWORKS:
   - LangChain, LlamaIndex, Semantic Kernel
   - Build applications with LLMs

3. ML LIBRARIES:
   - scikit-learn, XGBoost, LightGBM
   - Traditional ML algorithms

4. DEPLOYMENT FRAMEWORKS:
   - ONNX, TensorRT, vLLM
   - Optimize and serve models

WHY USE FRAMEWORKS?
===================

1. ABSTRACTION: Hide low-level complexity
2. EFFICIENCY: Optimized implementations
3. ECOSYSTEM: Pre-built models, tools, community
4. PRODUCTIVITY: Focus on problem, not infrastructure
5. PORTABILITY: Run on CPU, GPU, TPU

WHEN TO USE WHICH?
==================

PYTORCH: Research, flexibility, debugging
TENSORFLOW: Production, mobile, edge devices
JAX: High-performance computing, TPUs
LANGCHAIN: LLM applications, chains, agents
LLAMAINDEX: RAG, document Q&A

WHERE ARE THEY USED?
====================

PYTORCH: Meta, OpenAI, HuggingFace
TENSORFLOW: Google, enterprise production
JAX: Google DeepMind, research
LANGCHAIN: Startups, LLM applications
    """)


# =============================================================================
# SECTION 2: DEEP LEARNING FRAMEWORKS
# =============================================================================

def deep_learning_frameworks():
    """Detailed comparison of deep learning frameworks"""
    print("\n" + "=" * 70)
    print("DEEP LEARNING FRAMEWORKS")
    print("=" * 70)
    
    print("""
PYTORCH
=======
Creator: Meta (Facebook)
Philosophy: "Define by run" (dynamic graphs)

Strengths:
- Pythonic, intuitive API
- Easy debugging (standard Python debugger)
- Dominant in research
- HuggingFace ecosystem

Example:
    import torch
    import torch.nn as nn
    
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(784, 10)
        
        def forward(self, x):
            return self.fc(x)
    
    model = Model()
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

TENSORFLOW
==========
Creator: Google
Philosophy: Production-ready, scalable

Strengths:
- TensorFlow Serving (production deployment)
- TensorFlow Lite (mobile/edge)
- TensorBoard (visualization)
- Keras high-level API

Example:
    import tensorflow as tf
    from tensorflow import keras
    
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(x_train, y_train, epochs=5)

JAX
===
Creator: Google
Philosophy: Functional, composable transformations

Strengths:
- Automatic differentiation (grad)
- JIT compilation (jit)
- Vectorization (vmap)
- Parallelization (pmap)
- TPU optimization

Example:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap
    
    def loss(params, x, y):
        pred = jnp.dot(x, params)
        return jnp.mean((pred - y) ** 2)
    
    grad_loss = jit(grad(loss))
    
    # Vectorize over batch
    batched_grad = vmap(grad_loss, in_axes=(None, 0, 0))

COMPARISON TABLE:
================

Feature          | PyTorch    | TensorFlow | JAX
-----------------|------------|------------|------------
Ease of use      | Excellent  | Good       | Moderate
Debugging        | Excellent  | Good       | Moderate
Production       | Good       | Excellent  | Good
Research         | Excellent  | Good       | Excellent
Mobile/Edge      | Limited    | Excellent  | Limited
TPU support      | Limited    | Good       | Excellent
Community        | Large      | Large      | Growing
    """)


# =============================================================================
# SECTION 3: LLM APPLICATION FRAMEWORKS
# =============================================================================

def llm_frameworks():
    """LLM application frameworks"""
    print("\n" + "=" * 70)
    print("LLM APPLICATION FRAMEWORKS")
    print("=" * 70)
    
    print("""
LANGCHAIN
=========
Purpose: Build applications with LLMs

Components:
- Models: LLM wrappers (OpenAI, Anthropic, etc.)
- Prompts: Templates and management
- Chains: Sequence of operations
- Agents: LLM decides actions
- Memory: Conversation history
- Retrievers: RAG components

Example:
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    
    llm = ChatOpenAI(model="gpt-4")
    
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Write a poem about {topic}"
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run("artificial intelligence")

LLAMAINDEX
==========
Purpose: Connect LLMs to data (RAG focus)

Components:
- Data Connectors: Load from various sources
- Indexes: Organize data for retrieval
- Query Engines: Answer questions
- Agents: Complex reasoning

Example:
    from llama_index import VectorStoreIndex, SimpleDirectoryReader
    
    # Load documents
    documents = SimpleDirectoryReader("data/").load_data()
    
    # Create index
    index = VectorStoreIndex.from_documents(documents)
    
    # Query
    query_engine = index.as_query_engine()
    response = query_engine.query("What is the main topic?")

SEMANTIC KERNEL (Microsoft)
===========================
Purpose: Enterprise LLM applications

Features:
- Plugins (skills)
- Planners (orchestration)
- Memory (context management)
- .NET and Python support

HAYSTACK
========
Purpose: Production-ready NLP pipelines

Features:
- Document stores
- Retrievers
- Readers
- Pipelines

COMPARISON:
===========

Framework       | Best For                    | Complexity
----------------|-----------------------------|-----------
LangChain       | General LLM apps, agents    | Medium
LlamaIndex      | RAG, document Q&A           | Low
Semantic Kernel | Enterprise, .NET            | Medium
Haystack        | Production NLP pipelines    | Medium
    """)


# =============================================================================
# SECTION 4: FRAMEWORK SELECTION GUIDE
# =============================================================================

def selection_guide():
    """Framework selection guide"""
    print("\n" + "=" * 70)
    print("FRAMEWORK SELECTION GUIDE")
    print("=" * 70)
    
    print("""
DECISION TREE:

1. What are you building?
   
   A) Training neural networks from scratch
      -> PyTorch (research) or TensorFlow (production)
   
   B) LLM-powered application
      -> LangChain (general) or LlamaIndex (RAG)
   
   C) Traditional ML (classification, regression)
      -> scikit-learn
   
   D) High-performance computing / TPUs
      -> JAX

2. What's your priority?

   Research & Experimentation -> PyTorch
   Production Deployment -> TensorFlow
   Maximum Performance -> JAX
   Rapid Prototyping -> LangChain + OpenAI
   Document Q&A -> LlamaIndex

3. What's your team's experience?

   Python beginners -> Keras (TensorFlow)
   Experienced ML -> PyTorch
   Functional programming -> JAX
   Full-stack developers -> LangChain

COMMON COMBINATIONS:
====================

Research Lab:
    PyTorch + HuggingFace + Weights & Biases

Production ML:
    TensorFlow + TF Serving + Kubernetes

LLM Startup:
    LangChain + OpenAI + Pinecone + FastAPI

Enterprise RAG:
    LlamaIndex + Azure OpenAI + Cognitive Search
    """)


# =============================================================================
# SECTION 5: INTERVIEW QUESTIONS
# =============================================================================

def interview_questions():
    """Common framework interview questions"""
    print("\n" + "=" * 70)
    print("AI FRAMEWORKS INTERVIEW QUESTIONS")
    print("=" * 70)
    
    questions = [
        ("PyTorch vs TensorFlow: When to use each?",
         "PyTorch: Research, debugging, HuggingFace models. TensorFlow: Production, mobile (TFLite), enterprise. Both are capable; choose based on team expertise and deployment needs."),
        ("What is eager vs graph execution?",
         "Eager: Operations execute immediately (PyTorch default, TF 2.0). Graph: Build computation graph first, then execute (faster, optimizable). PyTorch uses torch.jit for graphs."),
        ("What is LangChain and when would you use it?",
         "Framework for LLM applications. Use for: chains of prompts, agents with tools, RAG systems, memory management. Abstracts common patterns."),
        ("How do you choose between LangChain and LlamaIndex?",
         "LangChain: General LLM apps, agents, complex chains. LlamaIndex: Focused on RAG, document indexing, simpler API for Q&A over documents."),
        ("What is JAX's main advantage?",
         "Composable function transformations: grad (autodiff), jit (compilation), vmap (vectorization), pmap (parallelization). Excellent for research requiring custom training loops."),
    ]
    
    for i, (q, a) in enumerate(questions, 1):
        print(f"\nQ{i}: {q}")
        print(f"A: {a}")


# =============================================================================
# SECTION 6: COMMON PITFALLS
# =============================================================================

def common_pitfalls():
    """Common framework mistakes"""
    print("\n" + "=" * 70)
    print("COMMON FRAMEWORK PITFALLS")
    print("=" * 70)
    
    pitfalls = [
        ("Choosing framework before understanding problem", "Define requirements first, then select framework"),
        ("Over-engineering with LangChain", "Start simple; add complexity only when needed"),
        ("Ignoring deployment requirements", "Consider production needs early (TFLite, ONNX, etc.)"),
        ("Not using pre-trained models", "HuggingFace has models for most tasks; don't reinvent"),
        ("Mixing frameworks unnecessarily", "Stick to one ecosystem when possible"),
        ("Ignoring framework updates", "APIs change; keep dependencies updated"),
    ]
    
    for mistake, solution in pitfalls:
        print(f"\nMistake: {mistake}")
        print(f"Solution: {solution}")


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main():
    print("=" * 70)
    print("   MODULE 13: AI FRAMEWORKS - COMPREHENSIVE 360 DEGREE COVERAGE")
    print("=" * 70)
    
    assistant = FrameworkAssistant()
    
    while True:
        print("\n" + "-" * 70)
        print("Choose a topic:")
        print()
        print("FUNDAMENTALS:")
        print("  1. AI Frameworks Overview")
        print("  2. Deep Learning Frameworks (PyTorch, TensorFlow, JAX)")
        print("  3. LLM Frameworks (LangChain, LlamaIndex)")
        print("  4. Framework Selection Guide")
        print()
        print("DEEP DIVE:")
        print("  5. Interview Questions")
        print("  6. Common Pitfalls")
        print()
        print("GENAI FEATURES:")
        print("  7. AI Explanation (OpenAI)")
        print("  8. AI Explanation (Ollama - FREE)")
        print()
        print("  9. Run ALL Topics")
        print("  0. Exit")
        print("-" * 70)
        
        choice = input("\nEnter choice (0-9): ").strip()
        
        if choice == "0":
            print("\nHappy learning!")
            break
        elif choice == "1":
            explain_frameworks_comprehensive()
        elif choice == "2":
            deep_learning_frameworks()
        elif choice == "3":
            llm_frameworks()
        elif choice == "4":
            selection_guide()
        elif choice == "5":
            interview_questions()
        elif choice == "6":
            common_pitfalls()
        elif choice == "7":
            concept = input("Enter concept to explain: ").strip()
            print(assistant.explain_with_openai(concept))
        elif choice == "8":
            concept = input("Enter concept to explain: ").strip()
            print(assistant.explain_with_ollama(concept))
        elif choice == "9":
            explain_frameworks_comprehensive()
            deep_learning_frameworks()
            llm_frameworks()
            selection_guide()
            interview_questions()
            common_pitfalls()
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
