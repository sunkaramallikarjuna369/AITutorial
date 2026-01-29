"""
=============================================================================
MODULE 03: NEURAL NETWORKS - COMPREHENSIVE 360 DEGREE COVERAGE
=============================================================================

This module provides COMPLETE coverage of Neural Networks including:
- 4W+H Explanations (What, Why, When, Where, How)
- History and Evolution
- Types of Neural Networks
- Real-world Applications
- Hands-on Implementations
- GenAI Model Integrations
- Interview Questions
- Common Pitfalls

SETUP:
------
pip install numpy torch  # Optional but recommended

For GenAI features:
pip install openai anthropic google-generativeai transformers

=============================================================================
"""

import math
import random
import os
from typing import List, Tuple, Optional

# Try to import optional libraries
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# GENAI INTEGRATION FOR NEURAL NETWORK EXPLANATIONS
# =============================================================================

class NeuralNetworkExplainer:
    """
    Use GenAI models to explain neural network concepts
    
    Supports: OpenAI, Anthropic Claude, Google Gemini, Ollama (local), HuggingFace
    """
    
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_key = os.getenv("GOOGLE_API_KEY")
    
    def explain_with_openai(self, concept: str, detail_level: str = "beginner") -> str:
        """Get AI explanation using OpenAI GPT"""
        try:
            from openai import OpenAI
            if not self.openai_key:
                return f"[Set OPENAI_API_KEY to get AI explanations of {concept}]"
            
            client = OpenAI(api_key=self.openai_key)
            prompt = f"""Explain {concept} in neural networks for a {detail_level} audience.
            Include:
            1. Simple analogy
            2. Mathematical intuition
            3. Code example
            4. Real-world application
            Keep it concise but comprehensive."""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            return response.choices[0].message.content
        except ImportError:
            return "[Install openai: pip install openai]"
        except Exception as e:
            return f"[OpenAI Error: {e}]"
    
    def explain_with_claude(self, concept: str) -> str:
        """Get AI explanation using Anthropic Claude"""
        try:
            import anthropic
            if not self.anthropic_key:
                return f"[Set ANTHROPIC_API_KEY for Claude explanations]"
            
            client = anthropic.Anthropic(api_key=self.anthropic_key)
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": f"Explain {concept} in neural networks with a simple analogy and example."
                }]
            )
            return response.content[0].text
        except ImportError:
            return "[Install anthropic: pip install anthropic]"
        except Exception as e:
            return f"[Claude Error: {e}]"
    
    def explain_with_ollama(self, concept: str, model: str = "llama2") -> str:
        """Get AI explanation using Ollama (FREE, local)"""
        try:
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": f"Explain {concept} in neural networks simply with an example.",
                    "stream": False
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            return f"[Ollama error: {response.status_code}]"
        except Exception as e:
            return f"[Start Ollama: ollama serve. Error: {e}]"


# =============================================================================
# SECTION 1: COMPREHENSIVE NEURAL NETWORK EXPLANATION (4W+H)
# =============================================================================

def explain_neural_networks_comprehensive():
    """
    Comprehensive 360-degree explanation of Neural Networks
    """
    print("=" * 70)
    print("NEURAL NETWORKS - COMPREHENSIVE 360 DEGREE COVERAGE")
    print("=" * 70)
    
    print("""
WHAT IS A NEURAL NETWORK?
=========================

A Neural Network is a computational model inspired by the human brain,
consisting of interconnected nodes (neurons) organized in layers that
can learn patterns from data.

STRUCTURE:
    Input Layer -> Hidden Layer(s) -> Output Layer

MATHEMATICAL DEFINITION:
    For a single neuron:
    output = activation(sum(inputs * weights) + bias)
    
    For a layer:
    output = activation(X @ W + b)

WHY USE NEURAL NETWORKS?
========================

1. UNIVERSAL APPROXIMATORS:
   - Can learn ANY continuous function given enough neurons
   - Proven by Universal Approximation Theorem (1989)

2. AUTOMATIC FEATURE LEARNING:
   - No need for manual feature engineering
   - Network discovers relevant features automatically

3. SCALABILITY:
   - Performance improves with more data
   - Can leverage parallel computing (GPUs)

WHEN TO USE NEURAL NETWORKS?
============================

USE WHEN:
    - You have LOTS of data (thousands to millions of examples)
    - The problem is complex (non-linear relationships)
    - Feature engineering is difficult
    - You need state-of-the-art performance

DON'T USE WHEN:
    - You have small datasets (< 1000 examples)
    - Interpretability is critical (use decision trees)
    - Simple linear relationships exist

WHERE ARE NEURAL NETWORKS USED?
===============================

- COMPUTER VISION: Image classification, object detection
- NLP: Translation, sentiment analysis, ChatGPT
- SPEECH: Recognition, text-to-speech
- AUTONOMOUS SYSTEMS: Self-driving cars, robotics
- SCIENCE: Drug discovery (AlphaFold), climate modeling

HOW DO NEURAL NETWORKS WORK?
============================

STEP 1: FORWARD PROPAGATION
    Input -> Multiply by weights -> Add bias -> Apply activation -> Output

STEP 2: LOSS CALCULATION
    Compare prediction to ground truth

STEP 3: BACKPROPAGATION
    Calculate gradients using chain rule

STEP 4: WEIGHT UPDATE (Gradient Descent)
    W = W - learning_rate * gradient

STEP 5: REPEAT
    Iterate over entire dataset multiple times (epochs)
    """)


# =============================================================================
# SECTION 2: HISTORY AND EVOLUTION
# =============================================================================

def neural_network_history():
    """History and evolution of neural networks"""
    print("\n" + "=" * 70)
    print("HISTORY AND EVOLUTION OF NEURAL NETWORKS")
    print("=" * 70)
    
    timeline = [
        ("1943", "McCulloch-Pitts Neuron", "First mathematical model of a neuron"),
        ("1958", "Perceptron (Rosenblatt)", "First trainable neural network"),
        ("1969", "AI Winter Begins", "Minsky showed perceptrons can't learn XOR"),
        ("1986", "Backpropagation", "Efficient training algorithm"),
        ("1998", "LeNet-5 (LeCun)", "First successful CNN"),
        ("2012", "AlexNet", "Deep learning revolution begins"),
        ("2017", "Transformer", "Attention mechanism revolutionizes NLP"),
        ("2022", "ChatGPT", "Conversational AI goes mainstream"),
        ("2023", "GPT-4, Claude, Gemini", "Multimodal AI"),
    ]
    
    print("\nTIMELINE:")
    print("-" * 70)
    for year, event, description in timeline:
        print(f"{year}: {event} - {description}")


# =============================================================================
# SECTION 3: TYPES OF NEURAL NETWORKS
# =============================================================================

def neural_network_types():
    """Comprehensive overview of neural network types"""
    print("\n" + "=" * 70)
    print("TYPES OF NEURAL NETWORKS")
    print("=" * 70)
    
    types = [
        ("Feedforward (MLP)", "Data flows one direction", "Tabular data, classification"),
        ("CNN", "Spatial feature extraction", "Images, object detection"),
        ("RNN", "Memory of previous inputs", "Time series, sequences"),
        ("LSTM", "Long-term memory with gates", "Translation, speech"),
        ("Transformer", "Parallel processing with attention", "ChatGPT, BERT"),
        ("GAN", "Two networks compete", "Image generation"),
        ("Autoencoder", "Learns compressed representations", "Denoising, anomaly detection"),
        ("GNN", "Operates on graphs", "Social networks, molecules"),
    ]
    
    for name, feature, use in types:
        print(f"\n{name}")
        print(f"  Key Feature: {feature}")
        print(f"  Use Cases: {use}")


# =============================================================================
# SECTION 4: IMPLEMENTATION - SINGLE NEURON
# =============================================================================

class Neuron:
    """A single artificial neuron"""
    
    def __init__(self, num_inputs: int):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)
    
    def forward(self, inputs: List[float]) -> float:
        total = sum(x * w for x, w in zip(inputs, self.weights)) + self.bias
        return self.sigmoid(total)
    
    def sigmoid(self, x: float) -> float:
        x = max(-500, min(500, x))
        return 1 / (1 + math.exp(-x))
    
    def train(self, inputs: List[float], target: float, lr: float = 0.1) -> float:
        output = self.forward(inputs)
        error = target - output
        gradient = error * output * (1 - output)
        for i in range(len(self.weights)):
            self.weights[i] += lr * gradient * inputs[i]
        self.bias += lr * gradient
        return error ** 2


def single_neuron_demo():
    """Demo: Train a single neuron to learn AND gate"""
    print("\n" + "=" * 70)
    print("SINGLE NEURON DEMO: Learning AND Gate")
    print("=" * 70)
    
    data = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]
    neuron = Neuron(2)
    
    print("Training...")
    for epoch in range(1000):
        total_loss = sum(neuron.train(x, y) for x, y in data)
        if epoch % 200 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss:.6f}")
    
    print("\nResults:")
    for inputs, expected in data:
        output = neuron.forward(inputs)
        print(f"  {inputs} -> {output:.4f} (expected: {expected})")


# =============================================================================
# SECTION 5: IMPLEMENTATION - MULTI-LAYER NETWORK
# =============================================================================

class NeuralNetwork:
    """Multi-layer neural network with backpropagation"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.W1 = [[random.gauss(0, 0.5) for _ in range(input_size)] 
                   for _ in range(hidden_size)]
        self.b1 = [0.0 for _ in range(hidden_size)]
        
        self.W2 = [[random.gauss(0, 0.5) for _ in range(hidden_size)] 
                   for _ in range(output_size)]
        self.b2 = [0.0 for _ in range(output_size)]
    
    def sigmoid(self, x: float) -> float:
        x = max(-500, min(500, x))
        return 1 / (1 + math.exp(-x))
    
    def forward(self, X: List[float]) -> List[float]:
        self.hidden = []
        for i in range(self.hidden_size):
            z = sum(X[j] * self.W1[i][j] for j in range(self.input_size)) + self.b1[i]
            self.hidden.append(self.sigmoid(z))
        
        self.output = []
        for i in range(self.output_size):
            z = sum(self.hidden[j] * self.W2[i][j] for j in range(self.hidden_size)) + self.b2[i]
            self.output.append(self.sigmoid(z))
        
        return self.output
    
    def train_step(self, X: List[float], y: List[float], lr: float = 0.5) -> float:
        self.forward(X)
        
        output_errors = [y[i] - self.output[i] for i in range(self.output_size)]
        output_deltas = [output_errors[i] * self.output[i] * (1 - self.output[i]) 
                        for i in range(self.output_size)]
        
        hidden_errors = [sum(output_deltas[i] * self.W2[i][j] for i in range(self.output_size))
                        for j in range(self.hidden_size)]
        hidden_deltas = [hidden_errors[i] * self.hidden[i] * (1 - self.hidden[i])
                        for i in range(self.hidden_size)]
        
        for i in range(self.output_size):
            for j in range(self.hidden_size):
                self.W2[i][j] += lr * output_deltas[i] * self.hidden[j]
            self.b2[i] += lr * output_deltas[i]
        
        for i in range(self.hidden_size):
            for j in range(self.input_size):
                self.W1[i][j] += lr * hidden_deltas[i] * X[j]
            self.b1[i] += lr * hidden_deltas[i]
        
        return sum(e**2 for e in output_errors)


def xor_demo():
    """Demo: Neural network learning XOR"""
    print("\n" + "=" * 70)
    print("MULTI-LAYER NETWORK DEMO: Learning XOR")
    print("=" * 70)
    
    print("XOR requires hidden layer - single neuron cannot learn it!")
    
    data = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]
    nn = NeuralNetwork(2, 4, 1)
    
    print("\nTraining...")
    for epoch in range(5000):
        total_loss = sum(nn.train_step(x, y) for x, y in data)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss:.6f}")
    
    print("\nResults:")
    for inputs, expected in data:
        output = nn.forward(inputs)
        print(f"  {inputs} -> {output[0]:.4f} (expected: {expected[0]})")


# =============================================================================
# SECTION 6: PYTORCH IMPLEMENTATION
# =============================================================================

def pytorch_demo():
    """Demo: Neural network using PyTorch"""
    print("\n" + "=" * 70)
    print("PYTORCH NEURAL NETWORK DEMO")
    print("=" * 70)
    
    if not HAS_TORCH:
        print("\nInstall PyTorch: pip install torch")
        print("""
Example PyTorch code:

import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 8)
        self.layer2 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x

model = SimpleNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
criterion = nn.BCELoss()
        """)
        return
    
    class SimpleNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(2, 8)
            self.layer2 = nn.Linear(8, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            x = self.relu(self.layer1(x))
            x = self.sigmoid(self.layer2(x))
            return x
    
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    
    model = SimpleNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.BCELoss()
    
    print("Training PyTorch model...")
    for epoch in range(1000):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
    
    print("\nResults:")
    with torch.no_grad():
        predictions = model(X)
        for i in range(len(X)):
            print(f"  {X[i].tolist()} -> {predictions[i].item():.4f} (expected: {y[i].item()})")


# =============================================================================
# SECTION 7: ACTIVATION FUNCTIONS
# =============================================================================

def activation_functions_demo():
    """Comprehensive guide to activation functions"""
    print("\n" + "=" * 70)
    print("ACTIVATION FUNCTIONS")
    print("=" * 70)
    
    print("""
WHY ACTIVATION FUNCTIONS?
Without them, neural networks are just linear functions!
They add NON-LINEARITY, enabling learning of complex patterns.

COMMON ACTIVATION FUNCTIONS:

1. SIGMOID: sigma(x) = 1 / (1 + e^(-x))
   Range: (0, 1)
   Use: Binary classification output
   Problem: Vanishing gradient

2. TANH: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
   Range: (-1, 1)
   Use: Hidden layers (before ReLU era)

3. RELU: ReLU(x) = max(0, x)
   Range: [0, infinity)
   Use: Default for hidden layers
   Problem: Dying ReLU

4. LEAKY RELU: LeakyReLU(x) = x if x > 0 else 0.01*x
   Fixes dying ReLU problem

5. GELU: Used in Transformers (BERT, GPT)
   Smooth approximation of ReLU

6. SOFTMAX: Converts logits to probabilities
   Use: Multi-class classification output
    """)
    
    def sigmoid(x): return 1 / (1 + math.exp(-x))
    def tanh(x): return math.tanh(x)
    def relu(x): return max(0, x)
    def leaky_relu(x): return x if x > 0 else 0.01 * x
    
    test_values = [-2, -1, 0, 1, 2]
    print("\nComparison:")
    print(f"{'x':>6} | {'Sigmoid':>8} | {'Tanh':>8} | {'ReLU':>8} | {'LeakyReLU':>10}")
    print("-" * 50)
    for x in test_values:
        print(f"{x:>6} | {sigmoid(x):>8.4f} | {tanh(x):>8.4f} | {relu(x):>8.4f} | {leaky_relu(x):>10.4f}")


# =============================================================================
# SECTION 8: INTERVIEW QUESTIONS
# =============================================================================

def interview_questions():
    """Common neural network interview questions"""
    print("\n" + "=" * 70)
    print("NEURAL NETWORK INTERVIEW QUESTIONS")
    print("=" * 70)
    
    questions = [
        ("What is the vanishing gradient problem?",
         "Gradients become very small in deep networks, making training difficult. Solutions: ReLU, residual connections, batch normalization."),
        ("Why do we need non-linear activation functions?",
         "Without them, stacking layers is equivalent to a single linear transformation."),
        ("What is dropout?",
         "Randomly sets neurons to zero during training to prevent overfitting."),
        ("What is batch normalization?",
         "Normalizes layer inputs for faster training and regularization."),
        ("Explain backpropagation.",
         "Calculates gradients using chain rule, flowing backward from output to input."),
    ]
    
    for i, (q, a) in enumerate(questions, 1):
        print(f"\nQ{i}: {q}")
        print(f"A: {a}")


# =============================================================================
# SECTION 9: COMMON PITFALLS
# =============================================================================

def common_pitfalls():
    """Common mistakes when working with neural networks"""
    print("\n" + "=" * 70)
    print("COMMON PITFALLS")
    print("=" * 70)
    
    pitfalls = [
        ("Not normalizing input data", "Normalize to mean=0, std=1"),
        ("Learning rate too high/low", "Start with 0.001, use schedulers"),
        ("Not using validation set", "Always split: train/val/test"),
        ("Training too long", "Use early stopping"),
        ("Network too complex", "Start simple, add complexity as needed"),
        ("Forgetting model.eval()", "Always set eval mode for inference"),
    ]
    
    for mistake, solution in pitfalls:
        print(f"\nMistake: {mistake}")
        print(f"Solution: {solution}")


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main():
    print("=" * 70)
    print("   MODULE 03: NEURAL NETWORKS - COMPREHENSIVE 360 DEGREE COVERAGE")
    print("=" * 70)
    
    explainer = NeuralNetworkExplainer()
    
    while True:
        print("\n" + "-" * 70)
        print("Choose a topic:")
        print()
        print("FUNDAMENTALS:")
        print("  1. What is a Neural Network? (4W+H)")
        print("  2. History and Evolution")
        print("  3. Types of Neural Networks")
        print()
        print("IMPLEMENTATIONS:")
        print("  4. Single Neuron Demo (AND gate)")
        print("  5. Multi-Layer Network Demo (XOR)")
        print("  6. PyTorch Implementation")
        print()
        print("DEEP DIVE:")
        print("  7. Activation Functions")
        print("  8. Interview Questions")
        print("  9. Common Pitfalls")
        print()
        print("GENAI FEATURES:")
        print("  10. AI-Powered Explanation (OpenAI)")
        print("  11. AI-Powered Explanation (Ollama - FREE)")
        print()
        print("  12. Run ALL Topics")
        print("  0. Exit")
        print("-" * 70)
        
        choice = input("\nEnter choice (0-12): ").strip()
        
        if choice == "0":
            print("\nHappy learning!")
            break
        elif choice == "1":
            explain_neural_networks_comprehensive()
        elif choice == "2":
            neural_network_history()
        elif choice == "3":
            neural_network_types()
        elif choice == "4":
            single_neuron_demo()
        elif choice == "5":
            xor_demo()
        elif choice == "6":
            pytorch_demo()
        elif choice == "7":
            activation_functions_demo()
        elif choice == "8":
            interview_questions()
        elif choice == "9":
            common_pitfalls()
        elif choice == "10":
            concept = input("Enter concept to explain: ").strip()
            print(explainer.explain_with_openai(concept))
        elif choice == "11":
            concept = input("Enter concept to explain: ").strip()
            print(explainer.explain_with_ollama(concept))
        elif choice == "12":
            explain_neural_networks_comprehensive()
            neural_network_history()
            neural_network_types()
            single_neuron_demo()
            xor_demo()
            pytorch_demo()
            activation_functions_demo()
            interview_questions()
            common_pitfalls()
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
