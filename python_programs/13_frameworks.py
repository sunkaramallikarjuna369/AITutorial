"""
Module 13: AI Frameworks
========================
Tools and libraries for building AI applications!
"""

def explain_frameworks():
    print("=" * 60)
    print("AI Frameworks Overview")
    print("=" * 60)
    print("""
Major AI/ML Frameworks:

1. PyTorch (Meta)
   - Most popular for research
   - Dynamic computation graphs
   - Pythonic and intuitive

2. TensorFlow (Google)
   - Production-ready
   - TensorFlow Lite for mobile
   - TensorBoard for visualization

3. JAX (Google)
   - NumPy on steroids
   - Automatic differentiation
   - JIT compilation

4. Hugging Face
   - Pre-trained models hub
   - Transformers library
   - Easy fine-tuning

5. LangChain
   - LLM application framework
   - Chains and agents
   - RAG pipelines

6. LlamaIndex
   - Data framework for LLMs
   - Document indexing
   - Query engines
    """)

code_examples = '''
"""
AI Framework Examples
=====================
"""

# ============================================
# 1. PyTorch Basics
# ============================================

import torch
import torch.nn as nn
import torch.optim as optim

# Create a simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


# ============================================
# 2. TensorFlow/Keras
# ============================================

import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# ============================================
# 3. Hugging Face Transformers
# ============================================

from transformers import pipeline

# Easy-to-use pipelines
classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")

generator = pipeline("text-generation", model="gpt2")
text = generator("Once upon a time", max_length=50)

qa = pipeline("question-answering")
answer = qa(question="What is AI?", context="AI is artificial intelligence...")


# ============================================
# 4. LangChain
# ============================================

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0.7)

template = "Write a {adjective} poem about {topic}"
prompt = PromptTemplate(
    input_variables=["adjective", "topic"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(adjective="funny", topic="programming")


# ============================================
# 5. Scikit-learn
# ============================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Train a model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)


# ============================================
# 6. JAX
# ============================================

import jax
import jax.numpy as jnp

@jax.jit
def neural_network(params, x):
    for w, b in params:
        x = jnp.tanh(jnp.dot(x, w) + b)
    return x

# Automatic differentiation
grad_fn = jax.grad(loss_fn)


# ============================================
# 7. FastAI
# ============================================

from fastai.vision.all import *

# Train image classifier in 3 lines
dls = ImageDataLoaders.from_folder(path)
learn = vision_learner(dls, resnet34, metrics=accuracy)
learn.fine_tune(3)


# ============================================
# 8. Weights & Biases (Experiment Tracking)
# ============================================

import wandb

wandb.init(project="my-project")
wandb.config = {"learning_rate": 0.001, "epochs": 10}

for epoch in range(10):
    # Training...
    wandb.log({"loss": loss, "accuracy": acc})


print("Framework examples loaded!")
'''

def main():
    explain_frameworks()
    print("\nPython Code Examples:")
    print(code_examples)

if __name__ == "__main__":
    main()
