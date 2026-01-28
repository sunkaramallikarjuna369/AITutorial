import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Code, Box, Layers } from 'lucide-react'

const Module13Frameworks = () => {
  const [selectedFramework, setSelectedFramework] = useState<string | null>(null)

  const frameworks = [
    {
      name: 'PyTorch',
      company: 'Meta',
      icon: '🔥',
      color: 'bg-orange-500',
      description: 'Dynamic, Pythonic, research-friendly',
      pros: ['Easy debugging', 'Dynamic graphs', 'Great for research'],
      cons: ['Deployment can be complex', 'Smaller ecosystem'],
      useCase: 'Research, prototyping, NLP',
      code: `import torch
import torch.nn as nn

# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = Net()
print(model)`
    },
    {
      name: 'TensorFlow',
      company: 'Google',
      icon: '🧠',
      color: 'bg-orange-600',
      description: 'Production-ready, scalable, complete ecosystem',
      pros: ['Great for production', 'TensorBoard', 'Mobile/Edge support'],
      cons: ['Steeper learning curve', 'Less intuitive'],
      useCase: 'Production, mobile, large-scale',
      code: `import tensorflow as tf
from tensorflow import keras

# Define a simple neural network
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print(model.summary())`
    },
    {
      name: 'Hugging Face',
      company: 'Hugging Face',
      icon: '🤗',
      color: 'bg-yellow-500',
      description: 'Pre-trained models, easy NLP/CV',
      pros: ['Huge model hub', 'Easy to use', 'Great community'],
      cons: ['Abstraction hides details', 'Can be slow'],
      useCase: 'NLP, transformers, quick prototyping',
      code: `from transformers import pipeline

# Sentiment analysis in 2 lines!
classifier = pipeline("sentiment-analysis")
result = classifier("I love AI tutorials!")
print(result)

# Text generation
generator = pipeline("text-generation")
text = generator("AI is", max_length=50)
print(text)`
    },
    {
      name: 'LangChain',
      company: 'LangChain',
      icon: '🦜',
      color: 'bg-green-500',
      description: 'Build LLM applications easily',
      pros: ['Great for LLM apps', 'Many integrations', 'Active development'],
      cons: ['Rapidly changing API', 'Can be over-engineered'],
      useCase: 'Chatbots, RAG, agents',
      code: `from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Create a simple chain
llm = OpenAI(temperature=0.7)
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} simply."
)
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("quantum computing")
print(result)`
    },
    {
      name: 'JAX',
      company: 'Google',
      icon: '⚡',
      color: 'bg-purple-500',
      description: 'High-performance, functional, XLA',
      pros: ['Very fast', 'Auto-differentiation', 'TPU support'],
      cons: ['Functional paradigm', 'Smaller community'],
      useCase: 'Research, high-performance computing',
      code: `import jax
import jax.numpy as jnp
from jax import grad, jit

# JIT-compiled function
@jit
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']

# Automatic differentiation
def loss(params, x, y):
    pred = predict(params, x)
    return jnp.mean((pred - y) ** 2)

grad_fn = grad(loss)
print("Gradient computed!")`
    },
    {
      name: 'Keras',
      company: 'Google',
      icon: '🎯',
      color: 'bg-red-500',
      description: 'High-level, user-friendly API',
      pros: ['Very easy to learn', 'Clean API', 'Multi-backend'],
      cons: ['Less flexible', 'Abstraction limits control'],
      useCase: 'Beginners, quick prototyping',
      code: `from keras.models import Sequential
from keras.layers import Dense, Dropout

# Build model in minutes
model = Sequential([
    Dense(256, activation='relu', input_shape=(784,)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')
# model.fit(X_train, y_train, epochs=10)`
    },
  ]

  const pythonCode = `# AI Frameworks Comparison
# Choose the right tool for your task!

# ============================================
# PyTorch - Research & Flexibility
# ============================================

import torch
import torch.nn as nn
import torch.optim as optim

class PyTorchModel(nn.Module):
    """PyTorch: Dynamic, Pythonic, great for research"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

# Training loop
model = PyTorchModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# ============================================
# TensorFlow/Keras - Production Ready
# ============================================

import tensorflow as tf
from tensorflow import keras

# Keras: High-level, easy to use
model = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Training is one line!
model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Save for production
model.save('my_model.h5')

# ============================================
# Hugging Face Transformers
# ============================================

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline,
    Trainer,
    TrainingArguments
)

# Easy pipelines for common tasks
sentiment = pipeline("sentiment-analysis")
print(sentiment("I love this framework!"))

ner = pipeline("ner")
print(ner("John works at Google in California"))

generator = pipeline("text-generation", model="gpt2")
print(generator("AI will", max_length=50))

# Fine-tuning made easy
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)
trainer.train()

# ============================================
# LangChain - LLM Applications
# ============================================

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import load_tools, initialize_agent

# Simple chain
llm = OpenAI(temperature=0.7)
prompt = ChatPromptTemplate.from_template(
    "You are an expert in {topic}. Explain {concept} simply."
)
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(topic="AI", concept="neural networks")

# Conversation with memory
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)
conversation.predict(input="Hi, I'm learning AI")
conversation.predict(input="What should I learn first?")

# Agents with tools
tools = load_tools(["wikipedia", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description")
agent.run("What is the population of France times 2?")

# ============================================
# JAX - High Performance
# ============================================

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

# JIT compilation for speed
@jit
def neural_network(params, x):
    for w, b in params:
        x = jnp.tanh(jnp.dot(x, w) + b)
    return x

# Automatic differentiation
def loss_fn(params, x, y):
    preds = neural_network(params, x)
    return jnp.mean((preds - y) ** 2)

# Get gradients automatically
grad_fn = jit(grad(loss_fn))

# Vectorized operations with vmap
batched_predict = vmap(neural_network, in_axes=(None, 0))

# ============================================
# Scikit-learn - Classical ML
# ============================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create a pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# Grid search for best parameters
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")

# ============================================
# Framework Selection Guide
# ============================================

"""
Choose based on your needs:

| Task                    | Best Framework        |
|------------------------|----------------------|
| Research/Prototyping   | PyTorch              |
| Production/Mobile      | TensorFlow           |
| NLP/Transformers       | Hugging Face         |
| LLM Applications       | LangChain            |
| High Performance       | JAX                  |
| Classical ML           | Scikit-learn         |
| Quick Prototyping      | Keras                |
| Computer Vision        | PyTorch/TensorFlow   |
| Beginners              | Keras/Scikit-learn   |
"""`

  return (
    <div className="space-y-8">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-white mb-4">AI Frameworks</h1>
        <p className="text-xl text-purple-200">The tools that power modern AI development!</p>
      </div>

      <Tabs defaultValue="learn" className="w-full">
        <TabsList className="grid w-full grid-cols-3 bg-slate-800">
          <TabsTrigger value="learn">Learn</TabsTrigger>
          <TabsTrigger value="visualize">Visualize</TabsTrigger>
          <TabsTrigger value="code">Python Code</TabsTrigger>
        </TabsList>

        <TabsContent value="learn" className="space-y-6">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Box className="w-6 h-6 text-blue-400" />
                What are AI Frameworks?
              </CardTitle>
            </CardHeader>
            <CardContent className="text-slate-300 space-y-4">
              <p className="text-lg">
                <strong className="text-blue-400">AI Frameworks</strong> are software libraries that provide 
                tools, functions, and pre-built components to make building AI models easier. They handle 
                the complex math so you can focus on solving problems!
              </p>
              <div className="bg-slate-900/50 p-4 rounded-lg border border-blue-500/30">
                <h4 className="text-blue-400 font-semibold mb-2">Think of it like this:</h4>
                <p>Building AI without frameworks is like building a house with raw materials. Frameworks 
                are like having pre-made walls, doors, and windows - you can still customize everything, 
                but the hard work is already done!</p>
              </div>
            </CardContent>
          </Card>

          <div className="grid md:grid-cols-3 gap-4">
            {frameworks.slice(0, 6).map((fw, i) => (
              <Card key={i} className="bg-slate-800/50 border-slate-700 hover:border-slate-500 transition-all cursor-pointer"
                onClick={() => setSelectedFramework(selectedFramework === fw.name ? null : fw.name)}>
                <CardHeader className="pb-2">
                  <div className="flex items-center gap-3">
                    <div className={`w-12 h-12 ${fw.color} rounded-lg flex items-center justify-center text-2xl`}>
                      {fw.icon}
                    </div>
                    <div>
                      <CardTitle className="text-white text-lg">{fw.name}</CardTitle>
                      <CardDescription className="text-slate-500 text-xs">{fw.company}</CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-slate-400 text-sm mb-2">{fw.description}</p>
                  <Badge variant="outline" className="text-slate-500 text-xs">{fw.useCase}</Badge>
                </CardContent>
              </Card>
            ))}
          </div>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Framework Comparison</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-700">
                      <th className="text-left py-2 text-slate-400">Framework</th>
                      <th className="text-left py-2 text-slate-400">Best For</th>
                      <th className="text-left py-2 text-slate-400">Learning Curve</th>
                      <th className="text-left py-2 text-slate-400">Production</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { name: 'PyTorch', best: 'Research', curve: 'Medium', prod: '⭐⭐⭐' },
                      { name: 'TensorFlow', best: 'Production', curve: 'Hard', prod: '⭐⭐⭐⭐⭐' },
                      { name: 'Hugging Face', best: 'NLP', curve: 'Easy', prod: '⭐⭐⭐⭐' },
                      { name: 'LangChain', best: 'LLM Apps', curve: 'Easy', prod: '⭐⭐⭐' },
                      { name: 'Keras', best: 'Beginners', curve: 'Easy', prod: '⭐⭐⭐⭐' },
                    ].map((row, i) => (
                      <tr key={i} className="border-b border-slate-800">
                        <td className="py-2 text-white">{row.name}</td>
                        <td className="py-2 text-slate-400">{row.best}</td>
                        <td className="py-2 text-slate-400">{row.curve}</td>
                        <td className="py-2 text-yellow-400">{row.prod}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="visualize" className="space-y-6">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Layers className="w-5 h-5 text-blue-400" />
                Interactive Framework Explorer
              </CardTitle>
              <CardDescription className="text-slate-400">
                Click on a framework to see code examples!
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex flex-wrap gap-2">
                {frameworks.map((fw, i) => (
                  <Button
                    key={i}
                    variant={selectedFramework === fw.name ? 'default' : 'outline'}
                    className={selectedFramework === fw.name ? fw.color : ''}
                    onClick={() => setSelectedFramework(selectedFramework === fw.name ? null : fw.name)}
                  >
                    <span className="mr-2">{fw.icon}</span>
                    {fw.name}
                  </Button>
                ))}
              </div>

              {selectedFramework && (
                <div className="space-y-4 animate-in fade-in">
                  {frameworks.filter(fw => fw.name === selectedFramework).map((fw, i) => (
                    <div key={i}>
                      <div className="grid md:grid-cols-2 gap-4 mb-4">
                        <div className="bg-green-500/10 p-4 rounded-lg border border-green-500/30">
                          <h4 className="text-green-400 font-semibold mb-2">Pros</h4>
                          <ul className="space-y-1">
                            {fw.pros.map((pro, j) => (
                              <li key={j} className="text-slate-300 text-sm">+ {pro}</li>
                            ))}
                          </ul>
                        </div>
                        <div className="bg-red-500/10 p-4 rounded-lg border border-red-500/30">
                          <h4 className="text-red-400 font-semibold mb-2">Cons</h4>
                          <ul className="space-y-1">
                            {fw.cons.map((con, j) => (
                              <li key={j} className="text-slate-300 text-sm">- {con}</li>
                            ))}
                          </ul>
                        </div>
                      </div>
                      <div className="bg-slate-900 p-4 rounded-lg">
                        <h4 className="text-blue-400 font-semibold mb-2">Quick Example:</h4>
                        <pre className="text-green-400 text-sm overflow-x-auto">
                          <code>{fw.code}</code>
                        </pre>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Framework Ecosystem</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="relative bg-slate-900 rounded-lg p-6">
                <div className="grid grid-cols-3 gap-4">
                  <div className="text-center">
                    <h4 className="text-blue-400 font-semibold mb-3">Deep Learning</h4>
                    <div className="space-y-2">
                      <Badge className="bg-orange-500">PyTorch</Badge>
                      <Badge className="bg-orange-600">TensorFlow</Badge>
                      <Badge className="bg-purple-500">JAX</Badge>
                    </div>
                  </div>
                  <div className="text-center">
                    <h4 className="text-green-400 font-semibold mb-3">NLP & LLMs</h4>
                    <div className="space-y-2">
                      <Badge className="bg-yellow-500">Hugging Face</Badge>
                      <Badge className="bg-green-500">LangChain</Badge>
                      <Badge className="bg-cyan-500">LlamaIndex</Badge>
                    </div>
                  </div>
                  <div className="text-center">
                    <h4 className="text-purple-400 font-semibold mb-3">Classical ML</h4>
                    <div className="space-y-2">
                      <Badge className="bg-blue-500">Scikit-learn</Badge>
                      <Badge className="bg-indigo-500">XGBoost</Badge>
                      <Badge className="bg-pink-500">LightGBM</Badge>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Decision Flowchart</CardTitle>
              <CardDescription className="text-slate-400">
                Which framework should you use?
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {[
                  { q: 'Building an LLM application?', a: 'LangChain', color: 'bg-green-500' },
                  { q: 'Working with transformers/NLP?', a: 'Hugging Face', color: 'bg-yellow-500' },
                  { q: 'Need production deployment?', a: 'TensorFlow', color: 'bg-orange-600' },
                  { q: 'Doing research/prototyping?', a: 'PyTorch', color: 'bg-orange-500' },
                  { q: 'Just starting out?', a: 'Keras', color: 'bg-red-500' },
                  { q: 'Classical ML (not deep learning)?', a: 'Scikit-learn', color: 'bg-blue-500' },
                ].map((item, i) => (
                  <div key={i} className="flex items-center gap-3 p-3 bg-slate-900 rounded-lg">
                    <span className="text-slate-300 flex-1">{item.q}</span>
                    <span className="text-slate-500">→</span>
                    <Badge className={item.color}>{item.a}</Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="code" className="space-y-6">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Code className="w-5 h-5 text-blue-400" />
                Python Code: AI Frameworks
              </CardTitle>
              <CardDescription className="text-slate-400">
                Examples from all major frameworks!
              </CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="bg-slate-900 p-4 rounded-lg overflow-x-auto text-sm max-h-96 overflow-y-auto">
                <code className="text-green-400">{pythonCode}</code>
              </pre>
              <div className="mt-4 p-4 bg-blue-500/10 rounded-lg border border-blue-500/30">
                <h4 className="text-blue-400 font-semibold mb-2">Install All Frameworks:</h4>
                <code className="text-slate-300 text-sm bg-slate-800 px-2 py-1 rounded block">
                  pip install torch tensorflow transformers langchain scikit-learn jax
                </code>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default Module13Frameworks
