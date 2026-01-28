import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Bot, MessageSquare, Code, Scale } from 'lucide-react'

const Module09LLMs = () => {
  const [chatHistory, setChatHistory] = useState<{role: string, content: string}[]>([])
  const [userInput, setUserInput] = useState('')
  const [isTyping, setIsTyping] = useState(false)

  const simulateChat = () => {
    if (!userInput.trim()) return
    
    setChatHistory(prev => [...prev, { role: 'user', content: userInput }])
    setIsTyping(true)
    setUserInput('')
    
    setTimeout(() => {
      const responses = [
        "That's a great question! LLMs work by predicting the next word based on patterns learned from vast amounts of text.",
        "I understand what you're asking. Large Language Models use transformer architecture to process and generate text.",
        "Interesting! LLMs are trained on billions of words and learn to understand context and generate coherent responses.",
      ]
      setChatHistory(prev => [...prev, { 
        role: 'assistant', 
        content: responses[Math.floor(Math.random() * responses.length)]
      }])
      setIsTyping(false)
    }, 1500)
  }

  const pythonCode = `# Large Language Models (LLMs)
# The AI behind ChatGPT, Claude, and more!

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    GPT2LMHeadModel, GPT2Tokenizer,
    pipeline
)

# ============================================
# Understanding LLM Basics
# ============================================

"""
What makes an LLM "Large"?
- Billions of parameters (weights)
- Trained on massive text datasets
- Can understand and generate human-like text

Key concepts:
1. Tokenization: Breaking text into pieces
2. Context Window: How much text it can "see"
3. Temperature: Creativity vs consistency
4. Top-p/Top-k: Controlling word choices
"""

# ============================================
# Using GPT-2 (Smaller, Local Model)
# ============================================

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def generate_text(prompt, max_length=100, temperature=0.7):
    """
    Generate text continuation from a prompt
    
    temperature: 
    - Low (0.1-0.5): More focused, predictable
    - High (0.7-1.0): More creative, diverse
    """
    # Tokenize input
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate
    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode output
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated

# Example
prompt = "The future of artificial intelligence is"
result = generate_text(prompt)
print(result)

# ============================================
# Using Hugging Face Pipeline (Easy Way)
# ============================================

# Text generation pipeline
generator = pipeline('text-generation', model='gpt2')

def easy_generate(prompt, max_length=50):
    """Simple text generation with pipeline"""
    result = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7
    )
    return result[0]['generated_text']

# ============================================
# Chat with LLMs
# ============================================

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a chat-optimized model
chat_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def chat(user_input, chat_history_ids=None):
    """
    Have a conversation with the model
    """
    # Encode user input
    new_input_ids = chat_tokenizer.encode(
        user_input + chat_tokenizer.eos_token,
        return_tensors='pt'
    )
    
    # Append to chat history
    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids
    
    # Generate response
    chat_history_ids = chat_model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=chat_tokenizer.eos_token_id,
        temperature=0.7,
        top_p=0.9
    )
    
    # Decode response
    response = chat_tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )
    
    return response, chat_history_ids

# Example conversation
history = None
for user_msg in ["Hi, how are you?", "What's your favorite color?", "Tell me a joke"]:
    response, history = chat(user_msg, history)
    print(f"User: {user_msg}")
    print(f"Bot: {response}\\n")

# ============================================
# Using OpenAI API
# ============================================

import openai

openai.api_key = "your-api-key"

def chat_with_gpt(messages, model="gpt-3.5-turbo"):
    """
    Chat with OpenAI's GPT models
    
    messages format:
    [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]
    """
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content

# Example
messages = [
    {"role": "system", "content": "You are a helpful AI tutor."},
    {"role": "user", "content": "Explain machine learning in simple terms."}
]
response = chat_with_gpt(messages)
print(response)

# ============================================
# Using LangChain for LLM Applications
# ============================================

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Create an LLM instance
llm = OpenAI(temperature=0.7)

# Create a prompt template
template = """
You are an expert {topic} teacher.
Explain {concept} to a {audience}.
Keep it simple and use examples.
"""

prompt = PromptTemplate(
    input_variables=["topic", "concept", "audience"],
    template=template
)

# Create a chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain
result = chain.run(
    topic="AI",
    concept="neural networks",
    audience="10-year-old"
)
print(result)

# ============================================
# Token Counting and Context Management
# ============================================

import tiktoken

def count_tokens(text, model="gpt-3.5-turbo"):
    """
    Count tokens in text
    Important for staying within context limits!
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

# Example
text = "Hello, how are you doing today?"
num_tokens = count_tokens(text)
print(f"'{text}' has {num_tokens} tokens")

# Context window sizes:
# GPT-3.5: 4,096 or 16,384 tokens
# GPT-4: 8,192 or 32,768 or 128,000 tokens
# Claude: 100,000+ tokens

# ============================================
# Streaming Responses
# ============================================

def stream_response(prompt):
    """
    Stream response token by token
    Better user experience for long responses!
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in response:
        if chunk.choices[0].delta.get("content"):
            print(chunk.choices[0].delta.content, end="", flush=True)

# ============================================
# Local LLMs with Ollama
# ============================================

import requests

def chat_with_ollama(prompt, model="llama2"):
    """
    Use local LLMs with Ollama
    No API key needed, runs on your computer!
    """
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

# Example (requires Ollama running locally)
# response = chat_with_ollama("What is the capital of France?")`

  const llmModels = [
    { name: 'GPT-4', company: 'OpenAI', params: '1.7T', context: '128K', color: 'bg-green-500' },
    { name: 'Claude 3', company: 'Anthropic', params: '?', context: '200K', color: 'bg-purple-500' },
    { name: 'Gemini', company: 'Google', params: '?', context: '1M', color: 'bg-blue-500' },
    { name: 'LLaMA 2', company: 'Meta', params: '70B', context: '4K', color: 'bg-orange-500' },
    { name: 'Mistral', company: 'Mistral AI', params: '7B', context: '32K', color: 'bg-cyan-500' },
    { name: 'Falcon', company: 'TII', params: '180B', context: '2K', color: 'bg-red-500' },
  ]

  return (
    <div className="space-y-8">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-white mb-4">Large Language Models</h1>
        <p className="text-xl text-purple-200">The powerful AI behind ChatGPT and modern chatbots!</p>
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
                <Bot className="w-6 h-6 text-cyan-400" />
                What are Large Language Models?
              </CardTitle>
            </CardHeader>
            <CardContent className="text-slate-300 space-y-4">
              <p className="text-lg">
                <strong className="text-cyan-400">Large Language Models (LLMs)</strong> are AI systems trained 
                on massive amounts of text data. They can understand context, answer questions, write code, 
                and have conversations that feel remarkably human!
              </p>
              <div className="bg-slate-900/50 p-4 rounded-lg border border-cyan-500/30">
                <h4 className="text-cyan-400 font-semibold mb-2">Think of it like this:</h4>
                <p>Imagine someone who has read every book, article, and website on the internet. They've 
                learned patterns in language so well that they can continue any sentence, answer questions, 
                and even write in different styles. That's what an LLM does!</p>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Scale className="w-5 h-5 text-yellow-400" />
                What Makes LLMs "Large"?
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-4">
                <div className="bg-slate-900 p-4 rounded-lg text-center">
                  <p className="text-4xl font-bold text-cyan-400">175B+</p>
                  <p className="text-slate-400 text-sm">Parameters (GPT-3)</p>
                  <p className="text-xs text-slate-500 mt-1">Like brain connections</p>
                </div>
                <div className="bg-slate-900 p-4 rounded-lg text-center">
                  <p className="text-4xl font-bold text-green-400">45TB</p>
                  <p className="text-slate-400 text-sm">Training Data</p>
                  <p className="text-xs text-slate-500 mt-1">Books, websites, code</p>
                </div>
                <div className="bg-slate-900 p-4 rounded-lg text-center">
                  <p className="text-4xl font-bold text-purple-400">$100M+</p>
                  <p className="text-slate-400 text-sm">Training Cost</p>
                  <p className="text-xs text-slate-500 mt-1">Compute resources</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Key LLM Concepts</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {[
                { term: 'Tokens', desc: 'Words or word pieces the model processes. "Hello" = 1 token, "unbelievable" = 3 tokens', color: 'blue' },
                { term: 'Context Window', desc: 'How much text the model can "see" at once. GPT-4 can see 128K tokens!', color: 'green' },
                { term: 'Temperature', desc: 'Controls creativity. Low = predictable, High = creative and random', color: 'orange' },
                { term: 'Prompt', desc: 'The input you give to the model. Better prompts = better outputs!', color: 'purple' },
              ].map((item, i) => (
                <div key={i} className={`flex items-start gap-3 p-3 bg-${item.color}-500/10 rounded-lg border border-${item.color}-500/30`}>
                  <Badge className={`bg-${item.color}-500`}>{item.term}</Badge>
                  <p className="text-slate-300 text-sm">{item.desc}</p>
                </div>
              ))}
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Popular LLM Models</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-3">
                {llmModels.map((model, i) => (
                  <div key={i} className="bg-slate-900 p-3 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <div className={`w-8 h-8 ${model.color} rounded flex items-center justify-center`}>
                        <Bot className="w-4 h-4 text-white" />
                      </div>
                      <div>
                        <p className="text-white font-semibold text-sm">{model.name}</p>
                        <p className="text-slate-500 text-xs">{model.company}</p>
                      </div>
                    </div>
                    <div className="flex gap-2 text-xs">
                      <Badge variant="outline" className="text-slate-400">{model.params} params</Badge>
                      <Badge variant="outline" className="text-slate-400">{model.context} context</Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="visualize" className="space-y-6">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <MessageSquare className="w-5 h-5 text-cyan-400" />
                Chat with a Simulated LLM
              </CardTitle>
              <CardDescription className="text-slate-400">
                Experience how LLMs respond to your questions!
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="bg-slate-900 rounded-lg p-4 h-64 overflow-y-auto space-y-3">
                {chatHistory.length === 0 && (
                  <p className="text-slate-500 text-center">Start a conversation!</p>
                )}
                {chatHistory.map((msg, i) => (
                  <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-xs p-3 rounded-lg ${
                      msg.role === 'user' 
                        ? 'bg-cyan-500 text-white' 
                        : 'bg-slate-700 text-slate-200'
                    }`}>
                      {msg.content}
                    </div>
                  </div>
                ))}
                {isTyping && (
                  <div className="flex justify-start">
                    <div className="bg-slate-700 text-slate-200 p-3 rounded-lg">
                      <span className="animate-pulse">Thinking...</span>
                    </div>
                  </div>
                )}
              </div>
              
              <div className="flex gap-2">
                <input
                  type="text"
                  value={userInput}
                  onChange={(e) => setUserInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && simulateChat()}
                  placeholder="Ask about LLMs..."
                  className="flex-1 px-4 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
                />
                <Button onClick={simulateChat} className="bg-cyan-600 hover:bg-cyan-700">
                  Send
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">How LLMs Generate Text</CardTitle>
              <CardDescription className="text-slate-400">
                Next-word prediction in action
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="bg-slate-900 p-4 rounded-lg">
                <p className="text-slate-300 mb-4">
                  Input: <span className="text-cyan-400">"The cat sat on the"</span>
                </p>
                <div className="space-y-2">
                  <p className="text-slate-400 text-sm">LLM predicts next word probabilities:</p>
                  {[
                    { word: 'mat', prob: 35 },
                    { word: 'floor', prob: 25 },
                    { word: 'couch', prob: 20 },
                    { word: 'bed', prob: 12 },
                    { word: 'table', prob: 8 },
                  ].map((item, i) => (
                    <div key={i} className="flex items-center gap-3">
                      <span className="text-white w-16">{item.word}</span>
                      <div className="flex-1 bg-slate-700 rounded-full h-4 overflow-hidden">
                        <div 
                          className="bg-gradient-to-r from-cyan-500 to-blue-500 h-full"
                          style={{ width: `${item.prob}%` }}
                        />
                      </div>
                      <span className="text-slate-400 text-sm w-12">{item.prob}%</span>
                    </div>
                  ))}
                </div>
                <p className="text-slate-500 text-sm mt-4">
                  The model samples from these probabilities. Temperature controls how random the choice is!
                </p>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">LLM Architecture Overview</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="relative bg-slate-900 rounded-lg p-6">
                <div className="flex flex-col items-center space-y-4">
                  <div className="bg-blue-500 text-white px-6 py-3 rounded-lg w-full max-w-md text-center">
                    <p className="font-semibold">Input Text</p>
                    <p className="text-xs opacity-80">"What is AI?"</p>
                  </div>
                  <div className="text-slate-500">↓</div>
                  <div className="bg-purple-500 text-white px-6 py-3 rounded-lg w-full max-w-md text-center">
                    <p className="font-semibold">Tokenizer</p>
                    <p className="text-xs opacity-80">["What", "is", "AI", "?"]</p>
                  </div>
                  <div className="text-slate-500">↓</div>
                  <div className="bg-indigo-500 text-white px-6 py-3 rounded-lg w-full max-w-md text-center">
                    <p className="font-semibold">Transformer Layers (x96)</p>
                    <p className="text-xs opacity-80">Self-attention + Feed-forward</p>
                  </div>
                  <div className="text-slate-500">↓</div>
                  <div className="bg-green-500 text-white px-6 py-3 rounded-lg w-full max-w-md text-center">
                    <p className="font-semibold">Output Probabilities</p>
                    <p className="text-xs opacity-80">Next token prediction</p>
                  </div>
                  <div className="text-slate-500">↓</div>
                  <div className="bg-cyan-500 text-white px-6 py-3 rounded-lg w-full max-w-md text-center">
                    <p className="font-semibold">Generated Response</p>
                    <p className="text-xs opacity-80">"AI is artificial intelligence..."</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="code" className="space-y-6">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Code className="w-5 h-5 text-cyan-400" />
                Python Code: Working with LLMs
              </CardTitle>
              <CardDescription className="text-slate-400">
                From local models to API calls!
              </CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="bg-slate-900 p-4 rounded-lg overflow-x-auto text-sm max-h-96 overflow-y-auto">
                <code className="text-green-400">{pythonCode}</code>
              </pre>
              <div className="mt-4 p-4 bg-cyan-500/10 rounded-lg border border-cyan-500/30">
                <h4 className="text-cyan-400 font-semibold mb-2">Required Libraries:</h4>
                <code className="text-slate-300 text-sm bg-slate-800 px-2 py-1 rounded">
                  pip install transformers torch openai langchain tiktoken
                </code>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default Module09LLMs
