import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Calculator, TrendingDown, BarChart3, Percent, Target, ArrowRight } from 'lucide-react'

const Module00MathForAI = () => {
  const [selectedTopic, setSelectedTopic] = useState('vectors')
  const [gradientStep, setGradientStep] = useState(0)
  const [dotA, setDotA] = useState([3, 4])
  const [dotB, setDotB] = useState([4, 3])

  // Gradient descent simulation
  const gradientSteps = [
    { x: 5, y: 5, loss: 50 },
    { x: 4, y: 4, loss: 32 },
    { x: 3.2, y: 3.2, loss: 20.48 },
    { x: 2.56, y: 2.56, loss: 13.11 },
    { x: 2.05, y: 2.05, loss: 8.40 },
    { x: 1.64, y: 1.64, loss: 5.38 },
    { x: 1.31, y: 1.31, loss: 3.43 },
    { x: 1.05, y: 1.05, loss: 2.21 },
    { x: 0.84, y: 0.84, loss: 1.41 },
    { x: 0.67, y: 0.67, loss: 0.90 },
  ]

  const runGradientDescent = () => {
    if (gradientStep < gradientSteps.length - 1) {
      setGradientStep(prev => prev + 1)
    }
  }

  const resetGradientDescent = () => {
    setGradientStep(0)
  }

  // Calculate dot product
  const dotProduct = dotA[0] * dotB[0] + dotA[1] * dotB[1]
  const magA = Math.sqrt(dotA[0] ** 2 + dotA[1] ** 2)
  const magB = Math.sqrt(dotB[0] ** 2 + dotB[1] ** 2)
  const cosineSim = dotProduct / (magA * magB)

  const topics = [
    { id: 'vectors', name: 'Vectors', icon: ArrowRight },
    { id: 'matrices', name: 'Matrices', icon: BarChart3 },
    { id: 'derivatives', name: 'Derivatives', icon: TrendingDown },
    { id: 'probability', name: 'Probability', icon: Percent },
    { id: 'optimization', name: 'Optimization', icon: Target },
  ]

  const pythonCode = `"""
=============================================================================
MATHEMATICS FOR AI - Complete Python Examples
=============================================================================
"""

import math
import numpy as np

# =============================================================================
# 1. VECTORS - Building Blocks of AI
# =============================================================================

# WHAT: Ordered list of numbers
# WHY: All data becomes vectors in AI
# WHEN: Always - input data, embeddings, features
# WHERE: Word embeddings, image pixels, neural network layers
# HOW: As arrays/lists

# Vector operations
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Addition
print(f"v1 + v2 = {v1 + v2}")

# Dot product (similarity measure)
dot = np.dot(v1, v2)
print(f"v1 . v2 = {dot}")

# Cosine similarity (used in NLP, recommendations)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"Cosine similarity: {cosine_similarity(v1, v2):.4f}")


# =============================================================================
# 2. MATRICES - Heart of Neural Networks
# =============================================================================

# WHAT: 2D array of numbers
# WHY: Neural network layers ARE matrix operations
# WHEN: Every forward/backward pass
# WHERE: Weights, attention scores, convolutions
# HOW: Matrix multiplication

# Neural network layer as matrix multiplication
input_vector = np.array([1, 2, 3])
weights = np.array([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6]
])
bias = np.array([0.1, 0.2])

# Forward pass: output = input @ weights + bias
output = np.dot(input_vector, weights) + bias
print(f"Layer output: {output}")


# =============================================================================
# 3. DERIVATIVES & GRADIENTS - How AI Learns
# =============================================================================

# WHAT: Rate of change of a function
# WHY: Tells us how to adjust weights to reduce error
# WHEN: Every training step (backpropagation)
# WHERE: All neural network training
# HOW: Automatic differentiation (PyTorch/TensorFlow)

def f(x):
    return x ** 2

def derivative(f, x, h=0.0001):
    return (f(x + h) - f(x)) / h

x = 3
print(f"f(x) = x^2")
print(f"f({x}) = {f(x)}")
print(f"f'({x}) = {derivative(f, x):.4f}")  # Should be 2*x = 6


# =============================================================================
# 4. GRADIENT DESCENT - The Learning Algorithm
# =============================================================================

# WHAT: Algorithm to find minimum by following gradient downhill
# WHY: This IS how neural networks learn
# WHEN: Every training iteration
# WHERE: All ML training
# HOW: weight = weight - learning_rate * gradient

def gradient_descent(start, learning_rate=0.1, iterations=20):
    x = start
    history = [x]
    
    for _ in range(iterations):
        gradient = 2 * x  # Derivative of x^2
        x = x - learning_rate * gradient
        history.append(x)
    
    return history

# Find minimum of f(x) = x^2 (minimum at x=0)
history = gradient_descent(start=5)
print(f"Gradient descent: {history[:5]}... -> {history[-1]:.6f}")


# =============================================================================
# 5. PROBABILITY & SOFTMAX
# =============================================================================

# WHAT: Converting scores to probabilities
# WHY: AI outputs probabilities, not certainties
# WHEN: Classification outputs
# WHERE: Final layer of classifiers
# HOW: Softmax function

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
    return exp_logits / exp_logits.sum()

logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(f"Logits: {logits}")
print(f"Probabilities: {probs}")
print(f"Sum: {probs.sum():.4f}")  # Always 1.0


# =============================================================================
# 6. LOSS FUNCTIONS
# =============================================================================

# Mean Squared Error (Regression)
def mse_loss(predicted, actual):
    return np.mean((predicted - actual) ** 2)

# Cross-Entropy Loss (Classification)
def cross_entropy_loss(predicted_probs, true_labels):
    return -np.sum(true_labels * np.log(predicted_probs + 1e-10))

# Example
pred = np.array([0.7, 0.2, 0.1])
true = np.array([1, 0, 0])  # True class is 0
print(f"Cross-entropy loss: {cross_entropy_loss(pred, true):.4f}")


# =============================================================================
# 7. NORMALIZATION
# =============================================================================

# WHAT: Scaling data to standard range
# WHY: Faster training, prevents large values dominating
# WHEN: Data preprocessing
# WHERE: Before neural networks
# HOW: Z-score or min-max

data = np.array([10, 20, 30, 40, 50])

# Z-score normalization
mean, std = data.mean(), data.std()
z_normalized = (data - mean) / std
print(f"Z-score normalized: {z_normalized}")

# Min-max normalization
min_val, max_val = data.min(), data.max()
minmax_normalized = (data - min_val) / (max_val - min_val)
print(f"Min-max normalized: {minmax_normalized}")


print("\\n" + "="*50)
print("Math for AI - All concepts demonstrated!")
print("="*50)
`

  return (
    <div className="space-y-8">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-white mb-4">Mathematics for AI</h1>
        <p className="text-xl text-purple-200">The essential math behind every AI model - explained with 4W+H!</p>
      </div>

      <Tabs defaultValue="learn" className="w-full">
        <TabsList className="grid w-full grid-cols-3 bg-slate-800">
          <TabsTrigger value="learn">Learn (4W+H)</TabsTrigger>
          <TabsTrigger value="visualize">Visualize</TabsTrigger>
          <TabsTrigger value="code">Python Code</TabsTrigger>
        </TabsList>

        <TabsContent value="learn" className="space-y-6">
          {/* Topic selector */}
          <div className="flex flex-wrap gap-2 justify-center">
            {topics.map((topic) => (
              <Button
                key={topic.id}
                onClick={() => setSelectedTopic(topic.id)}
                variant={selectedTopic === topic.id ? "default" : "outline"}
                className={selectedTopic === topic.id ? "bg-purple-600" : ""}
              >
                <topic.icon className="w-4 h-4 mr-2" />
                {topic.name}
              </Button>
            ))}
          </div>

          {/* Vectors */}
          {selectedTopic === 'vectors' && (
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <ArrowRight className="w-6 h-6 text-blue-400" />
                  Vectors - Building Blocks of AI
                </CardTitle>
              </CardHeader>
              <CardContent className="text-slate-300 space-y-4">
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-blue-500/10 p-4 rounded-lg border border-blue-500/30">
                    <h4 className="text-blue-400 font-bold mb-2">WHAT?</h4>
                    <p>An ordered list of numbers representing a point or direction in space.</p>
                    <code className="block mt-2 text-sm bg-slate-900 p-2 rounded">v = [3, 4, 5]</code>
                  </div>
                  <div className="bg-green-500/10 p-4 rounded-lg border border-green-500/30">
                    <h4 className="text-green-400 font-bold mb-2">WHY?</h4>
                    <p>ALL data in AI becomes vectors! Images, text, audio - everything is converted to numbers.</p>
                  </div>
                  <div className="bg-yellow-500/10 p-4 rounded-lg border border-yellow-500/30">
                    <h4 className="text-yellow-400 font-bold mb-2">WHEN?</h4>
                    <p>Always! Every piece of data in ML is represented as a vector.</p>
                  </div>
                  <div className="bg-purple-500/10 p-4 rounded-lg border border-purple-500/30">
                    <h4 className="text-purple-400 font-bold mb-2">WHERE?</h4>
                    <p>Word embeddings, image pixels, neural network layers, attention mechanisms.</p>
                  </div>
                </div>
                <div className="bg-orange-500/10 p-4 rounded-lg border border-orange-500/30">
                  <h4 className="text-orange-400 font-bold mb-2">HOW?</h4>
                  <p>Through vector operations: addition, dot product, cosine similarity.</p>
                  <p className="mt-2 text-sm">Dot Product measures similarity: high = similar, low = different</p>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Matrices */}
          {selectedTopic === 'matrices' && (
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <BarChart3 className="w-6 h-6 text-green-400" />
                  Matrices - Heart of Neural Networks
                </CardTitle>
              </CardHeader>
              <CardContent className="text-slate-300 space-y-4">
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-blue-500/10 p-4 rounded-lg border border-blue-500/30">
                    <h4 className="text-blue-400 font-bold mb-2">WHAT?</h4>
                    <p>A 2D array of numbers arranged in rows and columns.</p>
                    <pre className="mt-2 text-sm bg-slate-900 p-2 rounded">
{`W = | 0.1  0.2 |
    | 0.3  0.4 |`}
                    </pre>
                  </div>
                  <div className="bg-green-500/10 p-4 rounded-lg border border-green-500/30">
                    <h4 className="text-green-400 font-bold mb-2">WHY?</h4>
                    <p>Neural network layers ARE matrix operations! Every layer multiplies input by weights matrix.</p>
                  </div>
                  <div className="bg-yellow-500/10 p-4 rounded-lg border border-yellow-500/30">
                    <h4 className="text-yellow-400 font-bold mb-2">WHEN?</h4>
                    <p>Every forward pass, every backward pass, every training step.</p>
                  </div>
                  <div className="bg-purple-500/10 p-4 rounded-lg border border-purple-500/30">
                    <h4 className="text-purple-400 font-bold mb-2">WHERE?</h4>
                    <p>Weight matrices, attention scores, convolution kernels, transformers.</p>
                  </div>
                </div>
                <div className="bg-orange-500/10 p-4 rounded-lg border border-orange-500/30">
                  <h4 className="text-orange-400 font-bold mb-2">HOW?</h4>
                  <code className="block text-sm">output = input @ weights + bias</code>
                  <p className="mt-2 text-sm">This simple equation IS a neural network layer!</p>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Derivatives */}
          {selectedTopic === 'derivatives' && (
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <TrendingDown className="w-6 h-6 text-red-400" />
                  Derivatives & Gradients - How AI Learns
                </CardTitle>
              </CardHeader>
              <CardContent className="text-slate-300 space-y-4">
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-blue-500/10 p-4 rounded-lg border border-blue-500/30">
                    <h4 className="text-blue-400 font-bold mb-2">WHAT?</h4>
                    <p>Derivative measures how much a function changes when input changes.</p>
                    <code className="block mt-2 text-sm bg-slate-900 p-2 rounded">f(x) = x² → f'(x) = 2x</code>
                  </div>
                  <div className="bg-green-500/10 p-4 rounded-lg border border-green-500/30">
                    <h4 className="text-green-400 font-bold mb-2">WHY?</h4>
                    <p>Derivatives tell us HOW TO IMPROVE! They show which direction reduces error.</p>
                  </div>
                  <div className="bg-yellow-500/10 p-4 rounded-lg border border-yellow-500/30">
                    <h4 className="text-yellow-400 font-bold mb-2">WHEN?</h4>
                    <p>Every training step! Backpropagation computes derivatives for all weights.</p>
                  </div>
                  <div className="bg-purple-500/10 p-4 rounded-lg border border-purple-500/30">
                    <h4 className="text-purple-400 font-bold mb-2">WHERE?</h4>
                    <p>loss.backward() in PyTorch, tape.gradient() in TensorFlow.</p>
                  </div>
                </div>
                <div className="bg-orange-500/10 p-4 rounded-lg border border-orange-500/30">
                  <h4 className="text-orange-400 font-bold mb-2">HOW? (Chain Rule)</h4>
                  <p>For composed functions, multiply derivatives through each layer.</p>
                  <code className="block mt-2 text-sm bg-slate-900 p-2 rounded">dL/dW = dL/dOutput × dOutput/dW</code>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Probability */}
          {selectedTopic === 'probability' && (
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Percent className="w-6 h-6 text-yellow-400" />
                  Probability - Handling Uncertainty
                </CardTitle>
              </CardHeader>
              <CardContent className="text-slate-300 space-y-4">
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-blue-500/10 p-4 rounded-lg border border-blue-500/30">
                    <h4 className="text-blue-400 font-bold mb-2">WHAT?</h4>
                    <p>Mathematics of uncertainty - measuring how likely events are.</p>
                    <code className="block mt-2 text-sm bg-slate-900 p-2 rounded">P(cat) = 0.85 (85% confident)</code>
                  </div>
                  <div className="bg-green-500/10 p-4 rounded-lg border border-green-500/30">
                    <h4 className="text-green-400 font-bold mb-2">WHY?</h4>
                    <p>AI models output PROBABILITIES, not certainties! We need to quantify confidence.</p>
                  </div>
                  <div className="bg-yellow-500/10 p-4 rounded-lg border border-yellow-500/30">
                    <h4 className="text-yellow-400 font-bold mb-2">WHEN?</h4>
                    <p>Classification outputs, language model next-token prediction, generative models.</p>
                  </div>
                  <div className="bg-purple-500/10 p-4 rounded-lg border border-purple-500/30">
                    <h4 className="text-purple-400 font-bold mb-2">WHERE?</h4>
                    <p>Softmax layer, cross-entropy loss, Bayesian methods, diffusion models.</p>
                  </div>
                </div>
                <div className="bg-orange-500/10 p-4 rounded-lg border border-orange-500/30">
                  <h4 className="text-orange-400 font-bold mb-2">HOW? (Softmax)</h4>
                  <p>Converts raw scores to probabilities that sum to 1.</p>
                  <code className="block mt-2 text-sm bg-slate-900 p-2 rounded">P(class_i) = exp(score_i) / sum(exp(scores))</code>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Optimization */}
          {selectedTopic === 'optimization' && (
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Target className="w-6 h-6 text-pink-400" />
                  Optimization - Finding the Best Solution
                </CardTitle>
              </CardHeader>
              <CardContent className="text-slate-300 space-y-4">
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-blue-500/10 p-4 rounded-lg border border-blue-500/30">
                    <h4 className="text-blue-400 font-bold mb-2">WHAT?</h4>
                    <p>Finding parameters that minimize (or maximize) a function.</p>
                    <code className="block mt-2 text-sm bg-slate-900 p-2 rounded">Find weights that minimize loss</code>
                  </div>
                  <div className="bg-green-500/10 p-4 rounded-lg border border-green-500/30">
                    <h4 className="text-green-400 font-bold mb-2">WHY?</h4>
                    <p>Training = finding weights that minimize prediction error!</p>
                  </div>
                  <div className="bg-yellow-500/10 p-4 rounded-lg border border-yellow-500/30">
                    <h4 className="text-yellow-400 font-bold mb-2">WHEN?</h4>
                    <p>Every training iteration - millions of optimization steps.</p>
                  </div>
                  <div className="bg-purple-500/10 p-4 rounded-lg border border-purple-500/30">
                    <h4 className="text-purple-400 font-bold mb-2">WHERE?</h4>
                    <p>SGD, Adam, AdamW optimizers in PyTorch/TensorFlow.</p>
                  </div>
                </div>
                <div className="bg-orange-500/10 p-4 rounded-lg border border-orange-500/30">
                  <h4 className="text-orange-400 font-bold mb-2">HOW? (Gradient Descent)</h4>
                  <code className="block text-sm bg-slate-900 p-2 rounded">weight = weight - learning_rate × gradient</code>
                  <p className="mt-2 text-sm">Move opposite to gradient direction to reduce loss!</p>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="visualize" className="space-y-6">
          {/* Vector Visualization */}
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Vector Dot Product & Similarity</CardTitle>
              <CardDescription className="text-slate-400">Adjust vectors to see how dot product measures similarity</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <label className="text-slate-300 text-sm">Vector A: [{dotA[0]}, {dotA[1]}]</label>
                    <div className="flex gap-2 mt-1">
                      <input
                        type="range"
                        min="-5"
                        max="5"
                        value={dotA[0]}
                        onChange={(e) => setDotA([parseInt(e.target.value), dotA[1]])}
                        className="flex-1"
                      />
                      <input
                        type="range"
                        min="-5"
                        max="5"
                        value={dotA[1]}
                        onChange={(e) => setDotA([dotA[0], parseInt(e.target.value)])}
                        className="flex-1"
                      />
                    </div>
                  </div>
                  <div>
                    <label className="text-slate-300 text-sm">Vector B: [{dotB[0]}, {dotB[1]}]</label>
                    <div className="flex gap-2 mt-1">
                      <input
                        type="range"
                        min="-5"
                        max="5"
                        value={dotB[0]}
                        onChange={(e) => setDotB([parseInt(e.target.value), dotB[1]])}
                        className="flex-1"
                      />
                      <input
                        type="range"
                        min="-5"
                        max="5"
                        value={dotB[1]}
                        onChange={(e) => setDotB([dotB[0], parseInt(e.target.value)])}
                        className="flex-1"
                      />
                    </div>
                  </div>
                  <div className="bg-slate-900 p-4 rounded-lg">
                    <p className="text-slate-300">Dot Product: <span className="text-purple-400 font-bold">{dotProduct}</span></p>
                    <p className="text-slate-300">Cosine Similarity: <span className="text-green-400 font-bold">{cosineSim.toFixed(4)}</span></p>
                    <p className="text-slate-400 text-sm mt-2">
                      {cosineSim > 0.8 ? "Very similar! (same direction)" :
                       cosineSim > 0.5 ? "Somewhat similar" :
                       cosineSim > 0 ? "Slightly related" :
                       cosineSim > -0.5 ? "Different directions" :
                       "Opposite directions!"}
                    </p>
                  </div>
                </div>
                <div className="bg-slate-900 rounded-lg p-4">
                  <svg viewBox="-60 -60 120 120" className="w-full h-48">
                    {/* Grid */}
                    <line x1="-50" y1="0" x2="50" y2="0" stroke="#475569" strokeWidth="0.5" />
                    <line x1="0" y1="-50" x2="0" y2="50" stroke="#475569" strokeWidth="0.5" />
                    
                    {/* Vector A */}
                    <line x1="0" y1="0" x2={dotA[0] * 8} y2={-dotA[1] * 8} stroke="#3b82f6" strokeWidth="2" />
                    <circle cx={dotA[0] * 8} cy={-dotA[1] * 8} r="3" fill="#3b82f6" />
                    <text x={dotA[0] * 8 + 5} y={-dotA[1] * 8} fill="#3b82f6" fontSize="10">A</text>
                    
                    {/* Vector B */}
                    <line x1="0" y1="0" x2={dotB[0] * 8} y2={-dotB[1] * 8} stroke="#22c55e" strokeWidth="2" />
                    <circle cx={dotB[0] * 8} cy={-dotB[1] * 8} r="3" fill="#22c55e" />
                    <text x={dotB[0] * 8 + 5} y={-dotB[1] * 8} fill="#22c55e" fontSize="10">B</text>
                  </svg>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Gradient Descent Visualization */}
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Gradient Descent Visualization</CardTitle>
              <CardDescription className="text-slate-400">Watch how gradient descent finds the minimum of f(x,y) = x² + y²</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-6">
                <div className="bg-slate-900 rounded-lg p-4">
                  <svg viewBox="0 0 200 200" className="w-full h-64">
                    {/* Contour lines (loss function) */}
                    {[80, 60, 40, 20, 10].map((r, i) => (
                      <circle
                        key={i}
                        cx="100"
                        cy="100"
                        r={r}
                        fill="none"
                        stroke={`rgba(139, 92, 246, ${0.2 + i * 0.15})`}
                        strokeWidth="1"
                      />
                    ))}
                    
                    {/* Path */}
                    {gradientSteps.slice(0, gradientStep + 1).map((step, i) => {
                      if (i === 0) return null
                      const prev = gradientSteps[i - 1]
                      return (
                        <line
                          key={i}
                          x1={100 + prev.x * 15}
                          y1={100 - prev.y * 15}
                          x2={100 + step.x * 15}
                          y2={100 - step.y * 15}
                          stroke="#f59e0b"
                          strokeWidth="2"
                        />
                      )
                    })}
                    
                    {/* Current position */}
                    <circle
                      cx={100 + gradientSteps[gradientStep].x * 15}
                      cy={100 - gradientSteps[gradientStep].y * 15}
                      r="6"
                      fill="#ef4444"
                    />
                    
                    {/* Minimum */}
                    <circle cx="100" cy="100" r="4" fill="#22c55e" />
                    <text x="105" y="115" fill="#22c55e" fontSize="10">min</text>
                  </svg>
                </div>
                <div className="space-y-4">
                  <div className="bg-slate-900 p-4 rounded-lg">
                    <p className="text-slate-300">Step: <span className="text-purple-400 font-bold">{gradientStep}</span></p>
                    <p className="text-slate-300">Position: <span className="text-blue-400">({gradientSteps[gradientStep].x.toFixed(2)}, {gradientSteps[gradientStep].y.toFixed(2)})</span></p>
                    <p className="text-slate-300">Loss: <span className="text-red-400 font-bold">{gradientSteps[gradientStep].loss.toFixed(2)}</span></p>
                  </div>
                  <div className="flex gap-2">
                    <Button onClick={runGradientDescent} className="bg-purple-600 hover:bg-purple-700">
                      Step
                    </Button>
                    <Button onClick={resetGradientDescent} variant="outline">
                      Reset
                    </Button>
                  </div>
                  <p className="text-slate-400 text-sm">
                    Each step moves opposite to the gradient, reducing the loss until we reach the minimum!
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Softmax Visualization */}
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Softmax: Scores → Probabilities</CardTitle>
              <CardDescription className="text-slate-400">See how raw scores become probabilities</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-4">
                {[
                  { name: 'Cat', score: 2.0, prob: 0.659 },
                  { name: 'Dog', score: 1.0, prob: 0.242 },
                  { name: 'Bird', score: 0.1, prob: 0.099 },
                ].map((item, i) => (
                  <div key={i} className="bg-slate-900 p-4 rounded-lg text-center">
                    <p className="text-slate-400 text-sm">Score: {item.score}</p>
                    <div className="my-2">
                      <Calculator className="w-6 h-6 mx-auto text-purple-400" />
                    </div>
                    <p className="text-white font-bold">{item.name}</p>
                    <div className="mt-2 bg-slate-800 rounded-full h-4 overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-purple-500 to-pink-500"
                        style={{ width: `${item.prob * 100}%` }}
                      />
                    </div>
                    <p className="text-green-400 font-bold mt-1">{(item.prob * 100).toFixed(1)}%</p>
                  </div>
                ))}
              </div>
              <p className="text-slate-400 text-sm text-center mt-4">
                Softmax converts any scores to probabilities that sum to 100%
              </p>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="code" className="space-y-6">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Calculator className="w-5 h-5 text-purple-400" />
                Complete Math for AI - Python Examples
              </CardTitle>
              <CardDescription className="text-slate-400">
                Run: python 00_math_for_ai.py
              </CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="bg-slate-900 p-4 rounded-lg overflow-x-auto text-sm text-slate-300">
                <code>{pythonCode}</code>
              </pre>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default Module00MathForAI
