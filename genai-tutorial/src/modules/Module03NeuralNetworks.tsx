import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Slider } from '@/components/ui/slider'
import { Brain, Code, Play, ArrowRight } from 'lucide-react'

const Module03NeuralNetworks = () => {
  const [inputValues, setInputValues] = useState([0.5, 0.8])
  const [weights] = useState([[0.4, 0.6], [0.3, 0.7], [0.5, 0.5]])
  const [activations, setActivations] = useState<number[][]>([[], [], []])
  const [isAnimating, setIsAnimating] = useState(false)

  const sigmoid = (x: number) => 1 / (1 + Math.exp(-x))

  const forwardPass = () => {
    setIsAnimating(true)
    
    const layer1 = weights.map(w => 
      sigmoid(inputValues[0] * w[0] + inputValues[1] * w[1])
    )
    
    setTimeout(() => {
      setActivations([inputValues, layer1, [sigmoid(layer1.reduce((a, b) => a + b, 0))]])
      setIsAnimating(false)
    }, 1000)
  }

  useEffect(() => {
    forwardPass()
  }, [inputValues])

  const pythonCode = `# Neural Networks from Scratch
# Understanding how artificial neurons work!

import numpy as np

# ============================================
# Single Neuron (Perceptron)
# ============================================

class Neuron:
    """
    A single artificial neuron - the building block of neural networks!
    
    Think of it like a tiny decision maker:
    1. It receives inputs (like signals from your senses)
    2. It multiplies each input by a weight (importance)
    3. It adds them all up
    4. It decides whether to "fire" based on the sum
    """
    
    def __init__(self, num_inputs):
        # Random weights to start (the neuron will learn better ones)
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
    
    def activate(self, x):
        """Sigmoid activation - squishes output between 0 and 1"""
        return 1 / (1 + np.exp(-x))
    
    def forward(self, inputs):
        """Process inputs and produce output"""
        # Weighted sum: multiply each input by its weight
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        # Apply activation function
        return self.activate(weighted_sum)

# Test a single neuron
neuron = Neuron(num_inputs=2)
inputs = np.array([0.5, 0.8])
output = neuron.forward(inputs)
print(f"Neuron output: {output:.4f}")

# ============================================
# Simple Neural Network Layer
# ============================================

class Layer:
    """
    A layer of neurons - multiple neurons working together!
    """
    
    def __init__(self, num_inputs, num_neurons):
        self.weights = np.random.randn(num_inputs, num_neurons) * 0.5
        self.biases = np.zeros(num_neurons)
    
    def forward(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.biases
        return 1 / (1 + np.exp(-weighted_sum))  # Sigmoid activation

# ============================================
# Complete Neural Network
# ============================================

class NeuralNetwork:
    """
    A complete neural network with multiple layers!
    
    Structure:
    Input Layer -> Hidden Layer(s) -> Output Layer
    
    Like a team of decision makers passing information along!
    """
    
    def __init__(self, layer_sizes):
        """
        layer_sizes: list of neurons in each layer
        Example: [2, 4, 1] means 2 inputs, 4 hidden neurons, 1 output
        """
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))
    
    def forward(self, inputs):
        """Pass data through all layers"""
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current
    
    def train(self, X, y, epochs=1000, learning_rate=0.1):
        """
        Train the network using backpropagation
        This is how the network learns from mistakes!
        """
        for epoch in range(epochs):
            total_error = 0
            
            for inputs, target in zip(X, y):
                # Forward pass
                output = self.forward(inputs)
                
                # Calculate error
                error = target - output
                total_error += np.mean(error ** 2)
                
                # Backpropagation (simplified)
                # Adjust weights based on error
                for layer in reversed(self.layers):
                    gradient = error * output * (1 - output)
                    layer.weights += learning_rate * np.outer(inputs, gradient)
                    layer.biases += learning_rate * gradient
            
            if epoch % 200 == 0:
                print(f"Epoch {epoch}, Error: {total_error/len(X):.4f}")

# ============================================
# Example: XOR Problem
# ============================================

# XOR is a classic problem that single neurons can't solve!
# But a neural network can!

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],  # 0 XOR 0 = 0
    [1],  # 0 XOR 1 = 1
    [1],  # 1 XOR 0 = 1
    [0]   # 1 XOR 1 = 0
])

# Create and train network
nn = NeuralNetwork([2, 4, 1])  # 2 inputs, 4 hidden, 1 output
nn.train(X, y, epochs=1000, learning_rate=0.5)

# Test the trained network
print("\\nTesting XOR:")
for inputs in X:
    output = nn.forward(inputs)
    print(f"{inputs[0]} XOR {inputs[1]} = {output[0]:.2f} (expected: {int(inputs[0]) ^ int(inputs[1])})")

# ============================================
# Using PyTorch (Popular Deep Learning Library)
# ============================================

import torch
import torch.nn as nn
import torch.optim as optim

class PyTorchNN(nn.Module):
    """Neural Network using PyTorch"""
    
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 4)   # 2 inputs -> 4 hidden
        self.layer2 = nn.Linear(4, 1)   # 4 hidden -> 1 output
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.sigmoid(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x

# Create model
model = PyTorchNN()

# Training setup
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

# Convert data to tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# Train
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        print(f"PyTorch Epoch {epoch}, Loss: {loss.item():.4f}")

print("\\nPyTorch Results:")
with torch.no_grad():
    predictions = model(X_tensor)
    for i, (inp, pred) in enumerate(zip(X, predictions)):
        print(f"{inp[0]} XOR {inp[1]} = {pred.item():.2f}")`

  return (
    <div className="space-y-8">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-white mb-4">Neural Networks</h1>
        <p className="text-xl text-purple-200">Brain-inspired computing - how AI learns to think!</p>
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
                <Brain className="w-6 h-6 text-purple-400" />
                What is a Neural Network?
              </CardTitle>
            </CardHeader>
            <CardContent className="text-slate-300 space-y-4">
              <p className="text-lg">
                A <strong className="text-purple-400">Neural Network</strong> is a computer system inspired by 
                the human brain! Just like your brain has billions of neurons connected together, a neural 
                network has artificial neurons that work together to solve problems.
              </p>
              <div className="bg-slate-900/50 p-4 rounded-lg border border-purple-500/30">
                <h4 className="text-purple-400 font-semibold mb-2">Think of it like this:</h4>
                <p>Imagine a team of friends passing notes in class. Each friend reads the note, adds their 
                own thoughts, and passes it to the next person. By the end, the final message is a combination 
                of everyone's input - that's how neural networks process information!</p>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Parts of a Neural Network</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-4">
                <div className="bg-blue-500/10 p-4 rounded-lg border border-blue-500/30">
                  <div className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center mb-3">
                    <span className="text-white font-bold">1</span>
                  </div>
                  <h4 className="text-blue-400 font-semibold mb-2">Input Layer</h4>
                  <p className="text-slate-400 text-sm">
                    Receives the data (like your eyes receiving light). Each input is a number representing 
                    something about the data.
                  </p>
                </div>
                <div className="bg-purple-500/10 p-4 rounded-lg border border-purple-500/30">
                  <div className="w-10 h-10 bg-purple-500 rounded-full flex items-center justify-center mb-3">
                    <span className="text-white font-bold">2</span>
                  </div>
                  <h4 className="text-purple-400 font-semibold mb-2">Hidden Layers</h4>
                  <p className="text-slate-400 text-sm">
                    The "thinking" part! These layers find patterns and features in the data. More layers = 
                    more complex patterns.
                  </p>
                </div>
                <div className="bg-green-500/10 p-4 rounded-lg border border-green-500/30">
                  <div className="w-10 h-10 bg-green-500 rounded-full flex items-center justify-center mb-3">
                    <span className="text-white font-bold">3</span>
                  </div>
                  <h4 className="text-green-400 font-semibold mb-2">Output Layer</h4>
                  <p className="text-slate-400 text-sm">
                    Gives the final answer! Could be a classification (cat or dog?) or a number (predicted price).
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">How a Single Neuron Works</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex flex-wrap items-center justify-center gap-4 text-center">
                <div className="bg-slate-900 p-3 rounded-lg">
                  <p className="text-blue-400 font-semibold">Inputs</p>
                  <p className="text-slate-400 text-sm">x₁, x₂, x₃...</p>
                </div>
                <ArrowRight className="text-slate-500" />
                <div className="bg-slate-900 p-3 rounded-lg">
                  <p className="text-purple-400 font-semibold">Multiply by Weights</p>
                  <p className="text-slate-400 text-sm">w₁x₁ + w₂x₂ + ...</p>
                </div>
                <ArrowRight className="text-slate-500" />
                <div className="bg-slate-900 p-3 rounded-lg">
                  <p className="text-yellow-400 font-semibold">Add Bias</p>
                  <p className="text-slate-400 text-sm">sum + b</p>
                </div>
                <ArrowRight className="text-slate-500" />
                <div className="bg-slate-900 p-3 rounded-lg">
                  <p className="text-orange-400 font-semibold">Activation</p>
                  <p className="text-slate-400 text-sm">σ(sum)</p>
                </div>
                <ArrowRight className="text-slate-500" />
                <div className="bg-slate-900 p-3 rounded-lg">
                  <p className="text-green-400 font-semibold">Output</p>
                  <p className="text-slate-400 text-sm">0 to 1</p>
                </div>
              </div>
              <div className="bg-yellow-500/10 p-4 rounded-lg border border-yellow-500/30">
                <h4 className="text-yellow-400 font-semibold mb-2">What are Weights?</h4>
                <p className="text-slate-300 text-sm">
                  Weights are like importance scores. A higher weight means that input matters more for the 
                  decision. The network learns the best weights during training!
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="visualize" className="space-y-6">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Play className="w-5 h-5 text-purple-400" />
                Interactive Neural Network
              </CardTitle>
              <CardDescription className="text-slate-400">
                Adjust the inputs and watch the signal flow through the network!
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid md:grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="text-slate-300 text-sm">Input 1: {inputValues[0].toFixed(2)}</label>
                  <Slider
                    value={[inputValues[0]]}
                    onValueChange={(v) => setInputValues([v[0], inputValues[1]])}
                    min={0}
                    max={1}
                    step={0.1}
                    className="mt-2"
                  />
                </div>
                <div>
                  <label className="text-slate-300 text-sm">Input 2: {inputValues[1].toFixed(2)}</label>
                  <Slider
                    value={[inputValues[1]]}
                    onValueChange={(v) => setInputValues([inputValues[0], v[0]])}
                    min={0}
                    max={1}
                    step={0.1}
                    className="mt-2"
                  />
                </div>
              </div>

              <div className="relative h-80 bg-slate-900 rounded-lg overflow-hidden p-4">
                <svg className="w-full h-full" viewBox="0 0 500 250">
                  {/* Connections from input to hidden */}
                  {[0, 1].map(i => 
                    [0, 1, 2].map(j => (
                      <line 
                        key={`ih-${i}-${j}`}
                        x1={80} 
                        y1={80 + i * 90} 
                        x2={200} 
                        y2={50 + j * 75}
                        stroke={isAnimating ? '#8b5cf6' : '#4b5563'}
                        strokeWidth={Math.abs(weights[j][i]) * 3}
                        className="transition-all duration-500"
                      />
                    ))
                  )}
                  
                  {/* Connections from hidden to output */}
                  {[0, 1, 2].map(i => (
                    <line 
                      key={`ho-${i}`}
                      x1={200} 
                      y1={50 + i * 75} 
                      x2={350} 
                      y2={125}
                      stroke={isAnimating ? '#22c55e' : '#4b5563'}
                      strokeWidth={2}
                      className="transition-all duration-500"
                    />
                  ))}

                  {/* Input neurons */}
                  {inputValues.map((val, i) => (
                    <g key={`input-${i}`}>
                      <circle 
                        cx={80} 
                        cy={80 + i * 90} 
                        r={25}
                        fill={`rgba(59, 130, 246, ${val})`}
                        stroke="#3b82f6"
                        strokeWidth={2}
                      />
                      <text x={80} y={85 + i * 90} textAnchor="middle" fill="white" fontSize="12">
                        {val.toFixed(2)}
                      </text>
                    </g>
                  ))}

                  {/* Hidden neurons */}
                  {activations[1].map((val, i) => (
                    <g key={`hidden-${i}`}>
                      <circle 
                        cx={200} 
                        cy={50 + i * 75} 
                        r={22}
                        fill={`rgba(139, 92, 246, ${val})`}
                        stroke="#8b5cf6"
                        strokeWidth={2}
                        className="transition-all duration-500"
                      />
                      <text x={200} y={55 + i * 75} textAnchor="middle" fill="white" fontSize="11">
                        {val.toFixed(2)}
                      </text>
                    </g>
                  ))}

                  {/* Output neuron */}
                  <circle 
                    cx={350} 
                    cy={125} 
                    r={30}
                    fill={`rgba(34, 197, 94, ${activations[2][0] || 0})`}
                    stroke="#22c55e"
                    strokeWidth={2}
                    className="transition-all duration-500"
                  />
                  <text x={350} y={130} textAnchor="middle" fill="white" fontSize="14" fontWeight="bold">
                    {(activations[2][0] || 0).toFixed(2)}
                  </text>

                  {/* Labels */}
                  <text x={80} y={20} textAnchor="middle" fill="#94a3b8" fontSize="12">Input Layer</text>
                  <text x={200} y={20} textAnchor="middle" fill="#94a3b8" fontSize="12">Hidden Layer</text>
                  <text x={350} y={20} textAnchor="middle" fill="#94a3b8" fontSize="12">Output</text>

                  {/* Legend */}
                  <text x={420} y={80} fill="#94a3b8" fontSize="10">Brightness =</text>
                  <text x={420} y={95} fill="#94a3b8" fontSize="10">Activation</text>
                  <text x={420} y={130} fill="#94a3b8" fontSize="10">Line thickness =</text>
                  <text x={420} y={145} fill="#94a3b8" fontSize="10">Weight strength</text>
                </svg>
              </div>

              <div className="bg-purple-500/10 p-4 rounded-lg border border-purple-500/30">
                <h4 className="text-purple-400 font-semibold mb-2">What's Happening?</h4>
                <p className="text-slate-300 text-sm">
                  The input values flow through the network. Each connection has a weight that determines 
                  how much influence it has. The hidden neurons combine the weighted inputs and apply an 
                  activation function. Finally, the output neuron gives us the result!
                </p>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Activation Functions</CardTitle>
              <CardDescription className="text-slate-400">
                These functions decide if a neuron should "fire" or not
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-4">
                <div className="bg-slate-900 p-4 rounded-lg">
                  <h4 className="text-blue-400 font-semibold mb-2">Sigmoid</h4>
                  <div className="h-24 flex items-center justify-center">
                    <svg viewBox="0 0 100 60" className="w-full h-full">
                      <path d="M 10 50 Q 30 50 50 30 Q 70 10 90 10" fill="none" stroke="#3b82f6" strokeWidth="2" />
                      <line x1="10" y1="30" x2="90" y2="30" stroke="#4b5563" strokeDasharray="2" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-xs">Squishes values between 0 and 1</p>
                </div>
                <div className="bg-slate-900 p-4 rounded-lg">
                  <h4 className="text-green-400 font-semibold mb-2">ReLU</h4>
                  <div className="h-24 flex items-center justify-center">
                    <svg viewBox="0 0 100 60" className="w-full h-full">
                      <path d="M 10 50 L 50 50 L 90 10" fill="none" stroke="#22c55e" strokeWidth="2" />
                      <line x1="10" y1="50" x2="90" y2="50" stroke="#4b5563" strokeDasharray="2" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-xs">Zero for negatives, linear for positives</p>
                </div>
                <div className="bg-slate-900 p-4 rounded-lg">
                  <h4 className="text-purple-400 font-semibold mb-2">Tanh</h4>
                  <div className="h-24 flex items-center justify-center">
                    <svg viewBox="0 0 100 60" className="w-full h-full">
                      <path d="M 10 55 Q 30 55 50 30 Q 70 5 90 5" fill="none" stroke="#8b5cf6" strokeWidth="2" />
                      <line x1="10" y1="30" x2="90" y2="30" stroke="#4b5563" strokeDasharray="2" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-xs">Squishes values between -1 and 1</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="code" className="space-y-6">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Code className="w-5 h-5 text-purple-400" />
                Python Code: Neural Networks
              </CardTitle>
              <CardDescription className="text-slate-400">
                Build neural networks from scratch and with PyTorch!
              </CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="bg-slate-900 p-4 rounded-lg overflow-x-auto text-sm max-h-96 overflow-y-auto">
                <code className="text-green-400">{pythonCode}</code>
              </pre>
              <div className="mt-4 p-4 bg-purple-500/10 rounded-lg border border-purple-500/30">
                <h4 className="text-purple-400 font-semibold mb-2">Required Libraries:</h4>
                <code className="text-slate-300 text-sm bg-slate-800 px-2 py-1 rounded">
                  pip install numpy torch
                </code>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default Module03NeuralNetworks
