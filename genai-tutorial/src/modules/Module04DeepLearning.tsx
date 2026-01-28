import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Layers, Image, MessageSquare, Code, Zap, ArrowDown } from 'lucide-react'

const Module04DeepLearning = () => {
  const [activeLayer, setActiveLayer] = useState<number | null>(null)
  const [_showFeatures, _setShowFeatures] = useState(false)

  const layers = [
    { name: 'Input', desc: 'Raw pixel data (28x28 = 784 values)', color: 'bg-blue-500', features: 'Raw image pixels' },
    { name: 'Conv 1', desc: 'Detects edges and simple patterns', color: 'bg-purple-500', features: 'Edges, lines, corners' },
    { name: 'Conv 2', desc: 'Combines edges into shapes', color: 'bg-indigo-500', features: 'Circles, curves, textures' },
    { name: 'Conv 3', desc: 'Recognizes parts of objects', color: 'bg-violet-500', features: 'Eyes, wheels, windows' },
    { name: 'Dense', desc: 'Combines all features', color: 'bg-pink-500', features: 'Full object understanding' },
    { name: 'Output', desc: 'Final classification', color: 'bg-green-500', features: 'Cat, Dog, Car, etc.' },
  ]

  const pythonCode = `# Deep Learning - Convolutional Neural Networks (CNNs)
# Teaching computers to see and understand images!

import numpy as np

# ============================================
# Understanding Convolution
# ============================================

def convolve2d(image, kernel):
    """
    Convolution: Sliding a small filter over an image
    
    Think of it like looking at a picture through a magnifying glass
    and moving it around to find patterns!
    """
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    
    # Output size
    out_h = img_h - k_h + 1
    out_w = img_w - k_w + 1
    output = np.zeros((out_h, out_w))
    
    # Slide the kernel over the image
    for i in range(out_h):
        for j in range(out_w):
            # Extract the region under the kernel
            region = image[i:i+k_h, j:j+k_w]
            # Multiply and sum (dot product)
            output[i, j] = np.sum(region * kernel)
    
    return output

# Example: Edge detection kernel
edge_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

# Simple 5x5 image
sample_image = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
])

result = convolve2d(sample_image, edge_kernel)
print("Edge detection result:")
print(result)

# ============================================
# Max Pooling
# ============================================

def max_pool2d(image, pool_size=2):
    """
    Max Pooling: Shrinks the image while keeping important features
    
    Like summarizing a paragraph - keep the main points!
    """
    h, w = image.shape
    out_h = h // pool_size
    out_w = w // pool_size
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            region = image[i*pool_size:(i+1)*pool_size, 
                          j*pool_size:(j+1)*pool_size]
            output[i, j] = np.max(region)
    
    return output

# ============================================
# Complete CNN with PyTorch
# ============================================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SimpleCNN(nn.Module):
    """
    A Convolutional Neural Network for image classification
    
    Architecture:
    1. Convolutional layers - Find patterns
    2. Pooling layers - Reduce size
    3. Fully connected layers - Make decisions
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Conv block 1: 28x28 -> 14x14
        x = self.pool(self.relu(self.conv1(x)))
        
        # Conv block 2: 14x14 -> 7x7
        x = self.pool(self.relu(self.conv2(x)))
        
        # Conv block 3: 7x7 -> 3x3
        x = self.pool(self.relu(self.conv3(x)))
        
        # Flatten for fully connected layers
        x = x.view(-1, 64 * 3 * 3)
        
        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

# ============================================
# Training on MNIST (Handwritten Digits)
# ============================================

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download MNIST dataset
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN(num_classes=10).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, train_loader, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Accuracy={accuracy:.2f}%')

# Test the model
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

# Run training
train(model, train_loader, epochs=5)
test(model, test_loader)

# ============================================
# Using Pre-trained Models (Transfer Learning)
# ============================================

from torchvision import models

# Load pre-trained ResNet
resnet = models.resnet18(pretrained=True)

# Freeze all layers
for param in resnet.parameters():
    param.requires_grad = False

# Replace final layer for your task
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 10)  # 10 classes

print("Pre-trained ResNet ready for fine-tuning!")`

  return (
    <div className="space-y-8">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-white mb-4">Deep Learning</h1>
        <p className="text-xl text-purple-200">Many layers of learning - how AI masters complex tasks!</p>
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
                <Layers className="w-6 h-6 text-indigo-400" />
                What is Deep Learning?
              </CardTitle>
            </CardHeader>
            <CardContent className="text-slate-300 space-y-4">
              <p className="text-lg">
                <strong className="text-indigo-400">Deep Learning</strong> is like having a super-powered 
                neural network with many layers! Each layer learns something different - from simple patterns 
                to complex concepts. It's called "deep" because of all these layers stacked together.
              </p>
              <div className="bg-slate-900/50 p-4 rounded-lg border border-indigo-500/30">
                <h4 className="text-indigo-400 font-semibold mb-2">Real-World Analogy:</h4>
                <p>Think of learning to recognize a face: First, you notice edges and colors (layer 1), 
                then shapes like eyes and nose (layer 2), then how they're arranged (layer 3), and finally 
                you recognize the whole face (output). Deep learning works the same way!</p>
              </div>
            </CardContent>
          </Card>

          <div className="grid md:grid-cols-2 gap-4">
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader>
                <div className="w-12 h-12 rounded-lg bg-blue-500 flex items-center justify-center mb-2">
                  <Image className="w-6 h-6 text-white" />
                </div>
                <CardTitle className="text-white">CNNs - For Images</CardTitle>
              </CardHeader>
              <CardContent className="text-slate-400">
                <p className="mb-3">Convolutional Neural Networks are perfect for understanding images!</p>
                <ul className="space-y-1 text-sm">
                  <li>• Face recognition</li>
                  <li>• Self-driving cars</li>
                  <li>• Medical image analysis</li>
                  <li>• Photo filters</li>
                </ul>
              </CardContent>
            </Card>

            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader>
                <div className="w-12 h-12 rounded-lg bg-green-500 flex items-center justify-center mb-2">
                  <MessageSquare className="w-6 h-6 text-white" />
                </div>
                <CardTitle className="text-white">RNNs - For Sequences</CardTitle>
              </CardHeader>
              <CardContent className="text-slate-400">
                <p className="mb-3">Recurrent Neural Networks understand sequences and time!</p>
                <ul className="space-y-1 text-sm">
                  <li>• Language translation</li>
                  <li>• Speech recognition</li>
                  <li>• Music generation</li>
                  <li>• Stock prediction</li>
                </ul>
              </CardContent>
            </Card>
          </div>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Key Deep Learning Concepts</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-start gap-3 p-3 bg-blue-500/10 rounded-lg border border-blue-500/30">
                <Badge className="bg-blue-500">Conv</Badge>
                <div>
                  <h4 className="text-white font-medium">Convolution</h4>
                  <p className="text-slate-400 text-sm">Sliding a small filter over data to find patterns - like using a magnifying glass!</p>
                </div>
              </div>
              <div className="flex items-start gap-3 p-3 bg-green-500/10 rounded-lg border border-green-500/30">
                <Badge className="bg-green-500">Pool</Badge>
                <div>
                  <h4 className="text-white font-medium">Pooling</h4>
                  <p className="text-slate-400 text-sm">Shrinking the data while keeping important info - like summarizing a book!</p>
                </div>
              </div>
              <div className="flex items-start gap-3 p-3 bg-purple-500/10 rounded-lg border border-purple-500/30">
                <Badge className="bg-purple-500">Drop</Badge>
                <div>
                  <h4 className="text-white font-medium">Dropout</h4>
                  <p className="text-slate-400 text-sm">Randomly turning off neurons during training - prevents memorizing, encourages learning!</p>
                </div>
              </div>
              <div className="flex items-start gap-3 p-3 bg-orange-500/10 rounded-lg border border-orange-500/30">
                <Badge className="bg-orange-500">Batch</Badge>
                <div>
                  <h4 className="text-white font-medium">Batch Normalization</h4>
                  <p className="text-slate-400 text-sm">Keeping values in a good range - helps the network learn faster and better!</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="visualize" className="space-y-6">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Zap className="w-5 h-5 text-indigo-400" />
                CNN Layer Visualization
              </CardTitle>
              <CardDescription className="text-slate-400">
                Click on each layer to see what it learns!
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col items-center space-y-2">
                {layers.map((layer, index) => (
                  <div key={index} className="w-full">
                    <Button
                      variant="outline"
                      className={`w-full py-6 ${layer.color} border-none text-white hover:opacity-90 transition-all ${activeLayer === index ? 'ring-2 ring-white scale-105' : ''}`}
                      onClick={() => setActiveLayer(activeLayer === index ? null : index)}
                    >
                      <div className="flex justify-between items-center w-full px-4">
                        <span className="font-bold">{layer.name}</span>
                        <span className="text-sm opacity-80">{layer.desc}</span>
                      </div>
                    </Button>
                    {activeLayer === index && (
                      <div className="bg-slate-900 p-4 rounded-b-lg border border-t-0 border-slate-600 animate-in slide-in-from-top">
                        <p className="text-slate-300">
                          <strong className="text-white">What this layer sees: </strong>
                          {layer.features}
                        </p>
                      </div>
                    )}
                    {index < layers.length - 1 && (
                      <div className="flex justify-center py-1">
                        <ArrowDown className="text-slate-500 w-5 h-5" />
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">How Convolution Works</CardTitle>
              <CardDescription className="text-slate-400">
                A filter slides over the image to detect patterns
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-4">
                <div className="bg-slate-900 p-4 rounded-lg">
                  <h4 className="text-blue-400 font-semibold mb-3 text-center">Input Image</h4>
                  <div className="grid grid-cols-5 gap-1 mx-auto w-fit">
                    {[0,0,0,0,0,0,1,1,1,0,0,1,0,1,0,0,1,1,1,0,0,0,0,0,0].map((v, i) => (
                      <div key={i} className={`w-6 h-6 rounded ${v ? 'bg-white' : 'bg-slate-700'}`} />
                    ))}
                  </div>
                </div>
                <div className="bg-slate-900 p-4 rounded-lg">
                  <h4 className="text-purple-400 font-semibold mb-3 text-center">Edge Filter</h4>
                  <div className="grid grid-cols-3 gap-1 mx-auto w-fit">
                    {[-1,-1,-1,-1,8,-1,-1,-1,-1].map((v, i) => (
                      <div key={i} className={`w-8 h-8 rounded flex items-center justify-center text-xs font-bold ${v > 0 ? 'bg-green-500 text-white' : 'bg-red-500 text-white'}`}>
                        {v}
                      </div>
                    ))}
                  </div>
                </div>
                <div className="bg-slate-900 p-4 rounded-lg">
                  <h4 className="text-green-400 font-semibold mb-3 text-center">Output (Edges)</h4>
                  <div className="grid grid-cols-3 gap-1 mx-auto w-fit">
                    {[0,0,0,0,8,0,0,0,0].map((v, i) => (
                      <div key={i} className={`w-8 h-8 rounded flex items-center justify-center text-xs font-bold ${v > 0 ? 'bg-green-500 text-white' : 'bg-slate-700 text-slate-400'}`}>
                        {v}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              <p className="text-slate-400 text-sm mt-4 text-center">
                The filter slides across the image, multiplying and summing values to detect edges!
              </p>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Popular Deep Learning Architectures</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-4">
                {[
                  { name: 'LeNet', year: '1998', desc: 'First successful CNN for digit recognition', layers: 5 },
                  { name: 'AlexNet', year: '2012', desc: 'Won ImageNet, started deep learning revolution', layers: 8 },
                  { name: 'VGGNet', year: '2014', desc: 'Very deep network with small filters', layers: 19 },
                  { name: 'ResNet', year: '2015', desc: 'Skip connections allow 150+ layers!', layers: 152 },
                ].map((arch, i) => (
                  <div key={i} className="bg-slate-900 p-4 rounded-lg">
                    <div className="flex justify-between items-start mb-2">
                      <h4 className="text-white font-semibold">{arch.name}</h4>
                      <Badge variant="outline" className="text-slate-400">{arch.year}</Badge>
                    </div>
                    <p className="text-slate-400 text-sm mb-2">{arch.desc}</p>
                    <div className="flex items-center gap-2">
                      <Layers className="w-4 h-4 text-indigo-400" />
                      <span className="text-indigo-400 text-sm">{arch.layers} layers</span>
                    </div>
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
                <Code className="w-5 h-5 text-indigo-400" />
                Python Code: Deep Learning with CNNs
              </CardTitle>
              <CardDescription className="text-slate-400">
                Build image classifiers with PyTorch!
              </CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="bg-slate-900 p-4 rounded-lg overflow-x-auto text-sm max-h-96 overflow-y-auto">
                <code className="text-green-400">{pythonCode}</code>
              </pre>
              <div className="mt-4 p-4 bg-indigo-500/10 rounded-lg border border-indigo-500/30">
                <h4 className="text-indigo-400 font-semibold mb-2">Required Libraries:</h4>
                <code className="text-slate-300 text-sm bg-slate-800 px-2 py-1 rounded">
                  pip install torch torchvision numpy
                </code>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default Module04DeepLearning
