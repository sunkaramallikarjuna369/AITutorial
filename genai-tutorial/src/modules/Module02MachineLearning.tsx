import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Slider } from '@/components/ui/slider'
import { Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter } from 'recharts'
import { Network, TrendingUp, GitBranch, Code, Play, Target } from 'lucide-react'

const Module02MachineLearning = () => {
  const [trainingData] = useState([
    { x: 1, y: 2 }, { x: 2, y: 4 }, { x: 3, y: 5 }, { x: 4, y: 8 }, { x: 5, y: 10 }
  ])
  const [slope, setSlope] = useState([1.5])
  const [intercept, setIntercept] = useState([1])
  const [isTraining, setIsTraining] = useState(false)
  const [epoch, setEpoch] = useState(0)

  const predictedLine = Array.from({ length: 6 }, (_, i) => ({
    x: i,
    y: slope[0] * i + intercept[0]
  }))

  const calculateError = () => {
    let totalError = 0
    trainingData.forEach(point => {
      const predicted = slope[0] * point.x + intercept[0]
      totalError += Math.pow(point.y - predicted, 2)
    })
    return (totalError / trainingData.length).toFixed(2)
  }

  const trainModel = () => {
    setIsTraining(true)
    setEpoch(0)
    
    let currentSlope = slope[0]
    let currentIntercept = intercept[0]
    const learningRate = 0.01
    let currentEpoch = 0
    
    const trainStep = () => {
      if (currentEpoch >= 50) {
        setIsTraining(false)
        return
      }
      
      let slopeGradient = 0
      let interceptGradient = 0
      
      trainingData.forEach(point => {
        const predicted = currentSlope * point.x + currentIntercept
        const error = predicted - point.y
        slopeGradient += error * point.x
        interceptGradient += error
      })
      
      currentSlope -= (learningRate * slopeGradient) / trainingData.length
      currentIntercept -= (learningRate * interceptGradient) / trainingData.length
      
      setSlope([currentSlope])
      setIntercept([currentIntercept])
      currentEpoch++
      setEpoch(currentEpoch)
      
      setTimeout(trainStep, 100)
    }
    
    trainStep()
  }

  const mlTypes = [
    { 
      icon: TrendingUp, 
      title: 'Supervised Learning', 
      desc: 'Learning with a teacher! The AI learns from labeled examples (like flashcards with answers)',
      color: 'bg-blue-500',
      example: 'Teaching AI to recognize cats by showing it pictures labeled "cat" or "not cat"'
    },
    { 
      icon: GitBranch, 
      title: 'Unsupervised Learning', 
      desc: 'Learning without a teacher! The AI finds patterns on its own',
      color: 'bg-green-500',
      example: 'AI grouping similar customers together without being told the groups'
    },
    { 
      icon: Target, 
      title: 'Reinforcement Learning', 
      desc: 'Learning by trial and error! The AI gets rewards for good actions',
      color: 'bg-purple-500',
      example: 'AI learning to play video games by getting points for winning'
    },
  ]

  const pythonCode = `# Machine Learning Basics - Linear Regression
# This program teaches a computer to predict values!

import numpy as np

# ============================================
# Simple Linear Regression from Scratch
# ============================================

class SimpleLinearRegression:
    """
    Linear Regression: Finding the best line through data points
    
    Imagine you're trying to draw a straight line through dots on paper
    that gets as close to all the dots as possible!
    """
    
    def __init__(self):
        self.slope = 0      # How steep the line is
        self.intercept = 0  # Where the line crosses the y-axis
    
    def fit(self, X, y, learning_rate=0.01, epochs=100):
        """
        Train the model - find the best line!
        
        X: Input values (like hours studied)
        y: Output values (like test scores)
        """
        n = len(X)
        
        for epoch in range(epochs):
            # Make predictions with current line
            predictions = self.slope * X + self.intercept
            
            # Calculate how wrong we are (error)
            error = predictions - y
            
            # Adjust the line to reduce error (gradient descent)
            self.slope -= learning_rate * (2/n) * np.sum(error * X)
            self.intercept -= learning_rate * (2/n) * np.sum(error)
            
            if epoch % 20 == 0:
                mse = np.mean(error ** 2)
                print(f"Epoch {epoch}: Error = {mse:.4f}")
        
        print(f"\\nFinal line: y = {self.slope:.2f}x + {self.intercept:.2f}")
    
    def predict(self, X):
        """Use the trained line to make predictions"""
        return self.slope * X + self.intercept

# Example: Predicting test scores based on hours studied
hours_studied = np.array([1, 2, 3, 4, 5, 6, 7, 8])
test_scores = np.array([45, 50, 55, 60, 68, 72, 78, 85])

# Create and train the model
model = SimpleLinearRegression()
model.fit(hours_studied, test_scores, learning_rate=0.01, epochs=100)

# Make predictions
new_hours = np.array([9, 10])
predictions = model.predict(new_hours)
print(f"\\nPredictions:")
print(f"9 hours of study -> Score: {predictions[0]:.1f}")
print(f"10 hours of study -> Score: {predictions[1]:.1f}")

# ============================================
# Classification: Supervised Learning
# ============================================

class SimpleClassifier:
    """
    A simple classifier that learns to categorize things
    Like teaching AI to sort fruits into apples and oranges!
    """
    
    def __init__(self):
        self.categories = {}
    
    def fit(self, features, labels):
        """Learn the average features for each category"""
        for feature, label in zip(features, labels):
            if label not in self.categories:
                self.categories[label] = []
            self.categories[label].append(feature)
        
        # Calculate average for each category
        for label in self.categories:
            self.categories[label] = np.mean(self.categories[label])
    
    def predict(self, feature):
        """Find the closest category"""
        closest_label = None
        min_distance = float('inf')
        
        for label, avg_feature in self.categories.items():
            distance = abs(feature - avg_feature)
            if distance < min_distance:
                min_distance = distance
                closest_label = label
        
        return closest_label

# Example: Classifying fruits by weight
weights = [150, 160, 155, 200, 210, 195]  # grams
labels = ['apple', 'apple', 'apple', 'orange', 'orange', 'orange']

classifier = SimpleClassifier()
classifier.fit(weights, labels)

# Test predictions
test_weights = [158, 205, 175]
for weight in test_weights:
    prediction = classifier.predict(weight)
    print(f"Weight {weight}g -> Predicted: {prediction}")

# ============================================
# Using scikit-learn (Popular ML Library)
# ============================================

# pip install scikit-learn

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Prepare data
X = hours_studied.reshape(-1, 1)  # sklearn needs 2D array
y = test_scores

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)

# Evaluate
predictions = sklearn_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"\\nScikit-learn Model MSE: {mse:.2f}")`

  return (
    <div className="space-y-8">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-white mb-4">Machine Learning Basics</h1>
        <p className="text-xl text-purple-200">Teaching computers to learn from examples - just like you!</p>
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
                <Network className="w-6 h-6 text-green-400" />
                What is Machine Learning?
              </CardTitle>
            </CardHeader>
            <CardContent className="text-slate-300 space-y-4">
              <p className="text-lg">
                <strong className="text-green-400">Machine Learning (ML)</strong> is a way for computers to learn 
                from examples instead of being told exactly what to do. It's like teaching a dog tricks - 
                you show them what to do, reward good behavior, and they learn!
              </p>
              <div className="bg-slate-900/50 p-4 rounded-lg border border-green-500/30">
                <h4 className="text-green-400 font-semibold mb-2">Real-World Example:</h4>
                <p>Imagine teaching a computer to recognize spam emails. Instead of writing rules like 
                "if email contains 'FREE MONEY', it's spam", you show the computer thousands of spam and 
                non-spam emails. The computer learns the patterns itself!</p>
              </div>
            </CardContent>
          </Card>

          <div className="grid md:grid-cols-3 gap-4">
            {mlTypes.map((type, index) => (
              <Card key={index} className="bg-slate-800/50 border-slate-700 hover:border-green-500 transition-all">
                <CardHeader>
                  <div className={`w-12 h-12 rounded-lg ${type.color} flex items-center justify-center mb-2`}>
                    <type.icon className="w-6 h-6 text-white" />
                  </div>
                  <CardTitle className="text-white text-lg">{type.title}</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <p className="text-slate-400 text-sm">{type.desc}</p>
                  <div className="bg-slate-900/50 p-2 rounded text-xs text-slate-500">
                    <strong>Example:</strong> {type.example}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">The ML Learning Process</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap justify-center gap-4">
                {['Collect Data', 'Prepare Data', 'Choose Model', 'Train Model', 'Evaluate', 'Predict'].map((step, i) => (
                  <div key={i} className="flex items-center">
                    <div className="bg-gradient-to-r from-green-500 to-emerald-500 text-white px-4 py-2 rounded-lg font-medium">
                      {i + 1}. {step}
                    </div>
                    {i < 5 && <span className="text-green-400 mx-2">→</span>}
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
                <Play className="w-5 h-5 text-green-400" />
                Interactive Linear Regression Trainer
              </CardTitle>
              <CardDescription className="text-slate-400">
                Watch the AI learn to draw the best line through the data points!
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="h-64 bg-slate-900 rounded-lg p-4">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis type="number" dataKey="x" domain={[0, 6]} stroke="#9ca3af" />
                    <YAxis type="number" domain={[0, 12]} stroke="#9ca3af" />
                    <Tooltip />
                    <Scatter name="Training Data" data={trainingData} fill="#22c55e" />
                    <Line 
                      type="linear" 
                      dataKey="y" 
                      data={predictedLine} 
                      stroke="#8b5cf6" 
                      strokeWidth={2}
                      dot={false}
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                <div className="space-y-4">
                  <div>
                    <label className="text-slate-300 text-sm">Slope (steepness): {slope[0].toFixed(2)}</label>
                    <Slider
                      value={slope}
                      onValueChange={setSlope}
                      min={0}
                      max={3}
                      step={0.1}
                      className="mt-2"
                      disabled={isTraining}
                    />
                  </div>
                  <div>
                    <label className="text-slate-300 text-sm">Intercept (starting point): {intercept[0].toFixed(2)}</label>
                    <Slider
                      value={intercept}
                      onValueChange={setIntercept}
                      min={-2}
                      max={4}
                      step={0.1}
                      className="mt-2"
                      disabled={isTraining}
                    />
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="bg-slate-900 p-4 rounded-lg">
                    <p className="text-slate-400 text-sm">Current Error (lower is better):</p>
                    <p className="text-2xl font-bold text-red-400">{calculateError()}</p>
                  </div>
                  <div className="bg-slate-900 p-4 rounded-lg">
                    <p className="text-slate-400 text-sm">Training Epoch:</p>
                    <p className="text-2xl font-bold text-green-400">{epoch}/50</p>
                  </div>
                </div>
              </div>

              <Button 
                onClick={trainModel} 
                disabled={isTraining}
                className="w-full bg-green-600 hover:bg-green-700"
              >
                {isTraining ? 'Training in Progress...' : 'Train the Model (Watch it Learn!)'}
              </Button>

              <div className="bg-blue-500/10 p-4 rounded-lg border border-blue-500/30">
                <h4 className="text-blue-400 font-semibold mb-2">What's Happening?</h4>
                <p className="text-slate-300 text-sm">
                  The AI is adjusting the line (slope and intercept) to minimize the error - 
                  the distance between the line and the green data points. This process is called 
                  <strong className="text-blue-400"> Gradient Descent</strong>!
                </p>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">How Different ML Algorithms Work</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-slate-900 p-4 rounded-lg">
                  <h4 className="text-blue-400 font-semibold mb-3">Linear Regression</h4>
                  <div className="h-32 flex items-center justify-center">
                    <svg viewBox="0 0 200 100" className="w-full h-full">
                      <circle cx="30" cy="70" r="5" fill="#22c55e" />
                      <circle cx="50" cy="55" r="5" fill="#22c55e" />
                      <circle cx="80" cy="45" r="5" fill="#22c55e" />
                      <circle cx="120" cy="35" r="5" fill="#22c55e" />
                      <circle cx="160" cy="20" r="5" fill="#22c55e" />
                      <line x1="20" y1="80" x2="180" y2="10" stroke="#8b5cf6" strokeWidth="2" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm mt-2">Finds the best straight line through data</p>
                </div>

                <div className="bg-slate-900 p-4 rounded-lg">
                  <h4 className="text-green-400 font-semibold mb-3">Decision Tree</h4>
                  <div className="h-32 flex items-center justify-center">
                    <svg viewBox="0 0 200 100" className="w-full h-full">
                      <circle cx="100" cy="15" r="10" fill="#22c55e" />
                      <line x1="100" y1="25" x2="60" y2="45" stroke="#22c55e" strokeWidth="2" />
                      <line x1="100" y1="25" x2="140" y2="45" stroke="#22c55e" strokeWidth="2" />
                      <circle cx="60" cy="55" r="8" fill="#3b82f6" />
                      <circle cx="140" cy="55" r="8" fill="#3b82f6" />
                      <line x1="60" y1="63" x2="40" y2="80" stroke="#3b82f6" strokeWidth="2" />
                      <line x1="60" y1="63" x2="80" y2="80" stroke="#3b82f6" strokeWidth="2" />
                      <rect x="30" y="82" width="20" height="12" fill="#8b5cf6" rx="2" />
                      <rect x="70" y="82" width="20" height="12" fill="#ef4444" rx="2" />
                      <rect x="130" y="82" width="20" height="12" fill="#8b5cf6" rx="2" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm mt-2">Makes decisions by asking yes/no questions</p>
                </div>

                <div className="bg-slate-900 p-4 rounded-lg">
                  <h4 className="text-purple-400 font-semibold mb-3">K-Nearest Neighbors</h4>
                  <div className="h-32 flex items-center justify-center">
                    <svg viewBox="0 0 200 100" className="w-full h-full">
                      <circle cx="40" cy="30" r="6" fill="#3b82f6" />
                      <circle cx="60" cy="50" r="6" fill="#3b82f6" />
                      <circle cx="45" cy="70" r="6" fill="#3b82f6" />
                      <circle cx="140" cy="40" r="6" fill="#ef4444" />
                      <circle cx="160" cy="60" r="6" fill="#ef4444" />
                      <circle cx="150" cy="80" r="6" fill="#ef4444" />
                      <circle cx="100" cy="50" r="8" fill="#22c55e" stroke="#fff" strokeWidth="2" />
                      <line x1="100" y1="50" x2="60" y2="50" stroke="#22c55e" strokeDasharray="4" />
                      <line x1="100" y1="50" x2="45" y2="70" stroke="#22c55e" strokeDasharray="4" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm mt-2">Classifies based on nearest neighbors</p>
                </div>

                <div className="bg-slate-900 p-4 rounded-lg">
                  <h4 className="text-orange-400 font-semibold mb-3">Clustering (K-Means)</h4>
                  <div className="h-32 flex items-center justify-center">
                    <svg viewBox="0 0 200 100" className="w-full h-full">
                      <circle cx="40" cy="30" r="5" fill="#3b82f6" />
                      <circle cx="50" cy="45" r="5" fill="#3b82f6" />
                      <circle cx="35" cy="50" r="5" fill="#3b82f6" />
                      <circle cx="45" cy="40" r="10" fill="none" stroke="#3b82f6" strokeDasharray="3" />
                      <circle cx="150" cy="35" r="5" fill="#ef4444" />
                      <circle cx="160" cy="50" r="5" fill="#ef4444" />
                      <circle cx="145" cy="55" r="5" fill="#ef4444" />
                      <circle cx="155" cy="45" r="10" fill="none" stroke="#ef4444" strokeDasharray="3" />
                      <circle cx="100" cy="75" r="5" fill="#22c55e" />
                      <circle cx="110" cy="85" r="5" fill="#22c55e" />
                      <circle cx="90" cy="80" r="5" fill="#22c55e" />
                      <circle cx="100" cy="80" r="10" fill="none" stroke="#22c55e" strokeDasharray="3" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm mt-2">Groups similar data points together</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="code" className="space-y-6">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Code className="w-5 h-5 text-green-400" />
                Python Code: Machine Learning Basics
              </CardTitle>
              <CardDescription className="text-slate-400">
                Learn to build ML models from scratch and with scikit-learn!
              </CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="bg-slate-900 p-4 rounded-lg overflow-x-auto text-sm max-h-96 overflow-y-auto">
                <code className="text-green-400">{pythonCode}</code>
              </pre>
              <div className="mt-4 p-4 bg-green-500/10 rounded-lg border border-green-500/30">
                <h4 className="text-green-400 font-semibold mb-2">Required Libraries:</h4>
                <code className="text-slate-300 text-sm bg-slate-800 px-2 py-1 rounded">
                  pip install numpy scikit-learn
                </code>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default Module02MachineLearning
