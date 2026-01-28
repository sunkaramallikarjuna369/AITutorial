import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Brain, Lightbulb, Zap, Bot, Eye, MessageSquare, Play, Code } from 'lucide-react'

const Module01Introduction = () => {
  const [aiThinking, setAiThinking] = useState(false)
  const [inputValue, setInputValue] = useState('')
  const [aiResponse, setAiResponse] = useState('')
  const [animatedNodes, setAnimatedNodes] = useState<number[]>([])

  useEffect(() => {
    const interval = setInterval(() => {
      setAnimatedNodes(prev => {
        const newNodes = [...prev]
        const randomNode = Math.floor(Math.random() * 6)
        if (newNodes.includes(randomNode)) {
          return newNodes.filter(n => n !== randomNode)
        }
        return [...newNodes, randomNode].slice(-3)
      })
    }, 500)
    return () => clearInterval(interval)
  }, [])

  const simulateAI = () => {
    if (!inputValue.trim()) return
    setAiThinking(true)
    setAiResponse('')
    
    setTimeout(() => {
      const responses = [
        `I understood your input: "${inputValue}". AI processes text by breaking it into tokens!`,
        `Analyzing "${inputValue}"... AI uses patterns learned from millions of examples!`,
        `Processing "${inputValue}"... This is how AI understands human language!`
      ]
      setAiResponse(responses[Math.floor(Math.random() * responses.length)])
      setAiThinking(false)
    }, 1500)
  }

  const pythonCode = `# Introduction to AI - Basic Concepts
# This program demonstrates simple AI decision making

def simple_ai_decision(temperature):
    """
    A simple rule-based AI that decides what to wear
    This is the most basic form of AI - following rules!
    """
    if temperature > 30:
        return "It's hot! Wear light clothes and stay hydrated."
    elif temperature > 20:
        return "Nice weather! A t-shirt would be perfect."
    elif temperature > 10:
        return "It's cool. Consider wearing a jacket."
    else:
        return "It's cold! Bundle up with warm clothes."

# Test our simple AI
temperatures = [35, 25, 15, 5]
for temp in temperatures:
    decision = simple_ai_decision(temp)
    print(f"Temperature: {temp}°C -> AI says: {decision}")

# Output:
# Temperature: 35°C -> AI says: It's hot! Wear light clothes and stay hydrated.
# Temperature: 25°C -> AI says: Nice weather! A t-shirt would be perfect.
# Temperature: 15°C -> AI says: It's cool. Consider wearing a jacket.
# Temperature: 5°C -> AI says: It's cold! Bundle up with warm clothes.

# ============================================
# Machine Learning vs Rule-Based AI
# ============================================

import random

class SimpleLearningAI:
    """
    A simple AI that learns from examples
    This demonstrates the basic concept of machine learning!
    """
    def __init__(self):
        self.learned_patterns = {}
    
    def learn(self, input_data, correct_output):
        """Learn from an example"""
        self.learned_patterns[input_data] = correct_output
        print(f"Learned: {input_data} -> {correct_output}")
    
    def predict(self, input_data):
        """Make a prediction based on what we learned"""
        if input_data in self.learned_patterns:
            return self.learned_patterns[input_data]
        return "I haven't learned this yet!"

# Create and train our AI
ai = SimpleLearningAI()
ai.learn("cat picture", "This is a cat!")
ai.learn("dog picture", "This is a dog!")
ai.learn("bird picture", "This is a bird!")

# Test predictions
print(ai.predict("cat picture"))  # This is a cat!
print(ai.predict("fish picture")) # I haven't learned this yet!`

  const aiTypes = [
    { icon: Bot, title: 'Narrow AI', desc: 'AI designed for specific tasks like playing chess or recognizing faces', color: 'bg-blue-500' },
    { icon: Brain, title: 'General AI', desc: 'AI that can learn any intellectual task a human can (still theoretical)', color: 'bg-purple-500' },
    { icon: Zap, title: 'Super AI', desc: 'AI smarter than all humans combined (science fiction for now)', color: 'bg-red-500' },
  ]

  return (
    <div className="space-y-8">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-white mb-4">Welcome to Artificial Intelligence!</h1>
        <p className="text-xl text-purple-200">Let's discover what AI is and how it works - explained simply!</p>
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
                <Lightbulb className="w-6 h-6 text-yellow-400" />
                What is Artificial Intelligence?
              </CardTitle>
            </CardHeader>
            <CardContent className="text-slate-300 space-y-4">
              <p className="text-lg">
                <strong className="text-purple-400">Artificial Intelligence (AI)</strong> is like teaching a computer to think and learn, 
                similar to how you learn new things! Just like you learned to recognize your friends' faces or understand 
                when someone is happy or sad, AI can learn to do similar things.
              </p>
              <div className="bg-slate-900/50 p-4 rounded-lg border border-purple-500/30">
                <h4 className="text-purple-400 font-semibold mb-2">Think of it this way:</h4>
                <p>When you see a cat, your brain instantly knows it's a cat because you've seen many cats before. 
                AI works the same way - we show it thousands of cat pictures, and it learns to recognize cats too!</p>
              </div>
            </CardContent>
          </Card>

          <div className="grid md:grid-cols-3 gap-4">
            {aiTypes.map((type, index) => (
              <Card key={index} className="bg-slate-800/50 border-slate-700 hover:border-purple-500 transition-all">
                <CardHeader>
                  <div className={`w-12 h-12 rounded-lg ${type.color} flex items-center justify-center mb-2`}>
                    <type.icon className="w-6 h-6 text-white" />
                  </div>
                  <CardTitle className="text-white text-lg">{type.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-slate-400 text-sm">{type.desc}</p>
                </CardContent>
              </Card>
            ))}
          </div>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Key Concepts to Remember</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-start gap-3 p-3 bg-blue-500/10 rounded-lg border border-blue-500/30">
                <Badge className="bg-blue-500">1</Badge>
                <div>
                  <h4 className="text-white font-medium">AI Learns from Data</h4>
                  <p className="text-slate-400 text-sm">Just like you learn from books and experiences, AI learns from data (pictures, text, numbers)</p>
                </div>
              </div>
              <div className="flex items-start gap-3 p-3 bg-green-500/10 rounded-lg border border-green-500/30">
                <Badge className="bg-green-500">2</Badge>
                <div>
                  <h4 className="text-white font-medium">AI Finds Patterns</h4>
                  <p className="text-slate-400 text-sm">AI is really good at finding patterns - like noticing that all cats have whiskers and pointy ears</p>
                </div>
              </div>
              <div className="flex items-start gap-3 p-3 bg-purple-500/10 rounded-lg border border-purple-500/30">
                <Badge className="bg-purple-500">3</Badge>
                <div>
                  <h4 className="text-white font-medium">AI Makes Predictions</h4>
                  <p className="text-slate-400 text-sm">Once AI learns patterns, it can predict things - like guessing if a new picture is a cat or dog</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="visualize" className="space-y-6">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Interactive AI Brain Visualization</CardTitle>
              <CardDescription className="text-slate-400">Watch how AI processes information through connected nodes</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="relative h-64 bg-slate-900 rounded-lg overflow-hidden">
                <svg className="w-full h-full" viewBox="0 0 400 200">
                  {/* Connections */}
                  <line x1="60" y1="100" x2="140" y2="60" stroke="#6366f1" strokeWidth="2" opacity="0.5" />
                  <line x1="60" y1="100" x2="140" y2="100" stroke="#6366f1" strokeWidth="2" opacity="0.5" />
                  <line x1="60" y1="100" x2="140" y2="140" stroke="#6366f1" strokeWidth="2" opacity="0.5" />
                  <line x1="140" y1="60" x2="260" y2="80" stroke="#8b5cf6" strokeWidth="2" opacity="0.5" />
                  <line x1="140" y1="100" x2="260" y2="80" stroke="#8b5cf6" strokeWidth="2" opacity="0.5" />
                  <line x1="140" y1="100" x2="260" y2="120" stroke="#8b5cf6" strokeWidth="2" opacity="0.5" />
                  <line x1="140" y1="140" x2="260" y2="120" stroke="#8b5cf6" strokeWidth="2" opacity="0.5" />
                  <line x1="260" y1="80" x2="340" y2="100" stroke="#a855f7" strokeWidth="2" opacity="0.5" />
                  <line x1="260" y1="120" x2="340" y2="100" stroke="#a855f7" strokeWidth="2" opacity="0.5" />
                  
                  {/* Input Node */}
                  <circle cx="60" cy="100" r="20" fill={animatedNodes.includes(0) ? '#22c55e' : '#3b82f6'} className="transition-all duration-300" />
                  <text x="60" y="105" textAnchor="middle" fill="white" fontSize="10">Input</text>
                  
                  {/* Hidden Layer 1 */}
                  <circle cx="140" cy="60" r="15" fill={animatedNodes.includes(1) ? '#22c55e' : '#6366f1'} className="transition-all duration-300" />
                  <circle cx="140" cy="100" r="15" fill={animatedNodes.includes(2) ? '#22c55e' : '#6366f1'} className="transition-all duration-300" />
                  <circle cx="140" cy="140" r="15" fill={animatedNodes.includes(3) ? '#22c55e' : '#6366f1'} className="transition-all duration-300" />
                  
                  {/* Hidden Layer 2 */}
                  <circle cx="260" cy="80" r="15" fill={animatedNodes.includes(4) ? '#22c55e' : '#8b5cf6'} className="transition-all duration-300" />
                  <circle cx="260" cy="120" r="15" fill={animatedNodes.includes(5) ? '#22c55e' : '#8b5cf6'} className="transition-all duration-300" />
                  
                  {/* Output Node */}
                  <circle cx="340" cy="100" r="20" fill="#a855f7" />
                  <text x="340" y="105" textAnchor="middle" fill="white" fontSize="10">Output</text>
                  
                  {/* Labels */}
                  <text x="60" y="30" textAnchor="middle" fill="#94a3b8" fontSize="12">Data In</text>
                  <text x="200" y="30" textAnchor="middle" fill="#94a3b8" fontSize="12">AI Processing</text>
                  <text x="340" y="30" textAnchor="middle" fill="#94a3b8" fontSize="12">Result</text>
                </svg>
                <div className="absolute bottom-4 left-4 right-4 text-center">
                  <p className="text-slate-400 text-sm">Green nodes show active processing - AI thinks by passing information through many connected nodes!</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Play className="w-5 h-5 text-green-400" />
                Try It Yourself: Simple AI Simulator
              </CardTitle>
              <CardDescription className="text-slate-400">Type something and see how AI processes your input!</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  placeholder="Type anything... (e.g., 'Hello AI!')"
                  className="flex-1 px-4 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-purple-500"
                />
                <Button onClick={simulateAI} disabled={aiThinking} className="bg-purple-600 hover:bg-purple-700">
                  {aiThinking ? 'Thinking...' : 'Ask AI'}
                </Button>
              </div>
              
              {aiThinking && (
                <div className="flex items-center gap-3 p-4 bg-purple-500/10 rounded-lg border border-purple-500/30">
                  <div className="animate-spin w-6 h-6 border-2 border-purple-500 border-t-transparent rounded-full" />
                  <span className="text-purple-300">AI is processing your input...</span>
                </div>
              )}
              
              {aiResponse && (
                <div className="p-4 bg-green-500/10 rounded-lg border border-green-500/30">
                  <div className="flex items-start gap-3">
                    <Bot className="w-6 h-6 text-green-400 mt-1" />
                    <div>
                      <p className="text-green-300 font-medium">AI Response:</p>
                      <p className="text-slate-300">{aiResponse}</p>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <div className="grid md:grid-cols-2 gap-4">
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Eye className="w-5 h-5 text-orange-400" />
                  AI Can See
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="aspect-video bg-slate-900 rounded-lg flex items-center justify-center relative overflow-hidden">
                  <div className="grid grid-cols-3 gap-2 p-4">
                    {['🐱', '🐕', '🐦', '🚗', '🏠', '🌳'].map((emoji, i) => (
                      <div key={i} className="w-16 h-16 bg-slate-800 rounded-lg flex items-center justify-center text-3xl hover:scale-110 transition-transform cursor-pointer">
                        {emoji}
                      </div>
                    ))}
                  </div>
                  <div className="absolute bottom-2 left-2 right-2 text-center text-slate-400 text-sm">
                    Computer Vision: AI recognizes objects in images
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <MessageSquare className="w-5 h-5 text-pink-400" />
                  AI Can Understand Language
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="aspect-video bg-slate-900 rounded-lg p-4 flex flex-col justify-center">
                  <div className="space-y-2">
                    <div className="bg-blue-500/20 p-2 rounded-lg text-blue-300 text-sm">
                      "The movie was amazing!" → 😊 Positive
                    </div>
                    <div className="bg-red-500/20 p-2 rounded-lg text-red-300 text-sm">
                      "I didn't like the food" → 😞 Negative
                    </div>
                    <div className="bg-yellow-500/20 p-2 rounded-lg text-yellow-300 text-sm">
                      "The weather is okay" → 😐 Neutral
                    </div>
                  </div>
                  <p className="text-slate-400 text-sm mt-3 text-center">
                    NLP: AI understands the meaning behind words
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="code" className="space-y-6">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Code className="w-5 h-5 text-green-400" />
                Python Code: Introduction to AI
              </CardTitle>
              <CardDescription className="text-slate-400">
                Copy and run this code to see AI concepts in action!
              </CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="bg-slate-900 p-4 rounded-lg overflow-x-auto text-sm">
                <code className="text-green-400">{pythonCode}</code>
              </pre>
              <div className="mt-4 p-4 bg-blue-500/10 rounded-lg border border-blue-500/30">
                <h4 className="text-blue-400 font-semibold mb-2">How to Run This Code:</h4>
                <ol className="text-slate-300 text-sm space-y-1 list-decimal list-inside">
                  <li>Install Python from python.org</li>
                  <li>Save the code as <code className="bg-slate-800 px-1 rounded">intro_to_ai.py</code></li>
                  <li>Open terminal and run: <code className="bg-slate-800 px-1 rounded">python intro_to_ai.py</code></li>
                </ol>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default Module01Introduction
