import { useState } from 'react'
import './App.css'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { 
  Brain, 
  Network, 
  Layers, 
  MessageSquare, 
  Eye, 
  Sparkles, 
  Zap, 
  Bot, 
  Lightbulb, 
  Database, 
  Settings, 
  Code, 
  Gamepad2, 
  Shield, 
  Globe,
  ChevronRight,
  ChevronLeft,
  Play,
  BookOpen,
  Home
} from 'lucide-react'

import Module01Introduction from './modules/Module01Introduction'
import Module02MachineLearning from './modules/Module02MachineLearning'
import Module03NeuralNetworks from './modules/Module03NeuralNetworks'
import Module04DeepLearning from './modules/Module04DeepLearning'
import Module05NLP from './modules/Module05NLP'
import Module06ComputerVision from './modules/Module06ComputerVision'
import Module07GenerativeAI from './modules/Module07GenerativeAI'
import Module08Transformers from './modules/Module08Transformers'
import Module09LLMs from './modules/Module09LLMs'
import Module10PromptEngineering from './modules/Module10PromptEngineering'
import Module11RAG from './modules/Module11RAG'
import Module12FineTuning from './modules/Module12FineTuning'
import Module13Frameworks from './modules/Module13Frameworks'
import Module14ReinforcementLearning from './modules/Module14ReinforcementLearning'
import Module15Ethics from './modules/Module15Ethics'
import Module16Applications from './modules/Module16Applications'

const modules = [
  { id: 1, title: 'Introduction to AI', icon: Brain, color: 'bg-blue-500', description: 'What is AI and why it matters' },
  { id: 2, title: 'Machine Learning Basics', icon: Network, color: 'bg-green-500', description: 'Teaching computers to learn' },
  { id: 3, title: 'Neural Networks', icon: Layers, color: 'bg-purple-500', description: 'Brain-inspired computing' },
  { id: 4, title: 'Deep Learning', icon: Layers, color: 'bg-indigo-500', description: 'Many layers of learning' },
  { id: 5, title: 'Natural Language Processing', icon: MessageSquare, color: 'bg-pink-500', description: 'Understanding human language' },
  { id: 6, title: 'Computer Vision', icon: Eye, color: 'bg-orange-500', description: 'Teaching computers to see' },
  { id: 7, title: 'Generative AI', icon: Sparkles, color: 'bg-yellow-500', description: 'Creating new content' },
  { id: 8, title: 'Transformers & Attention', icon: Zap, color: 'bg-red-500', description: 'The magic behind modern AI' },
  { id: 9, title: 'Large Language Models', icon: Bot, color: 'bg-cyan-500', description: 'ChatGPT and friends' },
  { id: 10, title: 'Prompt Engineering', icon: Lightbulb, color: 'bg-amber-500', description: 'Talking to AI effectively' },
  { id: 11, title: 'RAG Systems', icon: Database, color: 'bg-emerald-500', description: 'AI with memory' },
  { id: 12, title: 'Fine-tuning & Transfer', icon: Settings, color: 'bg-violet-500', description: 'Customizing AI models' },
  { id: 13, title: 'AI Frameworks', icon: Code, color: 'bg-slate-500', description: 'Tools for building AI' },
  { id: 14, title: 'Reinforcement Learning', icon: Gamepad2, color: 'bg-rose-500', description: 'Learning from rewards' },
  { id: 15, title: 'AI Ethics & Safety', icon: Shield, color: 'bg-teal-500', description: 'Responsible AI use' },
  { id: 16, title: 'AI Applications', icon: Globe, color: 'bg-sky-500', description: 'AI in the real world' },
]

const moduleComponents = [
  Module01Introduction,
  Module02MachineLearning,
  Module03NeuralNetworks,
  Module04DeepLearning,
  Module05NLP,
  Module06ComputerVision,
  Module07GenerativeAI,
  Module08Transformers,
  Module09LLMs,
  Module10PromptEngineering,
  Module11RAG,
  Module12FineTuning,
  Module13Frameworks,
  Module14ReinforcementLearning,
  Module15Ethics,
  Module16Applications,
]

function App() {
  const [currentModule, setCurrentModule] = useState<number | null>(null)
  const [completedModules, setCompletedModules] = useState<number[]>([])

  const markComplete = (moduleId: number) => {
    if (!completedModules.includes(moduleId)) {
      setCompletedModules([...completedModules, moduleId])
    }
  }

  const progress = (completedModules.length / modules.length) * 100

  if (currentModule !== null) {
    const ModuleComponent = moduleComponents[currentModule - 1]
    const module = modules[currentModule - 1]
    
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
        <header className="sticky top-0 z-50 bg-slate-900/80 backdrop-blur-sm border-b border-slate-700">
          <div className="container mx-auto px-4 py-3 flex items-center justify-between">
            <Button 
              variant="ghost" 
              onClick={() => setCurrentModule(null)}
              className="text-white hover:bg-slate-800"
            >
              <Home className="w-4 h-4 mr-2" />
              Back to Modules
            </Button>
            <div className="flex items-center gap-2">
              <Badge className={`${module.color} text-white`}>
                Module {currentModule}
              </Badge>
              <span className="text-white font-medium">{module.title}</span>
            </div>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={currentModule === 1}
                onClick={() => setCurrentModule(currentModule - 1)}
                className="border-slate-600 text-slate-300"
              >
                <ChevronLeft className="w-4 h-4" />
                Previous
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={currentModule === modules.length}
                onClick={() => {
                  markComplete(currentModule)
                  setCurrentModule(currentModule + 1)
                }}
                className="border-slate-600 text-slate-300"
              >
                Next
                <ChevronRight className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </header>
        <main className="container mx-auto px-4 py-8">
          <ModuleComponent />
        </main>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <header className="py-12 text-center">
        <div className="flex justify-center mb-6">
          <div className="relative">
            <Brain className="w-20 h-20 text-purple-400 animate-pulse" />
            <Sparkles className="w-8 h-8 text-yellow-400 absolute -top-2 -right-2 animate-bounce" />
          </div>
        </div>
        <h1 className="text-5xl font-bold text-white mb-4">
          GenAI Learning Adventure
        </h1>
        <p className="text-xl text-purple-200 max-w-2xl mx-auto">
          Explore the fascinating world of Artificial Intelligence through interactive visualizations and hands-on Python examples. Perfect for beginners and kids!
        </p>
        <div className="mt-8 flex justify-center items-center gap-4">
          <div className="bg-slate-800/50 rounded-full px-6 py-2 flex items-center gap-3">
            <BookOpen className="w-5 h-5 text-purple-400" />
            <span className="text-white">{modules.length} Modules</span>
          </div>
          <div className="bg-slate-800/50 rounded-full px-6 py-2 flex items-center gap-3">
            <Play className="w-5 h-5 text-green-400" />
            <span className="text-white">Interactive Visualizations</span>
          </div>
        </div>
        <div className="mt-6 max-w-md mx-auto">
          <div className="flex justify-between text-sm text-slate-400 mb-2">
            <span>Progress</span>
            <span>{completedModules.length}/{modules.length} completed</span>
          </div>
          <Progress value={progress} className="h-3" />
        </div>
      </header>

      <main className="container mx-auto px-4 pb-12">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {modules.map((module) => {
            const Icon = module.icon
            const isCompleted = completedModules.includes(module.id)
            
            return (
              <Card 
                key={module.id}
                className={`bg-slate-800/50 border-slate-700 hover:border-purple-500 transition-all duration-300 cursor-pointer transform hover:scale-105 hover:shadow-xl hover:shadow-purple-500/20 ${isCompleted ? 'ring-2 ring-green-500' : ''}`}
                onClick={() => setCurrentModule(module.id)}
              >
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <div className={`w-12 h-12 rounded-lg ${module.color} flex items-center justify-center`}>
                      <Icon className="w-6 h-6 text-white" />
                    </div>
                    <Badge variant="outline" className="text-slate-400 border-slate-600">
                      {String(module.id).padStart(2, '0')}
                    </Badge>
                  </div>
                  <CardTitle className="text-white mt-3">{module.title}</CardTitle>
                  <CardDescription className="text-slate-400">
                    {module.description}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Button 
                    className={`w-full ${isCompleted ? 'bg-green-600 hover:bg-green-700' : 'bg-purple-600 hover:bg-purple-700'}`}
                  >
                    {isCompleted ? 'Review' : 'Start Learning'}
                    <ChevronRight className="w-4 h-4 ml-2" />
                  </Button>
                </CardContent>
              </Card>
            )
          })}
        </div>
      </main>

      <footer className="py-8 text-center text-slate-400 border-t border-slate-800">
        <p>Built with love for curious minds everywhere</p>
      </footer>
    </div>
  )
}

export default App
