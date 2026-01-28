import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Globe, Stethoscope, Car, Briefcase, Code, Palette, GraduationCap, Factory } from 'lucide-react'

const Module16Applications = () => {
  const [selectedDomain, setSelectedDomain] = useState<string | null>(null)

  const domains = [
    {
      name: 'Healthcare',
      icon: Stethoscope,
      color: 'bg-red-500',
      applications: [
        { name: 'Medical Imaging', desc: 'Detect diseases in X-rays, MRIs, CT scans' },
        { name: 'Drug Discovery', desc: 'AI designs new medicines faster' },
        { name: 'Diagnosis Assistant', desc: 'Help doctors identify conditions' },
        { name: 'Personalized Treatment', desc: 'Tailor treatments to individuals' },
      ],
      frameworks: ['TensorFlow', 'PyTorch', 'MONAI'],
      example: 'Google DeepMind AlphaFold predicting protein structures'
    },
    {
      name: 'Autonomous Vehicles',
      icon: Car,
      color: 'bg-blue-500',
      applications: [
        { name: 'Self-Driving Cars', desc: 'Navigate roads without human input' },
        { name: 'Object Detection', desc: 'Identify pedestrians, vehicles, signs' },
        { name: 'Path Planning', desc: 'Calculate optimal routes' },
        { name: 'Sensor Fusion', desc: 'Combine camera, lidar, radar data' },
      ],
      frameworks: ['PyTorch', 'TensorFlow', 'ROS'],
      example: 'Tesla Autopilot, Waymo self-driving taxis'
    },
    {
      name: 'Finance',
      icon: Briefcase,
      color: 'bg-green-500',
      applications: [
        { name: 'Fraud Detection', desc: 'Identify suspicious transactions' },
        { name: 'Algorithmic Trading', desc: 'Automated stock trading' },
        { name: 'Credit Scoring', desc: 'Assess loan risk' },
        { name: 'Chatbots', desc: 'Customer service automation' },
      ],
      frameworks: ['Scikit-learn', 'XGBoost', 'LangChain'],
      example: 'JPMorgan COIN analyzing legal documents'
    },
    {
      name: 'Creative Arts',
      icon: Palette,
      color: 'bg-purple-500',
      applications: [
        { name: 'Image Generation', desc: 'Create art from text prompts' },
        { name: 'Music Composition', desc: 'Generate original music' },
        { name: 'Video Editing', desc: 'Automated editing and effects' },
        { name: 'Writing Assistant', desc: 'Help with creative writing' },
      ],
      frameworks: ['Stable Diffusion', 'DALL-E', 'GPT-4'],
      example: 'Midjourney creating stunning artwork'
    },
    {
      name: 'Education',
      icon: GraduationCap,
      color: 'bg-yellow-500',
      applications: [
        { name: 'Personalized Learning', desc: 'Adapt to student pace' },
        { name: 'Tutoring Systems', desc: 'AI tutors for any subject' },
        { name: 'Grading Automation', desc: 'Assess essays and assignments' },
        { name: 'Language Learning', desc: 'Practice conversations with AI' },
      ],
      frameworks: ['GPT-4', 'LangChain', 'Hugging Face'],
      example: 'Khan Academy Khanmigo AI tutor'
    },
    {
      name: 'Manufacturing',
      icon: Factory,
      color: 'bg-orange-500',
      applications: [
        { name: 'Quality Control', desc: 'Detect defects automatically' },
        { name: 'Predictive Maintenance', desc: 'Prevent equipment failures' },
        { name: 'Supply Chain', desc: 'Optimize inventory and logistics' },
        { name: 'Robotics', desc: 'Automated assembly lines' },
      ],
      frameworks: ['TensorFlow', 'PyTorch', 'OpenCV'],
      example: 'Siemens AI-powered factories'
    },
  ]

  const pythonCode = `# AI Applications Across Industries
# Real-world implementations and code examples

import torch
import numpy as np
from transformers import pipeline
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# ============================================
# Healthcare: Medical Image Analysis
# ============================================

import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class MedicalImageClassifier:
    """
    Classify medical images (X-rays, CT scans, etc.)
    
    Applications:
    - Detect pneumonia in chest X-rays
    - Identify tumors in MRI scans
    - Screen for diabetic retinopathy
    """
    
    def __init__(self, num_classes=2):
        # Use pre-trained model
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
        
        return {
            'prediction': 'Abnormal' if prediction.item() == 1 else 'Normal',
            'confidence': probabilities.max().item()
        }

# ============================================
# Finance: Fraud Detection
# ============================================

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class FraudDetector:
    """
    Detect fraudulent transactions
    
    Features typically used:
    - Transaction amount
    - Time of transaction
    - Location
    - Device information
    - Historical patterns
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = IsolationForest(contamination=0.01, random_state=42)
    
    def train(self, transactions):
        """Train on historical transaction data"""
        X = self.scaler.fit_transform(transactions)
        self.model.fit(X)
    
    def predict(self, transaction):
        """Predict if transaction is fraudulent"""
        X = self.scaler.transform([transaction])
        prediction = self.model.predict(X)
        
        return {
            'is_fraud': prediction[0] == -1,
            'risk_score': self.model.score_samples(X)[0]
        }

# Real-time fraud detection
def process_transaction(transaction, detector):
    result = detector.predict(transaction)
    
    if result['is_fraud']:
        # Block transaction and alert
        return {'status': 'BLOCKED', 'reason': 'Suspected fraud'}
    
    return {'status': 'APPROVED'}

# ============================================
# Autonomous Vehicles: Object Detection
# ============================================

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

class ObjectDetector:
    """
    Detect objects for autonomous driving
    
    Detects: cars, pedestrians, cyclists, traffic signs, etc.
    """
    
    def __init__(self):
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        
        self.classes = [
            'background', 'person', 'bicycle', 'car', 'motorcycle',
            'bus', 'truck', 'traffic light', 'stop sign'
        ]
    
    def detect(self, image):
        """Detect objects in image"""
        transform = transforms.ToTensor()
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        results = []
        for box, label, score in zip(
            predictions[0]['boxes'],
            predictions[0]['labels'],
            predictions[0]['scores']
        ):
            if score > 0.5:
                results.append({
                    'class': self.classes[label] if label < len(self.classes) else 'unknown',
                    'confidence': score.item(),
                    'bbox': box.tolist()
                })
        
        return results

# ============================================
# Education: AI Tutor
# ============================================

class AITutor:
    """
    Personalized AI tutoring system
    
    Features:
    - Explains concepts at student's level
    - Generates practice problems
    - Provides feedback on answers
    - Tracks progress
    """
    
    def __init__(self):
        self.llm = OpenAI(temperature=0.7)
        
        self.explain_template = PromptTemplate(
            input_variables=["topic", "level", "style"],
            template="""You are a friendly tutor. Explain {topic} to a {level} student.
Use {style} to make it engaging. Include examples and check for understanding."""
        )
        
        self.problem_template = PromptTemplate(
            input_variables=["topic", "difficulty"],
            template="""Create a {difficulty} practice problem about {topic}.
Include the problem, hints, and solution."""
        )
    
    def explain(self, topic, level="middle school", style="analogies"):
        chain = LLMChain(llm=self.llm, prompt=self.explain_template)
        return chain.run(topic=topic, level=level, style=style)
    
    def generate_problem(self, topic, difficulty="medium"):
        chain = LLMChain(llm=self.llm, prompt=self.problem_template)
        return chain.run(topic=topic, difficulty=difficulty)
    
    def check_answer(self, problem, student_answer, correct_answer):
        prompt = f"""
        Problem: {problem}
        Student's answer: {student_answer}
        Correct answer: {correct_answer}
        
        Provide feedback: Is the answer correct? If not, explain the mistake
        and guide the student toward the right answer without giving it away.
        """
        return self.llm(prompt)

# ============================================
# E-commerce: Recommendation System
# ============================================

class RecommendationEngine:
    """
    Product recommendation system
    
    Methods:
    - Collaborative filtering (users who bought X also bought Y)
    - Content-based (similar products)
    - Hybrid approaches
    """
    
    def __init__(self, num_users, num_items, embedding_dim=50):
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
    
    def predict(self, user_id, item_id):
        """Predict user's rating for an item"""
        user_emb = self.user_embeddings(torch.tensor(user_id))
        item_emb = self.item_embeddings(torch.tensor(item_id))
        return torch.dot(user_emb, item_emb).item()
    
    def recommend(self, user_id, num_recommendations=10):
        """Get top recommendations for a user"""
        user_emb = self.user_embeddings(torch.tensor(user_id))
        
        # Score all items
        all_items = torch.arange(self.item_embeddings.num_embeddings)
        item_embs = self.item_embeddings(all_items)
        scores = torch.matmul(item_embs, user_emb)
        
        # Get top items
        top_indices = torch.topk(scores, num_recommendations).indices
        return top_indices.tolist()

# ============================================
# Creative: AI Art Generator
# ============================================

from diffusers import StableDiffusionPipeline

class AIArtist:
    """
    Generate art from text descriptions
    """
    
    def __init__(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        ).to("cuda")
    
    def create(self, prompt, style=None, negative_prompt=None):
        """Generate image from text prompt"""
        if style:
            prompt = f"{prompt}, {style} style"
        
        image = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            guidance_scale=7.5
        ).images[0]
        
        return image

# ============================================
# Customer Service: AI Chatbot
# ============================================

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

class CustomerServiceBot:
    """
    AI-powered customer service chatbot
    
    Features:
    - Answer FAQs
    - Handle complaints
    - Process orders
    - Escalate to humans when needed
    """
    
    def __init__(self, company_info):
        self.llm = OpenAI(temperature=0.3)
        self.memory = ConversationBufferMemory()
        
        self.system_prompt = f"""You are a helpful customer service agent for {company_info}.
Be polite, helpful, and concise. If you can't help, offer to connect with a human agent."""
        
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory
        )
    
    def chat(self, user_message):
        response = self.conversation.predict(input=user_message)
        
        # Check if escalation needed
        if "human agent" in response.lower() or "cannot help" in response.lower():
            return {
                'response': response,
                'escalate': True
            }
        
        return {'response': response, 'escalate': False}

# ============================================
# Summary: Choosing the Right Approach
# ============================================

"""
| Domain          | Primary Techniques           | Key Frameworks        |
|-----------------|-----------------------------|-----------------------|
| Healthcare      | CNN, Transfer Learning      | PyTorch, MONAI        |
| Finance         | Anomaly Detection, NLP      | Scikit-learn, XGBoost |
| Autonomous      | Object Detection, RL        | PyTorch, ROS          |
| Education       | LLMs, RAG                   | LangChain, GPT-4      |
| E-commerce      | Recommendations, NLP        | TensorFlow, PyTorch   |
| Creative        | Diffusion, GANs             | Stable Diffusion      |
| Manufacturing   | Computer Vision, RL         | TensorFlow, OpenCV    |
"""`

  return (
    <div className="space-y-8">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-white mb-4">AI Applications</h1>
        <p className="text-xl text-purple-200">Real-world AI transforming every industry!</p>
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
                <Globe className="w-6 h-6 text-sky-400" />
                AI is Everywhere!
              </CardTitle>
            </CardHeader>
            <CardContent className="text-slate-300 space-y-4">
              <p className="text-lg">
                <strong className="text-sky-400">AI Applications</strong> are transforming every industry 
                imaginable - from healthcare to entertainment, finance to education. Let's explore how AI 
                is being used in the real world!
              </p>
              <div className="bg-slate-900/50 p-4 rounded-lg border border-sky-500/30">
                <h4 className="text-sky-400 font-semibold mb-2">Think of it like this:</h4>
                <p>AI is like electricity was 100 years ago - it's becoming essential infrastructure that 
                powers everything. Just as electricity transformed factories, homes, and cities, AI is 
                transforming how we work, learn, and live!</p>
              </div>
            </CardContent>
          </Card>

          <div className="grid md:grid-cols-3 gap-4">
            {domains.map((domain, i) => (
              <Card key={i} className="bg-slate-800/50 border-slate-700 hover:border-slate-500 transition-all cursor-pointer"
                onClick={() => setSelectedDomain(selectedDomain === domain.name ? null : domain.name)}>
                <CardHeader className="pb-2">
                  <div className={`w-12 h-12 ${domain.color} rounded-lg flex items-center justify-center mb-2`}>
                    <domain.icon className="w-6 h-6 text-white" />
                  </div>
                  <CardTitle className="text-white text-lg">{domain.name}</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-1">
                    {domain.applications.slice(0, 2).map((app, j) => (
                      <p key={j} className="text-slate-400 text-sm">• {app.name}</p>
                    ))}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">AI Impact by Numbers</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-4 gap-4">
                {[
                  { stat: '$15.7T', desc: 'AI contribution to global economy by 2030', color: 'text-green-400' },
                  { stat: '97M', desc: 'New jobs created by AI by 2025', color: 'text-blue-400' },
                  { stat: '40%', desc: 'Productivity increase with AI adoption', color: 'text-purple-400' },
                  { stat: '85%', desc: 'Customer interactions handled by AI by 2025', color: 'text-orange-400' },
                ].map((item, i) => (
                  <div key={i} className="bg-slate-900 p-4 rounded-lg text-center">
                    <p className={`text-3xl font-bold ${item.color}`}>{item.stat}</p>
                    <p className="text-slate-400 text-xs mt-1">{item.desc}</p>
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
                <Globe className="w-5 h-5 text-sky-400" />
                Explore AI Applications by Domain
              </CardTitle>
              <CardDescription className="text-slate-400">
                Click on a domain to see detailed applications!
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex flex-wrap gap-2">
                {domains.map((domain, i) => (
                  <Button
                    key={i}
                    variant={selectedDomain === domain.name ? 'default' : 'outline'}
                    className={selectedDomain === domain.name ? domain.color : ''}
                    onClick={() => setSelectedDomain(selectedDomain === domain.name ? null : domain.name)}
                  >
                    <domain.icon className="w-4 h-4 mr-2" />
                    {domain.name}
                  </Button>
                ))}
              </div>

              {selectedDomain && (
                <div className="space-y-4 animate-in fade-in">
                  {domains.filter(d => d.name === selectedDomain).map((domain, i) => (
                    <div key={i}>
                      <div className="grid md:grid-cols-2 gap-4 mb-4">
                        {domain.applications.map((app, j) => (
                          <div key={j} className="bg-slate-900 p-4 rounded-lg">
                            <h4 className="text-white font-semibold">{app.name}</h4>
                            <p className="text-slate-400 text-sm">{app.desc}</p>
                          </div>
                        ))}
                      </div>
                      
                      <div className="bg-slate-900 p-4 rounded-lg">
                        <div className="flex flex-wrap gap-4">
                          <div>
                            <p className="text-slate-400 text-sm mb-2">Popular Frameworks:</p>
                            <div className="flex gap-2">
                              {domain.frameworks.map((fw, j) => (
                                <Badge key={j} className="bg-sky-500">{fw}</Badge>
                              ))}
                            </div>
                          </div>
                          <div className="flex-1">
                            <p className="text-slate-400 text-sm mb-2">Real Example:</p>
                            <p className="text-sky-300 text-sm">{domain.example}</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">AI Technology Stack</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {[
                  { layer: 'Applications', items: ['Chatbots', 'Self-driving', 'Medical AI', 'Art Generation'], color: 'bg-sky-500' },
                  { layer: 'Models', items: ['GPT-4', 'BERT', 'ResNet', 'Stable Diffusion'], color: 'bg-purple-500' },
                  { layer: 'Frameworks', items: ['PyTorch', 'TensorFlow', 'LangChain', 'Hugging Face'], color: 'bg-green-500' },
                  { layer: 'Infrastructure', items: ['GPUs', 'Cloud', 'Databases', 'APIs'], color: 'bg-orange-500' },
                ].map((layer, i) => (
                  <div key={i} className={`${layer.color} p-4 rounded-lg`}>
                    <p className="text-white font-semibold mb-2">{layer.layer}</p>
                    <div className="flex flex-wrap gap-2">
                      {layer.items.map((item, j) => (
                        <Badge key={j} variant="secondary" className="bg-white/20 text-white">{item}</Badge>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Your AI Learning Journey</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="relative">
                <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-sky-500" />
                <div className="space-y-6 pl-10">
                  {[
                    { step: '1. Foundations', desc: 'Python, Math, ML basics', done: true },
                    { step: '2. Deep Learning', desc: 'Neural networks, CNNs, RNNs', done: true },
                    { step: '3. Specialization', desc: 'NLP, Computer Vision, or RL', done: true },
                    { step: '4. GenAI', desc: 'LLMs, Transformers, Diffusion', done: true },
                    { step: '5. Applications', desc: 'Build real projects!', done: true },
                    { step: '6. Keep Learning!', desc: 'AI evolves rapidly', done: false },
                  ].map((item, i) => (
                    <div key={i} className="relative">
                      <div className={`absolute -left-10 w-6 h-6 rounded-full flex items-center justify-center ${item.done ? 'bg-sky-500' : 'bg-slate-600'}`}>
                        {item.done ? '✓' : i + 1}
                      </div>
                      <div>
                        <p className="text-white font-semibold">{item.step}</p>
                        <p className="text-slate-400 text-sm">{item.desc}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="code" className="space-y-6">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Code className="w-5 h-5 text-sky-400" />
                Python Code: Real-World AI Applications
              </CardTitle>
              <CardDescription className="text-slate-400">
                Code examples from healthcare to e-commerce!
              </CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="bg-slate-900 p-4 rounded-lg overflow-x-auto text-sm max-h-96 overflow-y-auto">
                <code className="text-green-400">{pythonCode}</code>
              </pre>
              <div className="mt-4 p-4 bg-sky-500/10 rounded-lg border border-sky-500/30">
                <h4 className="text-sky-400 font-semibold mb-2">Required Libraries:</h4>
                <code className="text-slate-300 text-sm bg-slate-800 px-2 py-1 rounded">
                  pip install torch torchvision transformers langchain scikit-learn diffusers
                </code>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-r from-sky-500 to-purple-500 border-none">
            <CardContent className="py-8 text-center">
              <h3 className="text-2xl font-bold text-white mb-2">Congratulations! 🎉</h3>
              <p className="text-white/90">
                You've completed the GenAI Tutorial! You now have a solid foundation in AI, 
                from basic concepts to cutting-edge applications. Keep learning and building!
              </p>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default Module16Applications
