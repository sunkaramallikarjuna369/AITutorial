import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { MessageSquare, Search, Code, Smile, Frown, Meh } from 'lucide-react'

const Module05NLP = () => {
  const [inputText, setInputText] = useState('')
  const [tokens, setTokens] = useState<string[]>([])
  const [sentiment, setSentiment] = useState<string | null>(null)
  const [sentimentScore, setSentimentScore] = useState(0)

  const tokenize = () => {
    if (!inputText.trim()) return
    const words = inputText.toLowerCase().split(/\s+/).filter(w => w.length > 0)
    setTokens(words)
  }

  const analyzeSentiment = () => {
    if (!inputText.trim()) return
    
    const positiveWords = ['good', 'great', 'amazing', 'love', 'happy', 'excellent', 'wonderful', 'best', 'awesome', 'fantastic']
    const negativeWords = ['bad', 'terrible', 'hate', 'awful', 'worst', 'horrible', 'sad', 'angry', 'poor', 'disappointing']
    
    const words = inputText.toLowerCase().split(/\s+/)
    let score = 0
    
    words.forEach(word => {
      if (positiveWords.includes(word)) score += 1
      if (negativeWords.includes(word)) score -= 1
    })
    
    setSentimentScore(score)
    if (score > 0) setSentiment('positive')
    else if (score < 0) setSentiment('negative')
    else setSentiment('neutral')
  }

  const pythonCode = `# Natural Language Processing (NLP)
# Teaching computers to understand human language!

import re
from collections import Counter

# ============================================
# Text Preprocessing
# ============================================

def preprocess_text(text):
    """
    Clean and prepare text for analysis
    Like organizing your room before studying!
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Example
raw_text = "Hello! This is NLP 101. It's AMAZING!!!"
clean_text = preprocess_text(raw_text)
print(f"Original: {raw_text}")
print(f"Cleaned: {clean_text}")

# ============================================
# Tokenization
# ============================================

def tokenize(text):
    """
    Split text into individual words (tokens)
    Like breaking a sentence into Lego pieces!
    """
    return text.split()

def tokenize_sentences(text):
    """Split text into sentences"""
    return re.split(r'[.!?]+', text)

text = "I love learning NLP. It is so interesting!"
words = tokenize(text.lower())
sentences = tokenize_sentences(text)

print(f"Words: {words}")
print(f"Sentences: {sentences}")

# ============================================
# Stop Words Removal
# ============================================

STOP_WORDS = {'the', 'a', 'an', 'is', 'it', 'to', 'and', 'of', 'in', 'for', 'on', 'with'}

def remove_stop_words(tokens):
    """
    Remove common words that don't add much meaning
    Like removing 'um' and 'uh' from speech!
    """
    return [word for word in tokens if word not in STOP_WORDS]

tokens = ['the', 'cat', 'is', 'on', 'the', 'mat']
filtered = remove_stop_words(tokens)
print(f"Original: {tokens}")
print(f"Filtered: {filtered}")

# ============================================
# Bag of Words
# ============================================

def bag_of_words(documents):
    """
    Convert text to numbers by counting words
    Like counting how many of each candy you have!
    """
    # Build vocabulary
    vocab = set()
    for doc in documents:
        vocab.update(tokenize(doc.lower()))
    vocab = sorted(vocab)
    
    # Create vectors
    vectors = []
    for doc in documents:
        words = tokenize(doc.lower())
        word_counts = Counter(words)
        vector = [word_counts.get(word, 0) for word in vocab]
        vectors.append(vector)
    
    return vocab, vectors

docs = [
    "I love cats",
    "I love dogs",
    "Cats and dogs are pets"
]

vocab, vectors = bag_of_words(docs)
print(f"Vocabulary: {vocab}")
for i, vec in enumerate(vectors):
    print(f"Doc {i+1}: {vec}")

# ============================================
# Sentiment Analysis
# ============================================

class SimpleSentimentAnalyzer:
    """
    Determine if text is positive, negative, or neutral
    Like reading someone's mood from their message!
    """
    
    def __init__(self):
        self.positive_words = {
            'good', 'great', 'amazing', 'love', 'happy', 'excellent',
            'wonderful', 'best', 'awesome', 'fantastic', 'beautiful',
            'perfect', 'brilliant', 'outstanding', 'superb'
        }
        self.negative_words = {
            'bad', 'terrible', 'hate', 'awful', 'worst', 'horrible',
            'sad', 'angry', 'poor', 'disappointing', 'ugly', 'boring',
            'annoying', 'stupid', 'failure'
        }
    
    def analyze(self, text):
        words = text.lower().split()
        
        pos_count = sum(1 for w in words if w in self.positive_words)
        neg_count = sum(1 for w in words if w in self.negative_words)
        
        score = pos_count - neg_count
        
        if score > 0:
            return 'positive', score
        elif score < 0:
            return 'negative', score
        else:
            return 'neutral', score

analyzer = SimpleSentimentAnalyzer()

texts = [
    "This movie is amazing and wonderful!",
    "The food was terrible and disappointing.",
    "The weather is okay today."
]

for text in texts:
    sentiment, score = analyzer.analyze(text)
    print(f"'{text}' -> {sentiment} (score: {score})")

# ============================================
# Named Entity Recognition (Simple)
# ============================================

def simple_ner(text):
    """
    Find names, places, and organizations in text
    Like playing 'I Spy' with words!
    """
    # Simple rule: Capitalized words might be entities
    words = text.split()
    entities = []
    
    for i, word in enumerate(words):
        # Skip first word (always capitalized)
        if i == 0:
            continue
        # Check if word starts with capital
        if word[0].isupper():
            entities.append(word.strip('.,!?'))
    
    return entities

text = "John works at Google in New York. He met Sarah yesterday."
entities = simple_ner(text)
print(f"Entities found: {entities}")

# ============================================
# Using NLTK Library
# ============================================

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

text = "The cats are running quickly through the gardens."

# Tokenization
tokens = word_tokenize(text)
print(f"Tokens: {tokens}")

# Stop words removal
stop_words = set(stopwords.words('english'))
filtered = [w for w in tokens if w.lower() not in stop_words]
print(f"Filtered: {filtered}")

# Stemming (reduce to root form)
stemmer = PorterStemmer()
stemmed = [stemmer.stem(w) for w in filtered]
print(f"Stemmed: {stemmed}")

# Lemmatization (smarter root form)
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(w) for w in filtered]
print(f"Lemmatized: {lemmatized}")

# ============================================
# Using Hugging Face Transformers
# ============================================

from transformers import pipeline

# Sentiment Analysis
sentiment_pipeline = pipeline("sentiment-analysis")
result = sentiment_pipeline("I love learning about AI!")
print(f"Sentiment: {result}")

# Named Entity Recognition
ner_pipeline = pipeline("ner", grouped_entities=True)
result = ner_pipeline("Apple Inc. was founded by Steve Jobs in California.")
print(f"Entities: {result}")

# Text Classification
classifier = pipeline("zero-shot-classification")
result = classifier(
    "This is a tutorial about machine learning",
    candidate_labels=["education", "politics", "sports"]
)
print(f"Classification: {result}")`

  return (
    <div className="space-y-8">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-white mb-4">Natural Language Processing</h1>
        <p className="text-xl text-purple-200">Teaching computers to understand human language!</p>
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
                <MessageSquare className="w-6 h-6 text-pink-400" />
                What is NLP?
              </CardTitle>
            </CardHeader>
            <CardContent className="text-slate-300 space-y-4">
              <p className="text-lg">
                <strong className="text-pink-400">Natural Language Processing (NLP)</strong> is how we teach 
                computers to understand, interpret, and generate human language. It's the magic behind 
                chatbots, voice assistants, and translation apps!
              </p>
              <div className="bg-slate-900/50 p-4 rounded-lg border border-pink-500/30">
                <h4 className="text-pink-400 font-semibold mb-2">Think of it like this:</h4>
                <p>When you read a book, your brain automatically understands the words, their meanings, 
                and the emotions behind them. NLP teaches computers to do the same thing - read text and 
                understand what it means!</p>
              </div>
            </CardContent>
          </Card>

          <div className="grid md:grid-cols-2 gap-4">
            {[
              { title: 'Tokenization', desc: 'Breaking text into words or sentences', icon: '🔤', example: '"Hello world" → ["Hello", "world"]' },
              { title: 'Sentiment Analysis', desc: 'Understanding emotions in text', icon: '😊', example: '"I love this!" → Positive' },
              { title: 'Named Entity Recognition', desc: 'Finding names, places, organizations', icon: '🏷️', example: '"John works at Google" → Person: John, Org: Google' },
              { title: 'Text Classification', desc: 'Categorizing text into groups', icon: '📁', example: 'Email → Spam or Not Spam' },
              { title: 'Machine Translation', desc: 'Converting between languages', icon: '🌍', example: '"Hello" → "Hola" (Spanish)' },
              { title: 'Question Answering', desc: 'Finding answers in text', icon: '❓', example: 'Q: "What color is the sky?" A: "Blue"' },
            ].map((item, i) => (
              <Card key={i} className="bg-slate-800/50 border-slate-700">
                <CardHeader className="pb-2">
                  <div className="flex items-center gap-3">
                    <span className="text-2xl">{item.icon}</span>
                    <CardTitle className="text-white text-lg">{item.title}</CardTitle>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-slate-400 text-sm mb-2">{item.desc}</p>
                  <code className="text-xs bg-slate-900 px-2 py-1 rounded text-pink-300">{item.example}</code>
                </CardContent>
              </Card>
            ))}
          </div>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">The NLP Pipeline</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap justify-center gap-2">
                {['Raw Text', 'Tokenize', 'Clean', 'Analyze', 'Understand', 'Respond'].map((step, i) => (
                  <div key={i} className="flex items-center">
                    <div className="bg-gradient-to-r from-pink-500 to-purple-500 text-white px-4 py-2 rounded-lg font-medium text-sm">
                      {step}
                    </div>
                    {i < 5 && <span className="text-pink-400 mx-2">→</span>}
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
                <Search className="w-5 h-5 text-pink-400" />
                Interactive Tokenizer
              </CardTitle>
              <CardDescription className="text-slate-400">
                See how text is broken into tokens!
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="Type a sentence... (e.g., 'I love learning NLP!')"
                  className="flex-1 px-4 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-pink-500"
                />
                <Button onClick={tokenize} className="bg-pink-600 hover:bg-pink-700">
                  Tokenize
                </Button>
              </div>
              
              {tokens.length > 0 && (
                <div className="p-4 bg-slate-900 rounded-lg">
                  <p className="text-slate-400 text-sm mb-2">Tokens ({tokens.length} words):</p>
                  <div className="flex flex-wrap gap-2">
                    {tokens.map((token, i) => (
                      <Badge key={i} className="bg-pink-500/20 text-pink-300 border border-pink-500/30">
                        {token}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Smile className="w-5 h-5 text-yellow-400" />
                Sentiment Analyzer
              </CardTitle>
              <CardDescription className="text-slate-400">
                Detect emotions in your text!
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="Type something with emotion... (e.g., 'This is amazing!')"
                  className="flex-1 px-4 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-yellow-500"
                />
                <Button onClick={analyzeSentiment} className="bg-yellow-600 hover:bg-yellow-700">
                  Analyze
                </Button>
              </div>
              
              {sentiment && (
                <div className={`p-4 rounded-lg border ${
                  sentiment === 'positive' ? 'bg-green-500/10 border-green-500/30' :
                  sentiment === 'negative' ? 'bg-red-500/10 border-red-500/30' :
                  'bg-gray-500/10 border-gray-500/30'
                }`}>
                  <div className="flex items-center gap-3">
                    {sentiment === 'positive' && <Smile className="w-8 h-8 text-green-400" />}
                    {sentiment === 'negative' && <Frown className="w-8 h-8 text-red-400" />}
                    {sentiment === 'neutral' && <Meh className="w-8 h-8 text-gray-400" />}
                    <div>
                      <p className={`font-semibold capitalize ${
                        sentiment === 'positive' ? 'text-green-400' :
                        sentiment === 'negative' ? 'text-red-400' :
                        'text-gray-400'
                      }`}>
                        {sentiment} Sentiment
                      </p>
                      <p className="text-slate-400 text-sm">Score: {sentimentScore}</p>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Word Embeddings Visualization</CardTitle>
              <CardDescription className="text-slate-400">
                How AI represents words as numbers in space
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="relative h-64 bg-slate-900 rounded-lg p-4">
                <svg className="w-full h-full" viewBox="0 0 400 200">
                  {/* Axes */}
                  <line x1="50" y1="180" x2="380" y2="180" stroke="#4b5563" strokeWidth="1" />
                  <line x1="50" y1="20" x2="50" y2="180" stroke="#4b5563" strokeWidth="1" />
                  
                  {/* Word points */}
                  <circle cx="100" cy="60" r="8" fill="#22c55e" />
                  <text x="100" y="45" textAnchor="middle" fill="#22c55e" fontSize="12">king</text>
                  
                  <circle cx="150" cy="80" r="8" fill="#22c55e" />
                  <text x="150" y="65" textAnchor="middle" fill="#22c55e" fontSize="12">queen</text>
                  
                  <circle cx="120" cy="140" r="8" fill="#3b82f6" />
                  <text x="120" y="160" textAnchor="middle" fill="#3b82f6" fontSize="12">man</text>
                  
                  <circle cx="170" cy="150" r="8" fill="#3b82f6" />
                  <text x="170" y="170" textAnchor="middle" fill="#3b82f6" fontSize="12">woman</text>
                  
                  <circle cx="280" cy="100" r="8" fill="#ef4444" />
                  <text x="280" y="85" textAnchor="middle" fill="#ef4444" fontSize="12">cat</text>
                  
                  <circle cx="320" cy="110" r="8" fill="#ef4444" />
                  <text x="320" y="95" textAnchor="middle" fill="#ef4444" fontSize="12">dog</text>
                  
                  {/* Relationship arrows */}
                  <line x1="100" y1="60" x2="150" y2="80" stroke="#8b5cf6" strokeWidth="1" strokeDasharray="4" />
                  <line x1="120" y1="140" x2="170" y2="150" stroke="#8b5cf6" strokeWidth="1" strokeDasharray="4" />
                  
                  {/* Legend */}
                  <text x="250" y="170" fill="#94a3b8" fontSize="10">Similar words are close together!</text>
                </svg>
              </div>
              <p className="text-slate-400 text-sm mt-2 text-center">
                Word embeddings place similar words close together in a mathematical space. 
                "King - Man + Woman = Queen" actually works!
              </p>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="code" className="space-y-6">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Code className="w-5 h-5 text-pink-400" />
                Python Code: NLP Fundamentals
              </CardTitle>
              <CardDescription className="text-slate-400">
                From basic text processing to advanced NLP with transformers!
              </CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="bg-slate-900 p-4 rounded-lg overflow-x-auto text-sm max-h-96 overflow-y-auto">
                <code className="text-green-400">{pythonCode}</code>
              </pre>
              <div className="mt-4 p-4 bg-pink-500/10 rounded-lg border border-pink-500/30">
                <h4 className="text-pink-400 font-semibold mb-2">Required Libraries:</h4>
                <code className="text-slate-300 text-sm bg-slate-800 px-2 py-1 rounded">
                  pip install nltk transformers torch
                </code>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default Module05NLP
