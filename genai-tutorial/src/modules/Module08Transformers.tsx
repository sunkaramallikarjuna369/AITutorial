import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Zap, Eye, Code, ArrowRight } from 'lucide-react'

const Module08Transformers = () => {
  const [attentionWeights, setAttentionWeights] = useState<number[][]>([])
  const [selectedWord, setSelectedWord] = useState<number | null>(null)
  
  const sentence = ['The', 'cat', 'sat', 'on', 'the', 'mat']
  
  useEffect(() => {
    const weights = sentence.map(() => 
      sentence.map(() => Math.random())
    )
    weights.forEach((row, i) => {
      const sum = row.reduce((a, b) => a + b, 0)
      weights[i] = row.map(w => w / sum)
    })
    setAttentionWeights(weights)
  }, [])

  const pythonCode = `# Transformers & Attention Mechanisms
# The architecture behind modern AI breakthroughs!

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================
# Self-Attention Mechanism
# ============================================

class SelfAttention(nn.Module):
    """
    Self-Attention: Each word looks at all other words
    to understand context!
    
    Like reading a sentence and understanding how
    each word relates to every other word.
    """
    
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Query, Key, Value projections
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Create Q, K, V
        Q = self.query(x)  # What am I looking for?
        K = self.key(x)    # What do I contain?
        V = self.value(x)  # What information do I have?
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores: how much should each word attend to others?
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Softmax to get attention weights (probabilities)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Reshape and project output
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out(attended)
        
        return output, attention_weights

# ============================================
# Positional Encoding
# ============================================

class PositionalEncoding(nn.Module):
    """
    Add position information to embeddings
    
    Since attention doesn't know word order,
    we add special position signals!
    """
    
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        
        # Create position encodings
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * 
            (-math.log(10000.0) / embed_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ============================================
# Transformer Encoder Block
# ============================================

class TransformerBlock(nn.Module):
    """
    One block of a Transformer
    
    Contains:
    1. Self-Attention (look at context)
    2. Feed-Forward Network (process information)
    3. Layer Normalization (keep values stable)
    4. Residual Connections (help gradients flow)
    """
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        
        self.attention = SelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention with residual connection
        attended, weights = self.attention(self.norm1(x))
        x = x + self.dropout(attended)
        
        # Feed-forward with residual connection
        x = x + self.feed_forward(self.norm2(x))
        
        return x, weights

# ============================================
# Complete Transformer Encoder
# ============================================

class TransformerEncoder(nn.Module):
    """
    Stack of Transformer blocks for encoding text
    """
    
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, 
                 num_layers=6, ff_dim=2048, max_len=512):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len)
        
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # Embed tokens and add positions
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        # Pass through transformer blocks
        all_weights = []
        for layer in self.layers:
            x, weights = layer(x)
            all_weights.append(weights)
        
        return self.norm(x), all_weights

# ============================================
# Simple Transformer for Classification
# ============================================

class TransformerClassifier(nn.Module):
    """
    Transformer for text classification
    Like sentiment analysis or spam detection!
    """
    
    def __init__(self, vocab_size, num_classes, embed_dim=256, 
                 num_heads=4, num_layers=4):
        super().__init__()
        
        self.encoder = TransformerEncoder(
            vocab_size, embed_dim, num_heads, num_layers
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes)
        )
    
    def forward(self, x):
        # Encode the sequence
        encoded, _ = self.encoder(x)
        
        # Use [CLS] token or mean pooling
        pooled = encoded.mean(dim=1)
        
        # Classify
        return self.classifier(pooled)

# ============================================
# Using Hugging Face Transformers
# ============================================

from transformers import (
    BertTokenizer, BertModel,
    GPT2Tokenizer, GPT2LMHeadModel,
    AutoTokenizer, AutoModel
)

# BERT - Bidirectional Encoder
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(text):
    """Get contextual embeddings from BERT"""
    inputs = tokenizer(text, return_tensors='pt', padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Last hidden state contains embeddings
    embeddings = outputs.last_hidden_state
    
    # Get attention weights
    attention = outputs.attentions  # If output_attentions=True
    
    return embeddings

# Example
text = "The cat sat on the mat"
embeddings = get_bert_embeddings(text)
print(f"Embedding shape: {embeddings.shape}")

# ============================================
# Visualizing Attention
# ============================================

def visualize_attention(text, model, tokenizer):
    """
    See what the model pays attention to!
    """
    inputs = tokenizer(text, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Get attention from last layer
    attention = outputs.attentions[-1]  # [batch, heads, seq, seq]
    
    # Average over heads
    avg_attention = attention.mean(dim=1).squeeze()
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    print("Attention Matrix:")
    print("Tokens:", tokens)
    print(avg_attention.numpy())
    
    return avg_attention, tokens

# ============================================
# Different Transformer Architectures
# ============================================

# Encoder-only (BERT): Good for understanding
# - Text classification
# - Named entity recognition
# - Question answering

# Decoder-only (GPT): Good for generation
# - Text generation
# - Code completion
# - Chatbots

# Encoder-Decoder (T5, BART): Good for transformation
# - Translation
# - Summarization
# - Question answering

from transformers import T5ForConditionalGeneration, T5Tokenizer

t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')

def summarize_text(text):
    """Use T5 to summarize text"""
    input_text = f"summarize: {text}"
    inputs = t5_tokenizer(input_text, return_tensors='pt', max_length=512)
    
    outputs = t5_model.generate(
        inputs['input_ids'],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )
    
    summary = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary`

  return (
    <div className="space-y-8">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-white mb-4">Transformers & Attention</h1>
        <p className="text-xl text-purple-200">The revolutionary architecture behind modern AI!</p>
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
                <Zap className="w-6 h-6 text-red-400" />
                What are Transformers?
              </CardTitle>
            </CardHeader>
            <CardContent className="text-slate-300 space-y-4">
              <p className="text-lg">
                <strong className="text-red-400">Transformers</strong> are the architecture that powers 
                ChatGPT, BERT, and most modern AI! They use a special mechanism called "attention" that 
                lets the model focus on the most important parts of the input.
              </p>
              <div className="bg-slate-900/50 p-4 rounded-lg border border-red-500/30">
                <h4 className="text-red-400 font-semibold mb-2">Think of it like this:</h4>
                <p>When you read "The cat sat on the mat because it was tired", you know "it" refers to 
                "cat" not "mat". Attention helps AI make these connections by letting each word "look at" 
                all other words to understand context!</p>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Eye className="w-5 h-5 text-blue-400" />
                The Attention Mechanism
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-slate-300">
                Attention answers the question: "When processing this word, how much should I focus on each other word?"
              </p>
              <div className="grid md:grid-cols-3 gap-4">
                <div className="bg-blue-500/10 p-4 rounded-lg border border-blue-500/30">
                  <h4 className="text-blue-400 font-semibold mb-2">Query (Q)</h4>
                  <p className="text-slate-400 text-sm">"What am I looking for?"</p>
                  <p className="text-xs text-slate-500 mt-2">Each word asks a question about what context it needs</p>
                </div>
                <div className="bg-green-500/10 p-4 rounded-lg border border-green-500/30">
                  <h4 className="text-green-400 font-semibold mb-2">Key (K)</h4>
                  <p className="text-slate-400 text-sm">"What do I contain?"</p>
                  <p className="text-xs text-slate-500 mt-2">Each word advertises what information it has</p>
                </div>
                <div className="bg-purple-500/10 p-4 rounded-lg border border-purple-500/30">
                  <h4 className="text-purple-400 font-semibold mb-2">Value (V)</h4>
                  <p className="text-slate-400 text-sm">"Here's my information"</p>
                  <p className="text-xs text-slate-500 mt-2">The actual content that gets passed along</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Transformer Architecture</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col items-center space-y-2">
                {[
                  { name: 'Input Embedding', desc: 'Convert words to numbers', color: 'bg-blue-500' },
                  { name: 'Positional Encoding', desc: 'Add position information', color: 'bg-cyan-500' },
                  { name: 'Multi-Head Attention', desc: 'Look at context from multiple angles', color: 'bg-purple-500' },
                  { name: 'Add & Normalize', desc: 'Stabilize the values', color: 'bg-green-500' },
                  { name: 'Feed Forward', desc: 'Process the information', color: 'bg-orange-500' },
                  { name: 'Output', desc: 'Final predictions', color: 'bg-red-500' },
                ].map((layer, i) => (
                  <div key={i} className="w-full max-w-md">
                    <div className={`${layer.color} text-white p-3 rounded-lg text-center`}>
                      <p className="font-semibold">{layer.name}</p>
                      <p className="text-xs opacity-80">{layer.desc}</p>
                    </div>
                    {i < 5 && <div className="flex justify-center py-1"><ArrowRight className="text-slate-500 rotate-90 w-5 h-5" /></div>}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Famous Transformer Models</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-4">
                {[
                  { name: 'BERT', type: 'Encoder', use: 'Understanding text', examples: 'Search, Q&A', color: 'bg-blue-500' },
                  { name: 'GPT', type: 'Decoder', use: 'Generating text', examples: 'ChatGPT, Writing', color: 'bg-green-500' },
                  { name: 'T5', type: 'Encoder-Decoder', use: 'Text transformation', examples: 'Translation, Summary', color: 'bg-purple-500' },
                ].map((model, i) => (
                  <div key={i} className="bg-slate-900 p-4 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <div className={`w-8 h-8 ${model.color} rounded flex items-center justify-center text-white font-bold text-sm`}>
                        {model.name[0]}
                      </div>
                      <h4 className="text-white font-semibold">{model.name}</h4>
                    </div>
                    <Badge variant="outline" className="text-slate-400 mb-2">{model.type}</Badge>
                    <p className="text-slate-400 text-sm">{model.use}</p>
                    <p className="text-slate-500 text-xs mt-1">{model.examples}</p>
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
                <Eye className="w-5 h-5 text-red-400" />
                Interactive Attention Visualization
              </CardTitle>
              <CardDescription className="text-slate-400">
                Click on a word to see what it pays attention to!
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex flex-wrap gap-2 justify-center">
                {sentence.map((word, i) => (
                  <Button
                    key={i}
                    variant={selectedWord === i ? 'default' : 'outline'}
                    className={selectedWord === i ? 'bg-red-500' : ''}
                    onClick={() => setSelectedWord(selectedWord === i ? null : i)}
                  >
                    {word}
                  </Button>
                ))}
              </div>

              {selectedWord !== null && attentionWeights.length > 0 && (
                <div className="bg-slate-900 p-4 rounded-lg">
                  <p className="text-slate-400 text-sm mb-3">
                    "{sentence[selectedWord]}" pays attention to:
                  </p>
                  <div className="space-y-2">
                    {sentence.map((word, i) => {
                      const weight = attentionWeights[selectedWord]?.[i] || 0
                      return (
                        <div key={i} className="flex items-center gap-3">
                          <span className="text-white w-16">{word}</span>
                          <div className="flex-1 bg-slate-700 rounded-full h-4 overflow-hidden">
                            <div 
                              className="bg-gradient-to-r from-red-500 to-orange-500 h-full transition-all duration-500"
                              style={{ width: `${weight * 100}%` }}
                            />
                          </div>
                          <span className="text-slate-400 text-sm w-12">{(weight * 100).toFixed(1)}%</span>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Attention Matrix Heatmap</CardTitle>
              <CardDescription className="text-slate-400">
                Brighter = more attention between words
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <div className="inline-block">
                  <div className="flex">
                    <div className="w-12" />
                    {sentence.map((word, i) => (
                      <div key={i} className="w-12 text-center text-slate-400 text-xs truncate">{word}</div>
                    ))}
                  </div>
                  {sentence.map((word, i) => (
                    <div key={i} className="flex">
                      <div className="w-12 text-slate-400 text-xs flex items-center">{word}</div>
                      {sentence.map((_, j) => {
                        const weight = attentionWeights[i]?.[j] || 0
                        return (
                          <div 
                            key={j}
                            className="w-12 h-12 m-0.5 rounded flex items-center justify-center text-xs"
                            style={{ 
                              backgroundColor: `rgba(239, 68, 68, ${weight})`,
                              color: weight > 0.5 ? 'white' : '#94a3b8'
                            }}
                          >
                            {(weight * 100).toFixed(0)}
                          </div>
                        )
                      })}
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Multi-Head Attention</CardTitle>
              <CardDescription className="text-slate-400">
                Multiple attention "heads" look at different aspects
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-4 gap-4">
                {['Syntax', 'Semantics', 'Position', 'Context'].map((head, i) => (
                  <div key={i} className="bg-slate-900 p-3 rounded-lg text-center">
                    <div className={`w-10 h-10 mx-auto rounded-full mb-2 flex items-center justify-center ${
                      ['bg-blue-500', 'bg-green-500', 'bg-purple-500', 'bg-orange-500'][i]
                    }`}>
                      <span className="text-white font-bold">{i + 1}</span>
                    </div>
                    <p className="text-white text-sm font-medium">Head {i + 1}</p>
                    <p className="text-slate-400 text-xs">{head}</p>
                  </div>
                ))}
              </div>
              <p className="text-slate-400 text-sm mt-4 text-center">
                Each head learns to focus on different relationships between words!
              </p>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="code" className="space-y-6">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Code className="w-5 h-5 text-red-400" />
                Python Code: Transformers & Attention
              </CardTitle>
              <CardDescription className="text-slate-400">
                Build transformers from scratch and use Hugging Face!
              </CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="bg-slate-900 p-4 rounded-lg overflow-x-auto text-sm max-h-96 overflow-y-auto">
                <code className="text-green-400">{pythonCode}</code>
              </pre>
              <div className="mt-4 p-4 bg-red-500/10 rounded-lg border border-red-500/30">
                <h4 className="text-red-400 font-semibold mb-2">Required Libraries:</h4>
                <code className="text-slate-300 text-sm bg-slate-800 px-2 py-1 rounded">
                  pip install torch transformers
                </code>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default Module08Transformers
