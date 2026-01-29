"""
Module 08: Transformers & Attention
===================================
The architecture behind ChatGPT, BERT, and modern AI!
"""

import math

def explain_transformers():
    print("=" * 60)
    print("Transformers & Attention Mechanism")
    print("=" * 60)
    print("""
Transformers revolutionized AI in 2017 with "Attention Is All You Need"!

Key Innovation: ATTENTION
- Instead of processing words one by one (like RNNs)
- Look at ALL words at once and decide which are important

Example: "The cat sat on the mat because it was tired"
- What does "it" refer to? The cat!
- Attention helps the model understand this connection

Components:
1. Self-Attention: Each word looks at all other words
2. Multi-Head Attention: Multiple attention patterns
3. Feed-Forward Networks: Process attention outputs
4. Positional Encoding: Remember word order

Architecture:
    Input → Embedding → [Encoder x N] → [Decoder x N] → Output
    
Famous Transformers:
- BERT: Encoder-only (understanding)
- GPT: Decoder-only (generation)
- T5: Encoder-Decoder (both)
    """)

code_examples = '''
"""
Transformer Implementation Examples
===================================
"""

import torch
import torch.nn as nn
import math

# ============================================
# 1. Scaled Dot-Product Attention
# ============================================

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """
    d_k = query.size(-1)
    
    # Calculate attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask (optional)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Softmax to get attention weights
    attention_weights = torch.softmax(scores, dim=-1)
    
    # Apply attention to values
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights


# ============================================
# 2. Multi-Head Attention
# ============================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(attn_output)


# ============================================
# 3. Positional Encoding
# ============================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ============================================
# 4. Transformer Encoder Layer
# ============================================

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


# ============================================
# 5. Complete Transformer
# ============================================

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        return self.fc_out(x)


# ============================================
# 6. Using Pre-trained Transformers
# ============================================

from transformers import BertModel, BertTokenizer

def use_bert(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    
    # Get embeddings
    last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden]
    pooler_output = outputs.pooler_output  # [batch, hidden]
    
    return pooler_output

print("Transformer examples loaded!")
'''

def main():
    explain_transformers()
    print("\nPython Code Examples:")
    print(code_examples)

if __name__ == "__main__":
    main()
