import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import create_padding_mask
class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_seq_len=1000):
        super(PositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

        # Create constant 'pe' matrix with values dependent on
        # position and embedding dimension
        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_seq_len, embedding_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, embedding_dim]
        x = x + self.pe[:, :x.size(1)]
        return x

class FFNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(FFNN, self).__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.d_k = embedding_dim // num_heads
        self.W_q = nn.Linear(embedding_dim, embedding_dim)
        self.W_k = nn.Linear(embedding_dim, embedding_dim)
        self.W_v = nn.Linear(embedding_dim, embedding_dim)
        self.W_o = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.ffnn = FFNN(embedding_dim, hidden_dim)

    def forward(self, x, mask=None):
        residual = x
        batch_size, seq_len, _ = x.size()
        Q = self.W_q(x) 
        K = self.W_k(x)
        V = self.W_v(x)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)  
        context = torch.matmul(attn, V)  # [batch_size, num_heads, seq_len, d_k]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)  # [batch_size, seq_len, embedding_dim]
        output = self.W_o(context)  # [batch_size, seq_len, embedding_dim]
        output = self.dropout(output)  
        x = self.layer_norm1(residual + output)  
        residual = x
        output = self.ffnn(x)  # [batch_size, seq_len, embedding_dim]
        output = self.dropout(output) 
        x = self.layer_norm2(residual + output) 

        return x

class Encoder(nn.Module):
    def __init__(self, num_layers, embedding_dim, vocab_size, hidden_dim, num_heads):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = PositionalEmbedding(embedding_dim)
        self.layers = nn.ModuleList([
            EncoderLayer(embedding_dim, hidden_dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, texts, src_mask=None):
        x = self.embedding(texts)
        x = self.position_embedding(x)
        for layer in self.layers:
            x = layer(x, mask=src_mask)
        return x
