import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import create_decoder_mask, create_padding_mask, create_look_ahead_mask

class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, sequence_length):
        super(PositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim 
        self.sequence_length = sequence_length
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim)
        )
        pe = torch.zeros(sequence_length, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('position_embeddings', pe)

    def forward(self, x):
        x = x + self.position_embeddings[:x.size(1), :].unsqueeze(0)
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

class Decoder_Layer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, h, dropout=0.1):
        super(Decoder_Layer, self).__init__()
        self.Wq = nn.Linear(embedding_dim, embedding_dim)
        self.Wk = nn.Linear(embedding_dim, embedding_dim)
        self.Wv = nn.Linear(embedding_dim, embedding_dim)
        self.Wo = nn.Linear(embedding_dim, embedding_dim)
        self.Wq2 = nn.Linear(embedding_dim, embedding_dim)
        self.Wk2 = nn.Linear(embedding_dim, embedding_dim)
        self.Wv2 = nn.Linear(embedding_dim, embedding_dim)
        self.Wo2 = nn.Linear(embedding_dim, embedding_dim)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.layer_norm3 = nn.LayerNorm(embedding_dim)
        self.ffnn = FFNN(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.h = h
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # x: [batch_size, tgt_seq_len, embedding_dim]
        # encoder_output: [batch_size, src_seq_len, embedding_dim]

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        batch_size, seq_len, _ = Q.shape
        d = self.embedding_dim // self.h

        Q = Q.view(batch_size, seq_len, self.h, d).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.h, d).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.h, d).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)

        if tgt_mask is not None:
            # Mask shape :[batch_size, 1, seq_len, seq_len]
            scores = scores.masked_fill(tgt_mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)

        output1 = self.Wo(context)
        output1 = self.dropout(output1)
        output1 = self.layer_norm1(x + output1)

        Q = self.Wq2(output1)
        K = self.Wk2(encoder_output)
        V = self.Wv2(encoder_output)

        K_seq_len = K.shape[1]

        Q = Q.view(batch_size, seq_len, self.h, d).transpose(1, 2)
        K = K.view(batch_size, K_seq_len, self.h, d).transpose(1, 2)
        V = V.view(batch_size, K_seq_len, self.h, d).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)

        if src_mask is not None:
            # Mask shape: [batch_size, 1, 1, src_seq_len]
            scores = scores.masked_fill(src_mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn) 

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)

        output2 = self.Wo2(context)
        output2 = self.dropout(output2) 
        output2 = self.layer_norm2(output1 + output2)

        ffnn_output = self.ffnn(output2)
        ffnn_output = self.dropout(ffnn_output)  

        output = self.layer_norm3(output2 + ffnn_output)

        return output


class Decoder(nn.Module):
    def __init__(self, num_layers, embedding_dim, vocab_size, hidden_dim, h, dropout=0.1):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.h = h
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = PositionalEmbedding(embedding_dim, sequence_length=1000)
        self.layers = nn.ModuleList([
            Decoder_Layer(embedding_dim, hidden_dim, h, dropout)
            for _ in range(num_layers)
        ])
        self.final_layer = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, texts, encoder_output, src_mask=None):
        embeddings = self.embedding(texts)
        embeddings = self.positional_embedding(embeddings)
        embeddings = self.dropout(embeddings) 
        x = embeddings
        tgt_seq = texts
        tgt_mask = create_decoder_mask(tgt_seq, pad_token_id=0)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask=src_mask, tgt_mask=tgt_mask)
        output = self.final_layer(x)
        return output
