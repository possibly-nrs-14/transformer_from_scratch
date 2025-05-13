import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import torch.nn.functional as F
import csv
import re
from encoder import Encoder
from decoder import Decoder
from utils import load_data, preprocess_data, map_words_to_ids, FFNN, TedDataset, collate_fn, create_padding_mask, create_decoder_mask
from tqdm import tqdm

class Transformer(nn.Module):
    def __init__(self, num_layers, embedding_dim, vocab_size_en, vocab_size_fr, hidden_dim, h):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, embedding_dim, vocab_size_en, hidden_dim, h)
        self.decoder = Decoder(num_layers, embedding_dim, vocab_size_fr, hidden_dim, h)
    def forward(self, texts, texts_out):
        src_mask = create_padding_mask(texts, pad_token_id=0)
        encoder_output = self.encoder(texts, src_mask=src_mask) 
        decoder_output = self.decoder(texts_out, encoder_output, src_mask=src_mask)
        return decoder_output
    def generate(self, texts, start_token_id, end_token_id, max_len=50):
        src_mask = create_padding_mask(texts, pad_token_id=0)
        encoder_output = self.encoder(texts, src_mask=src_mask)
        batch_size = texts.size(0)  # Get the batch size
        generated_sequences = torch.full((batch_size, 1), start_token_id, dtype=torch.long).to(texts.device)  
        for i in range(max_len):
            decoder_output = self.decoder(generated_sequences, encoder_output,  src_mask=src_mask)
            next_token_logits = decoder_output[:, -1, :]  
            next_token_id = next_token_logits.argmax(dim=-1, keepdim=True) 
            generated_sequences = torch.cat([generated_sequences, next_token_id], dim=1)
            if (next_token_id == end_token_id).all():
                break
        
        return generated_sequences
        
def Train(model, iterator, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for texts, targets in tqdm(iterator):
        optimizer.zero_grad()
        texts = texts.to(device)
        targets = targets.to(device)
        predictions = model(texts, targets[:, :-1])
        loss = criterion(torch.transpose(predictions, 1, 2), targets[:, 1:])
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
    return total_loss / len(iterator)


def evaluate_loss(model, iterator, criterion, device):
    model.eval()
    total_loss = 0
    for texts, targets in tqdm(iterator):
        texts = texts.to(device)
        targets = targets.to(device)
        predictions = model(texts, targets[:, :-1])
        loss = criterion(torch.transpose(predictions, 1, 2), targets[:, 1:])
        total_loss += loss.detach().item()
    return total_loss / len(iterator)
    

def train_loop(model, train_loader, val_loader, train_test_loader, optimizer, criterion, epochs, device):
    for epoch in range(epochs):
        train_loss = Train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        train_test_loss = evaluate_loss(model, train_test_loader, criterion, device)
        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'Train Loss:{train_test_loss:.4f}')
        print(f'Val Loss:{val_loss:.4f}')
train = load_data('train.en')
val = load_data('dev.en')
train_t = load_data('train.fr')
val_t = load_data('dev.fr')
train_targets = preprocess_data(train_t)
train_sentences = preprocess_data(train) 
val_sentences = preprocess_data(val)
val_targets = preprocess_data(val_t)
for i in range(len(val_targets)):
    val_targets[i].insert(0, "<start>")
    val_targets[i].append("<end>")

train_test_targets = train_targets.copy()       
for i in range(len(train_test_targets)):
    train_test_targets[i].insert(0, "<start>")
    train_test_targets[i].append("<end>")
unique_words_train_en = map_words_to_ids(train_sentences)
unique_words_train_fr = map_words_to_ids(train_targets)
 
if __name__ == "__main__":
    train_dataset = TedDataset(train_sentences, train_targets, unique_words_train_en, unique_words_train_fr)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_dataset = TedDataset(val_sentences, val_targets, unique_words_train_en, unique_words_train_fr)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    train_test_dataset = TedDataset(train_sentences, train_test_targets, unique_words_train_en, unique_words_train_fr)
    train_test_loader = DataLoader(train_test_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    embedding_dim = 256
    hidden_dim =  512
    num_layers = 2
    h = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(num_layers, embedding_dim, len(unique_words_train_en), len(unique_words_train_fr), hidden_dim, h)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    epochs = 10
    train_loop(model, train_loader, val_loader, train_test_loader, optimizer, criterion, epochs, device)
    torch.save(model.state_dict(), 'transformer.pt')
    


