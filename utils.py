import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import torch.nn.functional as F
import re
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
def load_data(f):
    rows = []
    g = open(f, "r")
    rows = g.readlines()
    g.close()
    return rows
def cleaner(doc):
    doc=doc.lower()
    doc=doc.replace("\n"," ")
    doc=re.sub("<reviewer></reviewer>","",doc)
    doc=re.sub("<translator></translator>","",doc)
    doc=re.sub(r"<reviewer href>.*</reviewer>","",doc)
    doc=re.sub(r"<translator href>.*</translator>","",doc)
    doc=re.sub("<title>","",doc)
    doc=re.sub("</title>","",doc)
    doc = re.sub(r"<url>.*</url>", "", doc)
    doc = re.sub("<talkid>", "", doc)
    doc = re.sub("</talkid>", "", doc)
    doc = re.sub(r"<keywords>.*</keywords>", "", doc)
    doc = re.sub("<speaker>", "", doc)
    doc = re.sub("</speaker>", "", doc)
    doc = re.sub("<description>", "", doc)
    doc = re.sub("</description>", "", doc)
    doc = re.sub("TED Talk Subtitles and Transcript: ", "", doc)
    doc=re.sub(r"[;—:&%$()/*^\[\]\{\}]"," ",doc)
    doc = re.sub("--", "", doc)
    doc=re.sub(r"[_#]","",doc)
    doc=re.sub(" '", " ",doc)
    doc=re.sub( "[.]'", "",doc)
    doc=re.sub(r"[,]"," ",doc)
    doc=re.sub(r"[?]"," ",doc)
    doc=re.sub(r"[\"]"," ",doc)
    doc=re.sub(r"[!]"," ",doc)
    doc=re.sub(r"dr\.","dr",doc)
    doc=re.sub(r"u\.s\.","us",doc)
    doc=re.sub("[.]"," ",doc)
    return doc

def Tokenizer(doc):
    words = doc.split(' ')
    real_words = []
    for word in words:
        if word:
            real_words.append(word)
    return real_words
def preprocess_data(data):
    for i in range(len(data)):
        data[i] = cleaner(data[i]) 
        data[i] = Tokenizer(data[i])
    sentences = [data[i] for i in range(len(data)) if len(data[i]) > 0]
    return sentences

def map_words_to_ids(sentences):
    unique_words = {}
    unique_words["<pad>"] = 0
    unique_words["<start>"] = 1
    unique_words["<end>"] = 2
    for sentence in sentences:
        for word in sentence:
            if word not in unique_words:
                unique_words[word] = len(unique_words)
    unique_words["<unk>"] = len(unique_words)
    return unique_words

class TedDataset(Dataset):
    def __init__(self, texts, targets, word_to_idx_texts, word_to_idx_targets):
        self.texts = [torch.tensor([word_to_idx_texts["<unk>"] if word not in word_to_idx_texts else word_to_idx_texts[word] for word in text], dtype=torch.long) for text in texts]
        self.targets = [torch.tensor([word_to_idx_targets["<unk>"] if word not in word_to_idx_targets else word_to_idx_targets[word] for word in text], dtype=torch.long) for text in targets]
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.targets[idx]
def collate_fn(batch):
    # Unpack the batch into texts and targets
    texts, targets = zip(*batch)

    # Calculate lengths of each sequence
    texts_lengths = [len(text) for text in texts]
    targets_lengths = [len(target) for target in targets]

    # Sort by the lengths of the texts in descending order
    texts_lengths_tensor = torch.tensor(texts_lengths)
    lengths, sorted_idx = texts_lengths_tensor.sort(descending=True)

    # Sort texts and targets based on the sorted_idx to maintain alignment
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    texts_padded = texts_padded[sorted_idx]
    
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    targets_padded = targets_padded[sorted_idx]

    return texts_padded, targets_padded


class FFNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(FFNN,self).__init__()
        self.layer1 = nn.Linear(embedding_dim, hidden_dim)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
    def forward(self, embeddings):
        embeds = self.layer1(embeddings)
        activated = self.layer2(embeds)
        output = self.layer3(activated)
        output = self.layer_norm(embeddings + output)
        return output

def create_padding_mask(input_tokens, pad_token_id=0):
    mask = (input_tokens != pad_token_id).int()  # [batch_size, seq_len]
    mask = mask.unsqueeze(1).unsqueeze(2)
    return mask
def create_look_ahead_mask(size):
    mask = torch.triu(torch.ones((size, size)), diagonal=1).type(torch.uint8)
    return mask == 0  # Invert mask so that True means keep the position

def create_decoder_mask(tgt_seq, pad_token_id=0):
    tgt_padding_mask =  create_padding_mask(tgt_seq, pad_token_id=pad_token_id)
    tgt_seq_len = tgt_seq.size(1)
    look_ahead_mask = create_look_ahead_mask(tgt_seq_len).to(tgt_seq.device)
    combined_mask = tgt_padding_mask & look_ahead_mask.unsqueeze(0).unsqueeze(1)
    return combined_mask  # Shape: [batch_size, 1, seq_len, seq_len]

