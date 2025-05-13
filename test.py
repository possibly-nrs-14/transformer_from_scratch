import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import torch.nn.functional as F
import csv
import re
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from encoder import Encoder
from decoder import Decoder
from utils import load_data, preprocess_data, map_words_to_ids, FFNN, TedDataset
from train import Transformer, unique_words_train_en, unique_words_train_fr
from tqdm import tqdm

class TedDataset_Test(Dataset):
    def __init__(self, texts, targets, word_to_idx_texts, word_to_idx_targets):
        tensors = [torch.tensor([word_to_idx_texts["<unk>"] if word not in word_to_idx_texts else word_to_idx_texts[word] for word in text], dtype=torch.long) for text in texts]
        self.texts = [[text, tensor] for (text, tensor) in zip(texts, tensors)]
        self.targets = [torch.tensor([word_to_idx_targets["<unk>"] if word not in word_to_idx_targets else word_to_idx_targets[word] for word in text], dtype=torch.long) for text in targets]
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx][0], self.texts[idx][1], self.targets[idx]
def collate_fn_2(batch):
    sentences, texts, targets = zip(*batch)
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

    return sentences, texts_padded, targets_padded
def evaluate_generate(model, iterator, device):
    model.eval()
    scores = []
    for sentences, texts, targets in tqdm(iterator):
        texts = texts.to(device)
        targets = targets.to(device)
        predictions = model.generate(texts, unique_words_train_fr["<start>"], unique_words_train_fr["<end>"])
        actual = [idx_to_words[targets[0][i].item()] for i in range(len(targets[0]))]
        pred = [idx_to_words[predictions[0][i].item()] for i in range(len(predictions[0]))]
        print(actual, pred, flush=True)
        bleu_score = sentence_bleu([actual], pred)
        scores.append([sentences[0], bleu_score])
    return scores
def test_loop(model, test_loader, device):
    scores = evaluate_generate(model, test_loader, device)
    av_bleu = sum([scores[i][1] for i in range (len(scores))]) / len(scores)
    f = open("testbleu.txt", "w")
    for i in range(len(scores)):
        f.write(' '.join(scores[i][0]) + " " + str(scores[i][1]) + "\n")
    f.write(str(av_bleu) + "\n")
    f.close()
test = load_data('test.en')
test_t = load_data('test.fr')
test_sentences = preprocess_data(test)
test_targets = preprocess_data(test_t)
for i in range(len(test_targets)):
    test_targets[i].insert(0, "<start>")
    test_targets[i].append("<end>")

idx_to_words = {idx: word for word, idx in unique_words_train_fr.items()}
test_dataset = TedDataset_Test(test_sentences, test_targets, unique_words_train_en, unique_words_train_fr)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn_2)
embedding_dim = 256
hidden_dim =  512
num_layers = 2
h = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Transformer(num_layers, embedding_dim, len(unique_words_train_en), len(unique_words_train_fr), hidden_dim, h)
model.load_state_dict(torch.load('transformer.pt'))
model = model.to(device)
test_loop(model, test_loader, device)