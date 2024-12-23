import os
import torch
import pandas as pd
import numpy as np
import torch.utils.data as data
import re
from tqdm import tqdm
import unicodedata

from loguru import logger
from torch.utils.data import Dataset
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def normalize_string(s):
    # normalize unicode characters
    s = unicode_to_ascii(s)
    # add white space before !.?
    s = re.sub(r'([!.?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = re.sub(r'\s+', r' ', s)
    # change to lower letter
    s = s.lower()
    return s

def cutted_data(cleaned, MIN_LENGTH=4, MAX_LENGTH=30):
    cutted_lines = list()
    for line in cleaned:
        length = len(line.split())
        if length > MIN_LENGTH and length < MAX_LENGTH:
            line = [word for word in line.split()]
            cutted_lines.append(' '.join(line))
    return cutted_lines

# Tokenization function
def process(text_path):
    fop = open(text_path, 'r', encoding='utf8')
    raw_data = fop.read()
    sentences = raw_data.strip().split('\n')
    raw_data_input = [normalize_string(data) for data in sentences]
    raw_data_input = cutted_data(raw_data_input)
    fop.close()

    return raw_data_input

class EUROP_CR(Dataset):
    base_folder = 'data/europarl/txt/en'
    def __init__(self, train=True):
        self.train = train  # training set or test set
        logger.info("Loading Europarl")
        # Load the dataset
        sentences = []
        
        for fn in tqdm(os.listdir(self.base_folder)):
            if not fn.endswith('.txt'): continue
            process_sentences = process(os.path.join(self.base_folder, fn))
            sentences += process_sentences
        
        # remove the same sentences
        a = {}
        for set in sentences:
            if set not in a:
                a[set] = 0
            a[set] += 1
        sentences = list(a.keys())
        
        self.data = [tokenizer.encode(
                s,
                add_special_tokens=True,
                max_length=66,
                padding="max_length",
            ) for s in sentences]
        
        print(f'Number of sentences: {len(sentences)}')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sentence = self.data[index]
        sentence = torch.tensor(sentence)
        return sentence, sentence