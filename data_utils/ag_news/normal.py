import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..\..\/"))
os.chdir(os.path.join(os.path.dirname(__file__), "..\..\/"))

import numpy as np
import pandas as pd
import torch
import re
from torch.utils.data import Dataset, DataLoader
from utils.preprocessing import clean
from sklearn.preprocessing import LabelEncoder

class BertNormalDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'world': 0, 'sports': 1, 'business': 2, 'science': 3}
    INDEX2LABEL = {0: 'world', 1: 'sports', 2: 'business', 3: 'science'}
    NUM_LABELS = 4
    
    def load_dataset(self, dataset):
        # Read file
        # data = pd.read_csv(path)
        # dataset = []

        # label encoder
        class_names = set(dataset.iloc[:,0].values)
        le = LabelEncoder()
        le.fit(dataset.iloc[:,0])
        class_names = le.classes_

        dataset['target'] = le.transform(dataset.iloc[:,0])

        #clean docs
        dataset['cleaned_text'] = clean(dataset['Description'])

        return dataset
    
    def __init__(self, dataset, tokenizer, no_special_token=False, *args, **kwargs):
        self.data = self.load_dataset(dataset)
        self.tokenizer = tokenizer
        self.no_special_token = no_special_token
    
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        text, labels = data['cleaned_text'], data['target']
        subwords = self.tokenizer.encode(text, add_special_tokens=not self.no_special_token)
        return np.array(subwords), np.array(labels), data['cleaned_text']
    
    def __len__(self):
        return len(self.data)

class BertNormalDataLoader(DataLoader):
    def __init__(self, dataset, max_seq_len=512, *args, **kwargs):
        super(BertNormalDataLoader, self).__init__(dataset=dataset, *args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        self.num_labels = dataset.NUM_LABELS
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        
        # Trimmed input based on specified max_len
        if self.max_seq_len < max_seq_len:
            max_seq_len = self.max_seq_len
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        class_labels_batch = np.zeros((batch_size,), dtype=np.int64)  # Array for class labels
        
        seq_list = []
        for i, (subwords, label, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            
            class_labels_batch[i] = label  # Store the single class label

            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, class_labels_batch, seq_list