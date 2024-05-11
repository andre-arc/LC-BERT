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


class BertExtractionDataset(Dataset):
    LABEL2INDEX = {'world': 0, 'sports': 1, 'business': 2, 'science': 3}
    INDEX2LABEL = {0: 'world', 1: 'sports', 2: 'business', 3: 'science'}
    NUM_LABELS = 4

    def __init__(self, device, dataset, tokenizer, model, max_len, *args, **kwargs):
        self.device = device
        self.data = self.load_dataset(dataset)
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.vecs = self.get_embeddings(max_len=max_len)
        
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        vecs = self.vecs[index]
        vecs, seq_label = vecs, data['target']
        
        return vecs, np.array(seq_label), data['cleaned_text']
    
    def __len__(self):
        return len(self.data)

    def load_dataset(self, dataset):
        # dataset = []
        # Read file
        # data = pd.read_csv(path)

        # label encoder
        class_names = set(dataset.iloc[:,0].values)
        le = LabelEncoder()
        le.fit(dataset.iloc[:,0])
        class_names = le.classes_

        dataset['target'] = le.transform(dataset.iloc[:,0])

        #clean docs
        dataset['cleaned_text'] = clean(dataset['Description'])

        return dataset
    
    def get_embeddings(self, max_len):
        
        torch.cuda.empty_cache()
        
        if(self.tokenizer is None or self.model is None):
            raise Exception("Sorry, But You must define Tokenizer and Model")
    
        vecs = []
        with torch.no_grad():
          i = 1
          for line in self.data['cleaned_text']:
            sentence = line
            inputs = self.tokenizer.encode_plus(sentence, return_tensors="pt",max_length=max_len, return_attention_mask=True,truncation=True)
            inputs['input_ids'] = inputs['input_ids'].to(self.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.device)

            hidden_states = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_hidden_states=True)[0]
            embedding = hidden_states[:, 0, :]
            vec = embedding.cpu().numpy()
    
            vecs.append(vec)
            print(f'\rWord Embedding Process: {i} / {len(self.data["cleaned_text"])} sentences | GPU Usages: {(torch.cuda.memory_allocated() / 1048576):3.1f}', end=' ')
            i+=1
    
        return np.array(vecs)

class BertExtractionDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, **kwargs):
        super(BertExtractionDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )