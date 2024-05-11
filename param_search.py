import os
import shutil
from copy import deepcopy
import random
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from tqdm import tqdm
from utils.functions import load_model, load_extraction_model
from data_utils.ag_news.whitening import BertWhiteningDataset, BertWhiteningDataLoader
from utils.forward_fn import forward_word_classification, modified_forward_word_classification
from utils.metrics import news_categorization_metrics_fn

args = {
        'device': 0,
        'seed': 88,
        'lr': 2e-3,
        'eps': 1e-8,
        'max_seq_len': 512,
        'num_labels': 4,
        'model_name': 'bilstm-dim-reduction',
        'task': 'sequence_classification',
        'num_labels': BertWhiteningDataset.NUM_LABELS,
        'dataset_class': BertWhiteningDataset,
        'dataloader_class': BertWhiteningDataLoader,
        'extract_model': 'bert-base-uncased',
        'forward_fn': modified_forward_word_classification,
        'metrics_fn': news_categorization_metrics_fn,
        'valid_criterion': 'F1',
        'train_set_path': './dataset/ag-news/train.csv',
        'valid_set_path': './dataset/ag-news/valid.csv',
        'test_set_path': './dataset/ag-news/test.csv',
        'vocab_path': "",
        'train_batch_size': 128,
        'valid_batch_size': 4,
        'vocab_path': "",
        'lower': True,
        'no_special_token': True,
        'k_fold': 1
    }

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

extract_model, extract_tokenizer, a, b_ = load_extraction_model(args)
train_dataset_path = args['train_set_path']
train_dataset = args['dataset_class'](args['device'], train_dataset_path, extract_tokenizer, extract_model, args['max_seq_len'],  lowercase=args["lower"], no_special_token=args['no_special_token'])
train_loader = args['dataloader_class'](dataset=train_dataset, batch_size=args['train_batch_size'], shuffle=False)  

class BiLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(BiLSTMLayer, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            bidirectional=True)
    
    def forward(self, input):
        outputs, _ = self.lstm(input)
        return outputs[:, -1, :]

class BiLSTMForWordClassification(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size):
        super(BiLSTMForWordClassification, self).__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            BiLSTMLayer(input_size=input_size, hidden_size=hidden_size, num_layers=2),
            nn.Linear(hidden_size*2, num_classes),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Softmax(dim=1)
        )
        # self.classifier = nn.Linear(input_size, num_classes)
        # self.dropout = nn.Dropout()

        self.num_classes = num_classes

    def forward(self, vector):
        logits = self.classifier(vector)
        outputs = logits
        
        return outputs

model = BiLSTMForWordClassification(args['num_labels'], input_size=256, hidden_size=32)

 # Make sure cuda is deterministic
torch.backends.cudnn.deterministic = True

 # Set random seed
set_seed(args['seed'])  # Added here for reproductibility   

# Learning parameters. 
lr = args['lr']
epochs = 20
device = 'cpu'
print(f"Computation device: {device}\n")

# Loss function. Required for defining `NeuralNetClassifier`
criterion = CrossEntropyLoss()

# Define hyperparameters to search
params = {
    'lr': [2e-3, 2e-4, 5e-3, 5e-4],
    # 'max_epochs': list(range(20, 55, 5)),
    'max_epochs': list(range(2, 7, 1)),
    'module__hidden_size': [32, 64, 128],
    'optimizer__eps': [1e-7, 1e-6, 1e-8],  # Add epsilon values for numerical stability
    'optimizer': [optim.Adam, optim.AdamW, optim.Adamax],
}

# Instance of `NeuralNetClassifier` to be passed to `GridSearchCV` 
net = NeuralNetClassifier(
    module=model, max_epochs=epochs,
    module__input_size=256,
    module__hidden_size=32,
    module__num_classes=4,
    optimizer=torch.optim.Adamax,
    criterion=criterion,
    lr=lr, verbose=1
)

"""
Define `GridSearchCV`.
4 lrs * 7 max_epochs * 4 module__first_conv_out * 3 module__first_fc_out
* 2 CVs = 672 fits.
"""
gs = GridSearchCV(
    net, params, refit=False, scoring='accuracy', verbose=1, cv=2
)
counter = 0
# Run each fit for 2 batches. So, if we have `n` fits, then it will
# actually for `n*2` times. We have 672 fits, so total, 
# 672 * 2 = 1344 runs.
search_batches = 2
"""
This will run `n` (`n` is calculated from `params`) number of fits 
on each batch of data, so be careful.
If you want to run the `n` number of fits just once, 
that is, on one batch of data,
add `break` after this line:
    `outputs = gs.fit(image, labels)`
Note: This will take a lot of time to run
"""
for i, data in enumerate(train_loader):
    counter += 1
    vector_batch, label_batch = data[:-1]
    # Prepare input & label
    vector_batch = torch.FloatTensor(vector_batch)
    label_batch = torch.LongTensor(label_batch)

    if device == "cuda":
        vector_batch = vector_batch.cuda()
        label_batch = label_batch.cuda()
    
    outputs = gs.fit(vector_batch, label_batch)

    # GridSearch for `search_batches` number of times.
    if counter == search_batches:
        break

print('SEARCH COMPLETE')
print("best score: {:.3f}, best params: {}".format(gs.best_score_, gs.best_params_))