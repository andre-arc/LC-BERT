import logging
import math
import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

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
            # nn.Dropout(0.1),
            BiLSTMLayer(input_size=input_size, hidden_size=hidden_size, num_layers=2),
            nn.Linear(hidden_size*2, num_classes),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Softmax(dim=1)
        )
        # self.classifier = nn.Linear(input_size, num_classes)
        # self.dropout = nn.Dropout()

        self.num_classes = num_classes

    def forward(self, vector, labels):
        logits = self.classifier(vector)
        outputs = (logits, )

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

            outputs = (loss,) + outputs
        
        return outputs

class MLPForWordClassification(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size):
        super(MLPForWordClassification, self).__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(input_size, num_classes),
            # nn.Softmax(dim=1)
        )
        # self.classifier = nn.Linear(input_size, num_classes)
        # self.dropout = nn.Dropout()

        self.num_classes = num_classes

    def forward(self, vector, labels):
        logits = self.classifier(vector)
        logits = torch.squeeze(logits)
        outputs = (logits, )

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

            outputs = (loss,) + outputs
        
        return outputs
