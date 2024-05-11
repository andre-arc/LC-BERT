import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel

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

class BiLSTMBertClassification(nn.module):
    def __init__(self, num_classes, input_size, hidden_size):
        super(TopicClassifierDimReduction, self).__init__()

        self.clf = nn.Sequential(
            nn.Dropout(0.1),
            BiLSTMLayer(input_size=input_size, hidden_size=hidden_size, num_layers=1),
            nn.Linear(hidden_size*2, num_classes),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Softmax(dim=1)
        )

    def forward(self, vector):
        logits = self.clf(vector)
        return logits

class BertFineTuneClassification(nn.Module):
    def __init__(self, bert_model, num_classes, hidden_size):
        super(TopicClassifierBertFineTune, self).__init__()
      
        self.bert = bert_model
            
        self.lstm = BiLSTMLayer(input_size=768, hidden_size=hidden_size, num_layers=1)
    
        self.clf = nn.Sequential(
          nn.Linear(hidden_size*2, num_classes),
          nn.ReLU(),
          nn.Dropout(0.2),
          nn.Softmax(dim=1)
        )
    
    def checkFrozen():
        # Check if the first BERT layer is frozen
        is_frozen = not self.bert.encoder.layer[0].parameters()[0].requires_grad
        
        if is_frozen:
            print("The first BERT layer is frozen.")
        else:
            print("The first BERT layer is unfrozen.")
    
    def freezeBertLayer():
        # Freeze all the parameters in the BERT model
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreezeBertLayer():
        # Freeze all the parameters in the BERT model
        for param in self.bert.parameters():
            param.requires_grad = True
