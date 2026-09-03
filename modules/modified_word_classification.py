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
    """
    The classifier emits RAW LOGITS, because `forward` passes them to
    `CrossEntropyLoss`, which applies log_softmax itself.

    Three defects were removed from this stack; do not put them back:

    1. `nn.Softmax(dim=1)` as the final layer. Feeding probabilities to
       CrossEntropyLoss computes log_softmax(softmax(z)), so with a perfect
       one-hot output the loss bottoms out at -ln(e/(e+3)) = 0.743 instead of 0,
       and confidence saturates at 0.475. Gradients are compressed to match and
       the decision boundary can never sharpen.
    2. `nn.ReLU()` on the logits. It clamped every negative class score to
       exactly 0, making distinct wrong classes indistinguishable; when all
       `num_classes` scores started negative - likely, with weights initialised
       symmetric about zero - the output was all-zeros, softmax returned a
       uniform distribution, and ReLU passed zero gradient to every unit. The
       model was dead on arrival with no route to recovery, which is why results
       swung between 0.14 and 0.83 F1 with initialisation order, kernel column
       signs and input distribution.
    3. `nn.Dropout(0.1)` between them, i.e. on the class scores themselves.
       Zeroing a logit is label noise at the decision layer, not
       regularisation. Dropout now precedes the classifier layer instead.

    Note the input is (batch, 1, input_size): a single timestep, so the LSTM
    recurrence never fires and this is effectively an MLP.
    """
    def __init__(self, num_classes, input_size, hidden_size):
        super(BiLSTMForWordClassification, self).__init__()

        self.classifier = nn.Sequential(
            BiLSTMLayer(input_size=input_size, hidden_size=hidden_size, num_layers=2),
            nn.Dropout(0.1),
            nn.Linear(hidden_size*2, num_classes),
        )

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
