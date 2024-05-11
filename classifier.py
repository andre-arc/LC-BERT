
import matplotlib.pyplot as plt




    
class TopicClassifierBertExtraction(nn.Module):
  def __init__(self, num_classes, input_size, hidden_size):
      super(TopicClassifierBertExtraction, self).__init__()

      self.clf = nn.Sequential(
        #   nn.Dropout(0.5),
          BiLSTMLayer(input_size=input_size, hidden_size=hidden_size, num_layers=1),
          nn.Linear(hidden_size*2, num_classes),
          nn.ReLU(),
          nn.Dropout(0.1),
          nn.Softmax(dim=1)
      )

  def forward(self, vector):
    logits = self.clf(vector)
    return logits
    
class TopicClassifierBertFineTune(nn.Module):
    def __init__(self, bert_model, num_classes, hidden_size):
        super(TopicClassifierBertFineTune, self).__init__()
      
        self.bert = bert_model
            
        self.lstm = nn.LSTM(input_size=768, 
                          hidden_size=hidden_size, 
                          num_layers=1, 
                          batch_first=True, 
                          bidirectional=True)
    
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
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        lstm_output, _=self.lstm(pooled_output)
        logits = self.clf(lstm_output)
        return logits
        
class TopicClassifierBert(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(TopicClassifierBert, self).__init__()
      
        self.bert = bert_model
        self.clf = nn.Linear(bert_model.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.clf(outputs[1])
        return logits