import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
  def __init__(self, mode, embeddings, labels):
    self.embeddings = embeddings
    self.labels = labels
    self.mode = mode

  def __len__(self):
    return len(self.embeddings)

  def __getitem__(self, index):
    if self.mode == "dim_reduction":
        word_vector = torch.from_numpy(self.embeddings[index]).unsqueeze(0).float()
    else:
        word_vector = torch.from_numpy(self.embeddings[index]).float()
            
    label = torch.tensor(self.labels[index], dtype=torch.long)

    return word_vector, label

class BertTextDataset(Dataset):
  def __init__(self, tokenizer, sentences, labels, max_len=64):
    self.sentences = sentences
    self.max_len = max_len
    self.tokenizer = tokenizer
    self.labels = labels

  def __len__(self):
    return len(self.sentences)

  def __getitem__(self, index):
    text = self.sentences[index]
    label = torch.tensor(self.labels[index], dtype=torch.long)
    
    encoded = self.tokenizer.encode_plus(text, return_tensors="pt",max_length=self.max_len, return_attention_mask=True,truncation=True, padding='max_length')
    input_ids = encoded['input_ids'][0]
    attention_mask = encoded['attention_mask'][0]

    return input_ids, attention_mask, label