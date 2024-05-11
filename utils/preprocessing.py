import nltk
import os
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.utils import simple_preprocess
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

#Defining a function to clean the text
def clean(docs):
    # Insert function for preprocessing the text
    def sent_to_words(sentences):
        for sentence in sentences:
            yield (simple_preprocess(str(sentence), deacc = True))
    # Tokenize the text
    tokens = sent_to_words(docs)
    # Create stopwords set
    stop = set(stopwords.words("english"))
    # Create lemmatizer
    lmtzr = WordNetLemmatizer()
    # Remove stopwords from text
    tokens_stopped = [[word for word in post if word not in stop] for post in tokens]
    # Lemmatize text
    tokens_cleaned = [[lmtzr.lemmatize(word) for word in post] for post in tokens_stopped]
    
    tokens_to_string = [' '.join(doc) for doc in tokens_cleaned]

    # Return cleaned string text
    return tokens_to_string

def split_dataset(features, labels, seed_number):
  x_train, x_val, y_train, y_val = train_test_split(features,
                                                      labels,
                                                      random_state=seed_number,
                                                      test_size=0.2)

  x_train, x_test, y_train, y_test = train_test_split(x_train,
                                                      y_train,
                                                      random_state=seed_number,
                                                      test_size=0.2)
  return x_train, x_val, x_test, y_train, y_val, y_test

def checkCreateDirectory(path):
  isExist = os.path.exists(path)
  if not isExist:

    # Create a new directory because it does not exist
    os.makedirs(path)

def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def loop_fn(mode, dataset, dataloader, model, optimizer, criterion, device):

  if mode == 'train':
    # Set model to training mode
    model.train()

  if mode == 'test':
    # Set model to evaluation mode
    model.eval()

  cost = correct = 0

  # Loop over batches in training data
  for step, batch in enumerate(dataloader):
    # Retrieve inputs and labels
    vector, labels = batch

    vector = vector.to(device)
    labels = labels.to(device)

    # model.zero_grad()

    # Compute model output and loss
    outputs = model(vector)
    loss = criterion(outputs, labels)

    if mode == 'train':
      # Backpropagate loss and update model parameters
      loss.backward()
      optimizer.step()
      # Clear gradients
      optimizer.zero_grad()

    # Compute correct & loss
    cost += loss.item() * vector.shape[0]
    correct += (outputs.argmax(1) == labels).sum().item()
  
  cost = cost / len(dataset)
  acc = correct / len(dataset)
  return cost, acc
  
def loop_bert_train(mode, dataset, dataloader, model, optimizer, criterion, device):

  if mode == 'train':
    # Set model to training mode
    model.train()

  if mode == 'test':
    # Set model to evaluation mode
    model.eval()

  cost = correct = 0

  # Loop over batches in training data
  for step, batch in enumerate(dataloader):
    # Retrieve inputs and labels
    input_ids, attention_mask, labels = batch

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    # model.zero_grad()

    # Compute model output and loss
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    loss = criterion(outputs, labels)

    if mode == 'train':
      # Backpropagate loss and update model parameters
      loss.backward()
      optimizer.step()
      # Clear gradients
      optimizer.zero_grad()

    # Compute correct & loss
    cost += loss.item() * labels.shape[0]
    correct += (outputs.argmax(1) == labels).sum().item()
  
  cost = cost / len(dataset)
  acc = correct / len(dataset)
  return cost, acc