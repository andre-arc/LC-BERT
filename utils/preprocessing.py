"""
Text preprocessing utilities for LC-BERT project.

This module provides flexible text preprocessing with three different methods:
1. 'gensim': Fast manual preprocessing (default)
2. 'nltk': NLTK-based preprocessing (more accurate)
3. 'stanza': Stanza NLP pipeline (most accurate, slowest)

Usage:
    # Set method via environment variable (before importing)
    import os
    os.environ['PREPROCESSING_METHOD'] = 'nltk'
    from utils.preprocessing import clean

    # Or set directly in code
    import utils.preprocessing as preprocessing
    preprocessing.set_preprocessing_method('stanza')

    # Then use clean function
    cleaned_text = preprocessing.clean(["Sample text to clean"])
"""

import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
import re

# Preprocessing method (can be set via environment variable or changed here)
# Options: 'gensim', 'nltk', 'stanza'
PREPROCESSING_METHOD = os.environ.get('PREPROCESSING_METHOD', 'gensim')

# Import required libraries based on method
if PREPROCESSING_METHOD == 'nltk':
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.wordnet import WordNetLemmatizer
    # Uncomment to download NLTK data (only needed once)
    # nltk.download('stopwords')
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')
elif PREPROCESSING_METHOD == 'stanza':
    import stanza
    from gensim.utils import simple_preprocess
    # Uncomment to download Stanza model (only needed once)
    # stanza.download('en')
elif PREPROCESSING_METHOD == 'gensim':
    from gensim.utils import simple_preprocess
else:
    raise ValueError(f"Invalid PREPROCESSING_METHOD: {PREPROCESSING_METHOD}. Choose 'gensim', 'nltk', or 'stanza'.")

# English stopwords set
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will',
    'just', 'should', 'now'
}

# Simple lemmatization dictionary for common words
LEMMA_DICT = {
    # Common verbs
    'running': 'run', 'runs': 'run', 'ran': 'run',
    'going': 'go', 'goes': 'go', 'went': 'go', 'gone': 'go',
    'having': 'have', 'has': 'have', 'had': 'have',
    'doing': 'do', 'does': 'do', 'did': 'do', 'done': 'do',
    'saying': 'say', 'says': 'say', 'said': 'say',
    'getting': 'get', 'gets': 'get', 'got': 'get', 'gotten': 'get',
    'making': 'make', 'makes': 'make', 'made': 'make',
    'taking': 'take', 'takes': 'take', 'took': 'take', 'taken': 'take',
    'coming': 'come', 'comes': 'come', 'came': 'come',
    'seeing': 'see', 'sees': 'see', 'saw': 'see', 'seen': 'see',
    'knowing': 'know', 'knows': 'know', 'knew': 'know', 'known': 'know',
    'thinking': 'think', 'thinks': 'think', 'thought': 'think',
    'looking': 'look', 'looks': 'look', 'looked': 'look',
    'wanting': 'want', 'wants': 'want', 'wanted': 'want',
    'giving': 'give', 'gives': 'give', 'gave': 'give', 'given': 'give',
    'using': 'use', 'uses': 'use', 'used': 'use',
    'finding': 'find', 'finds': 'find', 'found': 'find',
    'telling': 'tell', 'tells': 'tell', 'told': 'tell',
    'asking': 'ask', 'asks': 'ask', 'asked': 'ask',
    'working': 'work', 'works': 'work', 'worked': 'work',
    'calling': 'call', 'calls': 'call', 'called': 'call',
    'trying': 'try', 'tries': 'try', 'tried': 'try',
    'feeling': 'feel', 'feels': 'feel', 'felt': 'feel',
    'leaving': 'leave', 'leaves': 'leave', 'left': 'leave',
    'putting': 'put', 'puts': 'put',
    'meaning': 'mean', 'means': 'mean', 'meant': 'mean',
    'keeping': 'keep', 'keeps': 'keep', 'kept': 'keep',
    'beginning': 'begin', 'begins': 'begin', 'began': 'begin', 'begun': 'begin',
    'seeming': 'seem', 'seems': 'seem', 'seemed': 'seem',
    'helping': 'help', 'helps': 'help', 'helped': 'help',
    'talking': 'talk', 'talks': 'talk', 'talked': 'talk',
    'turning': 'turn', 'turns': 'turn', 'turned': 'turn',
    'starting': 'start', 'starts': 'start', 'started': 'start',
    'showing': 'show', 'shows': 'show', 'showed': 'show', 'shown': 'show',
    'hearing': 'hear', 'hears': 'hear', 'heard': 'hear',
    'playing': 'play', 'plays': 'play', 'played': 'play',
    'moving': 'move', 'moves': 'move', 'moved': 'move',
    'living': 'live', 'lives': 'live', 'lived': 'live',
    'believing': 'believe', 'believes': 'believe', 'believed': 'believe',
    'bringing': 'bring', 'brings': 'bring', 'brought': 'bring',
    'happening': 'happen', 'happens': 'happen', 'happened': 'happen',
    'writing': 'write', 'writes': 'write', 'wrote': 'write', 'written': 'write',
    'providing': 'provide', 'provides': 'provide', 'provided': 'provide',
    'sitting': 'sit', 'sits': 'sit', 'sat': 'sit',
    'standing': 'stand', 'stands': 'stand', 'stood': 'stand',
    'losing': 'lose', 'loses': 'lose', 'lost': 'lose',
    'paying': 'pay', 'pays': 'pay', 'paid': 'pay',
    'meeting': 'meet', 'meets': 'meet', 'met': 'meet',
    'including': 'include', 'includes': 'include', 'included': 'include',
    'continuing': 'continue', 'continues': 'continue', 'continued': 'continue',
    'setting': 'set', 'sets': 'set',
    'learning': 'learn', 'learns': 'learn', 'learned': 'learn', 'learnt': 'learn',
    'changing': 'change', 'changes': 'change', 'changed': 'change',
    'leading': 'lead', 'leads': 'lead', 'led': 'lead',
    'understanding': 'understand', 'understands': 'understand', 'understood': 'understand',
    'watching': 'watch', 'watches': 'watch', 'watched': 'watch',
    'following': 'follow', 'follows': 'follow', 'followed': 'follow',
    'stopping': 'stop', 'stops': 'stop', 'stopped': 'stop',
    'creating': 'create', 'creates': 'create', 'created': 'create',
    'speaking': 'speak', 'speaks': 'speak', 'spoke': 'speak', 'spoken': 'speak',
    'reading': 'read', 'reads': 'read',
    'allowing': 'allow', 'allows': 'allow', 'allowed': 'allow',
    'adding': 'add', 'adds': 'add', 'added': 'add',
    'spending': 'spend', 'spends': 'spend', 'spent': 'spend',
    'growing': 'grow', 'grows': 'grow', 'grew': 'grow', 'grown': 'grow',
    'opening': 'open', 'opens': 'open', 'opened': 'open',
    'walking': 'walk', 'walks': 'walk', 'walked': 'walk',
    'winning': 'win', 'wins': 'win', 'won': 'win',
    'offering': 'offer', 'offers': 'offer', 'offered': 'offer',
    'remembering': 'remember', 'remembers': 'remember', 'remembered': 'remember',
    'considering': 'consider', 'considers': 'consider', 'considered': 'consider',
    'appearing': 'appear', 'appears': 'appear', 'appeared': 'appear',
    'buying': 'buy', 'buys': 'buy', 'bought': 'buy',
    'waiting': 'wait', 'waits': 'wait', 'waited': 'wait',
    'serving': 'serve', 'serves': 'serve', 'served': 'serve',
    'dying': 'die', 'dies': 'die', 'died': 'die',
    'sending': 'send', 'sends': 'send', 'sent': 'send',
    'expecting': 'expect', 'expects': 'expect', 'expected': 'expect',
    'building': 'build', 'builds': 'build', 'built': 'build',
    'staying': 'stay', 'stays': 'stay', 'stayed': 'stay',
    'falling': 'fall', 'falls': 'fall', 'fell': 'fall', 'fallen': 'fall',
    'cutting': 'cut', 'cuts': 'cut',
    'reaching': 'reach', 'reaches': 'reach', 'reached': 'reach',
    'killing': 'kill', 'kills': 'kill', 'killed': 'kill',
    'remaining': 'remain', 'remains': 'remain', 'remained': 'remain',
    
    # Common nouns
    'studies': 'study', 'studied': 'study', 'studying': 'study',
    'companies': 'company',
    'countries': 'country',
    'cities': 'city',
    'stories': 'story',
    'parties': 'party',
    'families': 'family',
    'babies': 'baby',
    'ladies': 'lady',
    'iries': 'iry',
    'men': 'man',
    'women': 'woman',
    'children': 'child',
    'teeth': 'tooth',
    'feet': 'foot',
    'people': 'person',
    'leaves': 'leaf',
    'lives': 'life',
    'knives': 'knife',
    'wives': 'wife',
    'halves': 'half',
    'shelves': 'shelf',
}

def simple_lemmatize(word):
    """Apply simple lemmatization rules"""
    # Check dictionary first
    if word in LEMMA_DICT:
        return LEMMA_DICT[word]
    
    # Apply suffix rules
    if len(word) > 4:
        if word.endswith('ies'):
            return word[:-3] + 'y'
        elif word.endswith('ied'):
            return word[:-3] + 'y'
    
    if len(word) > 3:
        if word.endswith('ing'):
            # Remove 'ing' and check if valid
            base = word[:-3]
            if len(base) >= 3:
                return base
        elif word.endswith('ed'):
            # Remove 'ed' and check if valid
            base = word[:-2]
            if len(base) >= 3:
                return base
        elif word.endswith('es'):
            return word[:-2]
    
    if len(word) > 2:
        if word.endswith('s') and not word.endswith('ss') and not word.endswith('us'):
            return word[:-1]
    
    return word

def clean(docs):
    """
    Clean and preprocess text documents using the selected method.

    Method is selected via PREPROCESSING_METHOD variable:
    - 'gensim': Manual preprocessing with Gensim tokenization (default, fast)
    - 'nltk': NLTK-based stopwords removal and lemmatization (more accurate)
    - 'stanza': Stanza NLP pipeline (most accurate, slowest)

    Steps:
    1. Whitespace removal
    2. Tokenization
    3. Punctuation removal
    4. Stopwords removal
    5. Lemmatization

    Args:
        docs: List of text documents or single document string

    Returns:
        List of cleaned text strings
    """
    # Ensure docs is iterable
    if isinstance(docs, str):
        docs = [docs]

    # Dictionary mapping method names to cleaning functions
    cleaning_methods = {
        'gensim': _clean_gensim,
        'nltk': _clean_nltk,
        'stanza': _clean_stanza,
    }

    # Get the cleaning function for the selected method
    clean_func = cleaning_methods.get(PREPROCESSING_METHOD, _clean_gensim)

    return clean_func(docs)


def _clean_gensim(docs):
    """
    Clean text using Gensim's simple_preprocess with manual lemmatization.
    This is the fastest method.
    """
    # Tokenize using gensim's simple_preprocess
    # This handles: whitespace, lowercase, punctuation removal, min_len=2
    def sent_to_words(sentences):
        for sentence in sentences:
            yield simple_preprocess(str(sentence), deacc=True)

    tokens = list(sent_to_words(docs))

    # Remove stopwords
    tokens_stopped = [
        [word for word in post if word not in STOPWORDS]
        for post in tokens
    ]

    # Lemmatize
    tokens_cleaned = [
        [simple_lemmatize(word) for word in post]
        for post in tokens_stopped
    ]

    # Convert back to strings
    tokens_to_string = [' '.join(doc) for doc in tokens_cleaned]

    return tokens_to_string


def _clean_nltk(docs):
    """
    Clean text using NLTK for stopwords removal and lemmatization.
    More accurate than manual method but slower.
    """
    # Tokenize using gensim's simple_preprocess
    def sent_to_words(sentences):
        for sentence in sentences:
            yield simple_preprocess(str(sentence), deacc=True)

    # Tokenize the text
    tokens = list(sent_to_words(docs))

    # Create stopwords set from NLTK
    stop = set(stopwords.words("english"))

    # Create lemmatizer
    lmtzr = WordNetLemmatizer()

    # Remove stopwords from text
    tokens_stopped = [
        [word for word in post if word not in stop]
        for post in tokens
    ]

    # Lemmatize text using NLTK WordNetLemmatizer
    tokens_cleaned = [
        [lmtzr.lemmatize(word) for word in post]
        for post in tokens_stopped
    ]

    # Convert back to strings
    tokens_to_string = [' '.join(doc) for doc in tokens_cleaned]

    return tokens_to_string


def _clean_stanza(docs):
    """
    Clean text using Stanza NLP pipeline.
    Most accurate but slowest method.
    """
    # Initialize Stanza pipeline (cached after first call)
    if not hasattr(_clean_stanza, 'nlp'):
        _clean_stanza.nlp = stanza.Pipeline('en', processors='tokenize,lemma,pos', use_gpu=True, verbose=False)

    cleaned_docs = []
    for doc in docs:
        processed = _clean_stanza.nlp(doc.lower())

        tokens = []
        for sentence in processed.sentences:
            for word in sentence.words:
                # Filter stopwords and keep only alphabetic words
                if word.text not in STOPWORDS and word.text.isalpha():
                    tokens.append(word.lemma)

        cleaned_docs.append(' '.join(tokens))

    return cleaned_docs


def set_preprocessing_method(method):
    """
    Set the preprocessing method to use.

    Args:
        method: One of 'gensim', 'nltk', or 'stanza'

    Raises:
        ValueError: If method is not one of the valid options
    """
    global PREPROCESSING_METHOD

    valid_methods = ['gensim', 'nltk', 'stanza']
    if method not in valid_methods:
        raise ValueError(f"Invalid preprocessing method: {method}. Choose from {valid_methods}")

    PREPROCESSING_METHOD = method
    print(f"Preprocessing method set to: {method}")

    # Import required libraries for the new method
    if method == 'nltk':
        try:
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem.wordnet import WordNetLemmatizer
        except ImportError:
            raise ImportError("NLTK not installed. Install with: pip install nltk")
    elif method == 'stanza':
        try:
            import stanza
        except ImportError:
            raise ImportError("Stanza not installed. Install with: pip install stanza")


def get_preprocessing_method():
    """
    Get the current preprocessing method.

    Returns:
        str: Current preprocessing method ('gensim', 'nltk', or 'stanza')
    """
    return PREPROCESSING_METHOD


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