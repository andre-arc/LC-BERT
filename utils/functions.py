from transformers import BertConfig, BertTokenizer, RobertaConfig, RobertaTokenizer, BertModel, RobertaModel
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertModel
from transformers import AlbertConfig, AlbertTokenizer, AlbertModel
from modules.word_classification import BertForWordClassification, RobertaForWordClassification, DistilBertForWordClassification, AlbertForWordClassification
from modules.modified_word_classification import BiLSTMForWordClassification, MLPForWordClassification
# from modules.modified_word_classification impor

def load_model(args):
    if 'bert-base-uncased' in args['model_name']:
        # bert-base-multilingual-uncased or bert-base-multilingual-cased
        # Prepare config & tokenizer
        vocab_path, config_path = None, None
        tokenizer = BertTokenizer.from_pretrained(args['model_name'])
        config = BertConfig.from_pretrained(args['model_name'])
        if type(args['num_labels']) == list:
            config.num_labels = max(args['num_labels'])
            config.num_labels_list = args['num_labels']
        else:
            config.num_labels = args['num_labels']
        
        # Instantiate model
        model = BertForWordClassification.from_pretrained(args['model_name'], config=config)

#         for param in model.bert.parameters():
#             param.requires_grad = False

    elif 'roberta-base' in args['model_name']:

        # Prepare config & tokenizer
        vocab_path, config_path = None, None
        tokenizer = RobertaTokenizer.from_pretrained(args['model_name'])            
        config = RobertaConfig.from_pretrained(args['model_name'])
        if type(args['num_labels']) == list:
            config.num_labels = max(args['num_labels'])
            config.num_labels_list = args['num_labels']
        else:
            config.num_labels = args['num_labels']

        # Instantiate model
        model = RobertaForWordClassification.from_pretrained(args['model_name'], config=config)

#         for param in model.roberta.parameters():
#           param.requires_grad = False

    elif 'distilbert-base-uncased' in args['model_name']:
        # Prepare config & tokenizer
        vocab_path, config_path = None, None
        tokenizer = DistilBertTokenizer.from_pretrained(args['model_name'])
        config = DistilBertConfig.from_pretrained(args['model_name'])
        if type(args['num_labels']) == list:
            config.num_labels = max(args['num_labels'])
            config.num_labels_list = args['num_labels']
        else:
            config.num_labels = args['num_labels']

        # Instantiate model
        model = DistilBertForWordClassification.from_pretrained(args['model_name'], config=config)

#         for param in model.distilbert.parameters():
#             param.requires_grad = False

    elif 'albert-base-v2' in args['model_name']:
        # Prepare config & tokenizer
        vocab_path, config_path = None, None
        tokenizer = AlbertTokenizer.from_pretrained(args['model_name'])
        config = AlbertConfig.from_pretrained(args['model_name'])
        if type(args['num_labels']) == list:
            config.num_labels = max(args['num_labels'])
            config.num_labels_list = args['num_labels']
        else:
            config.num_labels = args['num_labels']

        # Instantiate model
        model = AlbertForWordClassification.from_pretrained(args['model_name'], config=config)

#         for param in model.albert.parameters():
#             param.requires_grad = False

    elif 'bilstm' == args['model_name']:
        # Prepare config & tokenizer
        vocab_path, config_path = None, None
        model = BiLSTMForWordClassification(args['num_labels'], 768, 32)
        tokenizer = None

    elif 'bilstm-dim-reduction' == args['model_name']:
        # Prepare config & tokenizer
        vocab_path, config_path = None, None
        model = BiLSTMForWordClassification(args['num_labels'], 256, 32)
        tokenizer = None

    elif 'mlp-dim-reduction' == args['model_name']:
        # Prepare config & tokenizer
        vocab_path, config_path = None, None
        model = MLPForWordClassification(args['num_labels'], 256, 32)
        tokenizer = None

    return model, tokenizer, vocab_path, config_path

def load_tokenizer(args):
    if 'bert-base-uncased' in args['model_name']:
        # bert-base-multilingual-uncased or bert-base-multilingual-cased
        tokenizer = BertTokenizer.from_pretrained(args['model_name'])

    elif 'roberta-base' in args['model_name']:
        # Prepare config & tokenizer
        tokenizer = RobertaTokenizer.from_pretrained(args['model_name'])

    elif 'distilbert-base-uncased' in args['model_name']:
        tokenizer = DistilBertTokenizer.from_pretrained(args['model_name'])

    elif 'albert-base-v2' in args['model_name']:
        tokenizer = AlbertTokenizer.from_pretrained(args['model_name'])

    else:
        tokenizer = None

    return tokenizer

def load_extraction_model(args):
    if 'bert-base-uncased' in args['extract_model']:
        # bert-base-multilingual-uncased or bert-base-multilingual-cased
        # Prepare config & tokenizer
        vocab_path, config_path = None, None
        tokenizer = BertTokenizer.from_pretrained(args['extract_model'])
        config = BertConfig.from_pretrained(args['extract_model'])

        # Instantiate model
        model = BertModel.from_pretrained(args['extract_model'], config=config)

    elif 'roberta-base' in args['extract_model']:
        # Prepare config & tokenizer
        vocab_path, config_path = None, None
        tokenizer = RobertaTokenizer.from_pretrained(args['extract_model'])
        config = RobertaConfig.from_pretrained(args['extract_model'])

        # Instantiate model
        model = RobertaModel.from_pretrained(args['extract_model'], config=config)

    elif 'distilbert-base-uncased' in args['extract_model']:
        # Prepare config & tokenizer
        vocab_path, config_path = None, None
        tokenizer = DistilBertTokenizer.from_pretrained(args['extract_model'])
        config = DistilBertConfig.from_pretrained(args['extract_model'])

        # Instantiate model
        model = DistilBertModel.from_pretrained(args['extract_model'], config=config)

    elif 'albert-base-v2' in args['extract_model']:
        # Prepare config & tokenizer
        vocab_path, config_path = None, None
        tokenizer = AlbertTokenizer.from_pretrained(args['extract_model'])
        config = AlbertConfig.from_pretrained(args['extract_model'])

        # Instantiate model
        model = AlbertModel.from_pretrained(args['extract_model'], config=config)

    return model, tokenizer, vocab_path, config_path