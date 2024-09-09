import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..\/"))
os.chdir(os.path.join(os.path.dirname(__file__), "..\/"))

from data_utils.ag_news.normal import BertNormalDataset, BertNormalDataLoader
from data_utils.ag_news.extraction import BertExtractionDataset, BertExtractionDataLoader
from data_utils.ag_news.whitening import BertWhiteningDataset, BertWhiteningDataLoader
from utils.forward_fn import forward_word_classification, modified_forward_word_classification
from utils.metrics import news_categorization_metrics_fn
from utils.functions import load_model, load_extraction_model

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR

import time

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# Set GPU memory limit to 50%
# torch.cuda.set_per_process_memory_fraction(0.16)
args = ['dataset']
report = []

dataset_configs = {
    "ag-news-normal": {
        'task': 'sequence_classification',
        'num_labels': BertNormalDataset.NUM_LABELS,
        'dataset_class': BertNormalDataset,
        'dataloader_class': BertNormalDataLoader,
        'forward_fn': forward_word_classification,
        'metrics_fn': news_categorization_metrics_fn,
        'valid_criterion': 'F1',
        'train_set_path': './dataset/ag-news/train.csv',
        'valid_set_path': './dataset/ag-news/valid.csv',
        'test_set_path': './dataset/ag-news/test.csv',
        'vocab_path': "",
        'k_fold': 1,
        # 'word_tokenizer_class': TweetTokenizer
    },
    # "ag-news-bert-extraction": {
    #     'task': 'sequence_classification',
    #     'num_labels': BertExtractionDataset.NUM_LABELS,
    #     'dataset_class': BertExtractionDataset,
    #     'dataloader_class': BertExtractionDataLoader,
    #     'extract_model': 'bert-base-uncased',
    #     'forward_fn': modified_forward_word_classification,
    #     'metrics_fn': news_categorization_metrics_fn,
    #     'valid_criterion': 'F1',
    #     'train_set_path': './dataset/ag-news/train.csv',
    #     'valid_set_path': './dataset/ag-news/valid.csv',
    #     'test_set_path': './dataset/ag-news/test.csv',
    #     'vocab_path': "",
    #     'k_fold': 1,
    #     # 'word_tokenizer_class': TweetTokenizer
    # },
    # "ag-news-roberta-extraction": {
    #     'task': 'sequence_classification',
    #     'num_labels': BertExtractionDataset.NUM_LABELS,
    #     'dataset_class': BertExtractionDataset,
    #     'dataloader_class': BertExtractionDataLoader,
    #     'extract_model': 'roberta-base',
    #     'forward_fn': modified_forward_word_classification,
    #     'metrics_fn': news_categorization_metrics_fn,
    #     'valid_criterion': 'F1',
    #     'train_set_path': './dataset/ag-news/train.csv',
    #     'valid_set_path': './dataset/ag-news/valid.csv',
    #     'test_set_path': './dataset/ag-news/test.csv',
    #     'vocab_path': "",
    #     'k_fold': 1,
    #     # 'word_tokenizer_class': TweetTokenizer
    # },
    "ag-news-bert-whitening-svd": {
        'task': 'sequence_classification',
        'num_labels': BertWhiteningDataset.NUM_LABELS,
        'dataset_class': BertWhiteningDataset,
        'dataloader_class': BertWhiteningDataLoader,
        'extract_model': 'bert-base-uncased',
        'dim_technique': 'svd',
        'forward_fn': modified_forward_word_classification,
        'metrics_fn': news_categorization_metrics_fn,
        'valid_criterion': 'F1',
        'train_set_path': './dataset/ag-news/train.csv',
        'valid_set_path': './dataset/ag-news/valid.csv',
        'test_set_path': './dataset/ag-news/test.csv',
        'vocab_path': "",
        'k_fold': 1
    },
    # "ag-news-bert-whitening-eigen": {
    #     'task': 'sequence_classification',
    #     'num_labels': BertWhiteningDataset.NUM_LABELS,
    #     'dataset_class': BertWhiteningDataset,
    #     'dataloader_class': BertWhiteningDataLoader,
    #     'extract_model': 'bert-base-uncased',
    #     'dim_technique': 'eigen',
    #     'forward_fn': modified_forward_word_classification,
    #     'metrics_fn': news_categorization_metrics_fn,
    #     'valid_criterion': 'F1',
    #     'train_set_path': './dataset/ag-news/train.csv',
    #     'valid_set_path': './dataset/ag-news/valid.csv',
    #     'test_set_path': './dataset/ag-news/test.csv',
    #     'vocab_path': "",
    #     'k_fold': 1
    # },
    "ag-news-bert-whitening-pca": {
        'task': 'sequence_classification',
        'num_labels': BertWhiteningDataset.NUM_LABELS,
        'dataset_class': BertWhiteningDataset,
        'dataloader_class': BertWhiteningDataLoader,
        'extract_model': 'bert-base-uncased',
        'dim_technique': 'pca',
        'forward_fn': modified_forward_word_classification,
        'metrics_fn': news_categorization_metrics_fn,
        'valid_criterion': 'F1',
        'train_set_path': './dataset/ag-news/train.csv',
        'valid_set_path': './dataset/ag-news/valid.csv',
        'test_set_path': './dataset/ag-news/test.csv',
        'vocab_path': "",
        'k_fold': 1
    },
    # "ag-news-bert-whitening-pca-svd": {
    #     'task': 'sequence_classification',
    #     'num_labels': BertWhiteningDataset.NUM_LABELS,
    #     'dataset_class': BertWhiteningDataset,
    #     'dataloader_class': BertWhiteningDataLoader,
    #     'extract_model': 'bert-base-uncased',
    #     'dim_technique': 'pca-svd',
    #     'forward_fn': modified_forward_word_classification,
    #     'metrics_fn': news_categorization_metrics_fn,
    #     'valid_criterion': 'F1',
    #     'train_set_path': './dataset/ag-news/train.csv',
    #     'valid_set_path': './dataset/ag-news/valid.csv',
    #     'test_set_path': './dataset/ag-news/test.csv',
    #     'vocab_path': "",
    #     'k_fold': 1
    # },
    "ag-news-bert-whitening-zca": {
        'task': 'sequence_classification',
        'num_labels': BertWhiteningDataset.NUM_LABELS,
        'dataset_class': BertWhiteningDataset,
        'dataloader_class': BertWhiteningDataLoader,
        'extract_model': 'bert-base-uncased',
        'dim_technique': 'zca',
        'forward_fn': modified_forward_word_classification,
        'metrics_fn': news_categorization_metrics_fn,
        'valid_criterion': 'F1',
        'train_set_path': './dataset/ag-news/train.csv',
        'valid_set_path': './dataset/ag-news/valid.csv',
        'test_set_path': './dataset/ag-news/test.csv',
        'vocab_path': "",
        'k_fold': 1
    },
    # "ag-news-bert-whitening-zca-svd": {
    #     'task': 'sequence_classification',
    #     'num_labels': BertWhiteningDataset.NUM_LABELS,
    #     'dataset_class': BertWhiteningDataset,
    #     'dataloader_class': BertWhiteningDataLoader,
    #     'extract_model': 'bert-base-uncased',
    #     'dim_technique': 'zca-svd',
    #     'forward_fn': modified_forward_word_classification,
    #     'metrics_fn': news_categorization_metrics_fn,
    #     'valid_criterion': 'F1',
    #     'train_set_path': './dataset/ag-news/train.csv',
    #     'valid_set_path': './dataset/ag-news/valid.csv',
    #     'test_set_path': './dataset/ag-news/test.csv',
    #     'vocab_path': "",
    #     'k_fold': 1
    # },
    "ag-news-roberta-whitening-svd": {
        'task': 'sequence_classification',
        'num_labels': BertWhiteningDataset.NUM_LABELS,
        'dataset_class': BertWhiteningDataset,
        'dataloader_class': BertWhiteningDataLoader,
        'extract_model': 'roberta-base',
        'dim_technique': 'svd',
        'forward_fn': modified_forward_word_classification,
        'metrics_fn': news_categorization_metrics_fn,
        'valid_criterion': 'F1',
        'train_set_path': './dataset/ag-news/train.csv',
        'valid_set_path': './dataset/ag-news/valid.csv',
        'test_set_path': './dataset/ag-news/test.csv',
        'vocab_path': "",
        'k_fold': 1
    },
    # "ag-news-roberta-whitening-eigen": {
    #     'task': 'sequence_classification',
    #     'num_labels': BertWhiteningDataset.NUM_LABELS,
    #     'dataset_class': BertWhiteningDataset,
    #     'dataloader_class': BertWhiteningDataLoader,
    #     'extract_model': 'roberta-base',
    #     'dim_technique': 'eigen',
    #     'forward_fn': modified_forward_word_classification,
    #     'metrics_fn': news_categorization_metrics_fn,
    #     'valid_criterion': 'F1',
    #     'train_set_path': './dataset/ag-news/train.csv',
    #     'valid_set_path': './dataset/ag-news/valid.csv',
    #     'test_set_path': './dataset/ag-news/test.csv',
    #     'vocab_path': "",
    #     'k_fold': 1
    # },
    "ag-news-roberta-whitening-pca": {
        'task': 'sequence_classification',
        'num_labels': BertWhiteningDataset.NUM_LABELS,
        'dataset_class': BertWhiteningDataset,
        'dataloader_class': BertWhiteningDataLoader,
        'extract_model': 'roberta-base',
        'dim_technique': 'pca',
        'forward_fn': modified_forward_word_classification,
        'metrics_fn': news_categorization_metrics_fn,
        'valid_criterion': 'F1',
        'train_set_path': './dataset/ag-news/train.csv',
        'valid_set_path': './dataset/ag-news/valid.csv',
        'test_set_path': './dataset/ag-news/test.csv',
        'vocab_path': "",
        'k_fold': 1
    },
    # "ag-news-roberta-whitening-pca-svd": {
    #     'task': 'sequence_classification',
    #     'num_labels': BertWhiteningDataset.NUM_LABELS,
    #     'dataset_class': BertWhiteningDataset,
    #     'dataloader_class': BertWhiteningDataLoader,
    #     'extract_model': 'roberta-base',
    #     'dim_technique': 'pca-svd',
    #     'forward_fn': modified_forward_word_classification,
    #     'metrics_fn': news_categorization_metrics_fn,
    #     'valid_criterion': 'F1',
    #     'train_set_path': './dataset/ag-news/train.csv',
    #     'valid_set_path': './dataset/ag-news/valid.csv',
    #     'test_set_path': './dataset/ag-news/test.csv',
    #     'vocab_path': "",
    #     'k_fold': 1
    # },
    "ag-news-roberta-whitening-zca": {
        'task': 'sequence_classification',
        'num_labels': BertWhiteningDataset.NUM_LABELS,
        'dataset_class': BertWhiteningDataset,
        'dataloader_class': BertWhiteningDataLoader,
        'extract_model': 'roberta-base',
        'dim_technique': 'zca',
        'forward_fn': modified_forward_word_classification,
        'metrics_fn': news_categorization_metrics_fn,
        'valid_criterion': 'F1',
        'train_set_path': './dataset/ag-news/train.csv',
        'valid_set_path': './dataset/ag-news/valid.csv',
        'test_set_path': './dataset/ag-news/test.csv',
        'vocab_path': "",
        'k_fold': 1
    },
    # "ag-news-roberta-whitening-zca-svd": {
    #     'task': 'sequence_classification',
    #     'num_labels': BertWhiteningDataset.NUM_LABELS,
    #     'dataset_class': BertWhiteningDataset,
    #     'dataloader_class': BertWhiteningDataLoader,
    #     'extract_model': 'roberta-base',
    #     'dim_technique': 'zca-svd',
    #     'forward_fn': modified_forward_word_classification,
    #     'metrics_fn': news_categorization_metrics_fn,
    #     'valid_criterion': 'F1',
    #     'train_set_path': './dataset/ag-news/train.csv',
    #     'valid_set_path': './dataset/ag-news/valid.csv',
    #     'test_set_path': './dataset/ag-news/test.csv',
    #     'vocab_path': "",
    #     'k_fold': 1
    # }
}

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2  # Convert to MB
    return size_all_mb

for dataset_key, dataset_config in dataset_configs.items():
    benchmark_model = ['bert-base-uncased', 'roberta-base']
    modified_model = ['bilstm-dim-reduction', 'mlp-dim-reduction']

    if dataset_key == 'ag-news-normal':
        for model_name in benchmark_model :
            dataset_config['model_name'] = model_name
            # Load and prepare the model and data
            model, tokenizer, vocab_path, config_path = load_model(dataset_config)
            size = round(get_model_size(model), 3)
            report.append({'model': dataset_key+"|"+model_name, 'size': round(get_model_size(model), 3)})
    else :
        for model_name in modified_model :
            dataset_config['model_name'] = model_name
            # Load and prepare the model and data
            model, tokenizer, vocab_path, config_path = load_model(dataset_config)
            size = round(get_model_size(model), 3)
            report.append({'model': dataset_key+"|"+model_name, 'size': round(get_model_size(model), 3)})

# Convert the report to a DataFrame
report_df = pd.DataFrame(report)

# Define the CSV file path
csv_file_path = 'report.csv'

# Save the report DataFrame to a CSV file
report_df.to_csv(csv_file_path, index=False)

print(f"Report saved to {csv_file_path}")
