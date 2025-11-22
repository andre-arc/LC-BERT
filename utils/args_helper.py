import sys, os

# Get the directory two levels up from the current file
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add to path and change directory
sys.path.append(parent_dir)
os.chdir(parent_dir)

from data_utils.ag_news.normal import BertNormalDataset, BertNormalDataLoader
from data_utils.ag_news.extraction import BertExtractionDataset, BertExtractionDataLoader
from data_utils.ag_news.whitening import BertWhiteningDataset, BertWhiteningDataLoader
from utils.forward_fn import forward_word_classification, modified_forward_word_classification
from utils.metrics import news_categorization_metrics_fn

from argparse import ArgumentParser

###
# args functions
###
def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.keys():
        if opts[key]:
            print('{:>30}: {:<50}'.format(key, opts[key]).center(80))
    print('=' * 80)

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="exp", help="Experiment name")
    parser.add_argument("--model_dir", type=str, default="save/", help="Model directory")
    parser.add_argument("--dataset", type=str, default='ag-news', help="Choose between ag-news, newsgroup")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Path, url or short name of the model")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Max number of tokens")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--eps", type=float, default=6.25e-5, help="EPS")
    parser.add_argument("--max_norm", type=float, default=10.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--device", type=str, default='cuda', help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--step_size", type=int, default=1, help="Step size")
    parser.add_argument("--early_stop", type=int, default=3, help="Step size")
    parser.add_argument("--gamma", type=float, default=0.5, help="Gamma")
    parser.add_argument("--debug", action='store_true', help="debugging mode")
    parser.add_argument("--force", action='store_true', help="force to rewrite experiment folder")
    parser.add_argument("--no_special_token", action='store_true', help="not adding special token as the input")
    parser.add_argument("--lower", action='store_true', help="lower case")
    parser.add_argument("--subset_percentage", default=100, help="subset percentage of dataset")

    args = vars(parser.parse_args())
    print_opts(args)
    return args

def get_eval_parser():
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="exp", help="Experiment name")
    parser.add_argument("--model_dir", type=str, default="./save", help="Model directory")
    parser.add_argument("--dataset", type=str, default='ag-news', help="Choose between ag-news, newsgroup")
    parser.add_argument("--model_type", type=str, default="bert-base-uncased", help="Type of the model")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Max number of tokens")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--debug", action='store_true', help="debugging mode")
    parser.add_argument("--no_special_token", action='store_true', help="not adding special token as the input")
    parser.add_argument("--lower", action='store_true', help="lower case")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--device", type=str, default='cuda', help="Device (cuda or cpu)")

    args = vars(parser.parse_args())
    print_opts(args)
    return args

#TODO: Need to change it into a json or something else that are easily extendable
def append_dataset_args(args):    
    if args['dataset'] == "ag-news-normal":
        args['task'] = 'sequence_classification'
        args['num_labels'] = BertNormalDataset.NUM_LABELS
        args['dataset_class'] = BertNormalDataset
        args['dataloader_class'] = BertNormalDataLoader
        args['forward_fn'] = forward_word_classification
        args['metrics_fn'] = news_categorization_metrics_fn
        args['valid_criterion'] = 'F1'
        args['train_set_path'] = './dataset/ag-news/train.csv'
        args['valid_set_path'] = './dataset/ag-news/valid.csv'
        args['test_set_path'] = './dataset/ag-news/test.csv'
        args['vocab_path']  = ""
        args['k_fold'] = 1
        # args['word_tokenizer_class'] = TweetTokenizer
    elif args['dataset'] == "ag-news-bert-extraction":
        args['task'] = 'sequence_classification'
        args['num_labels'] = BertExtractionDataset.NUM_LABELS
        args['dataset_class'] = BertExtractionDataset
        args['dataloader_class'] = BertExtractionDataLoader
        args['extract_model'] = 'bert-base-uncased'
        args['forward_fn'] = modified_forward_word_classification
        args['metrics_fn'] = news_categorization_metrics_fn
        args['valid_criterion'] = 'F1'
        args['train_set_path'] = './dataset/ag-news/train.csv'
        args['valid_set_path'] = './dataset/ag-news/valid.csv'
        args['test_set_path'] = './dataset/ag-news/test.csv'
        args['vocab_path']  = ""
        args['k_fold'] = 1
        # args['word_tokenizer_class'] = TweetTokenizer
    elif args['dataset'] == "ag-news-roberta-extraction":
        args['task'] = 'sequence_classification'
        args['num_labels'] = BertExtractionDataset.NUM_LABELS
        args['dataset_class'] = BertExtractionDataset
        args['dataloader_class'] = BertExtractionDataLoader
        args['extract_model'] = 'roberta-base'
        args['forward_fn'] = modified_forward_word_classification
        args['metrics_fn'] = news_categorization_metrics_fn
        args['valid_criterion'] = 'F1'
        args['train_set_path'] = './dataset/ag-news/train.csv'
        args['valid_set_path'] = './dataset/ag-news/valid.csv'
        args['test_set_path'] = './dataset/ag-news/test.csv'
        args['vocab_path']  = ""
        args['k_fold'] = 1
        # args['word_tokenizer_class'] = TweetTokenizer
    elif args['dataset'] == "ag-news-bert-whitening-svd":
        args['task'] = 'sequence_classification'
        args['num_labels'] = BertWhiteningDataset.NUM_LABELS
        args['dataset_class'] = BertWhiteningDataset
        args['dataloader_class'] = BertWhiteningDataLoader
        args['extract_model'] = 'bert-base-uncased'
        args['dim_technique'] = 'svd'
        args['forward_fn'] = modified_forward_word_classification
        args['metrics_fn'] = news_categorization_metrics_fn
        args['valid_criterion'] = 'F1'
        args['train_set_path'] = './dataset/ag-news/train.csv'
        args['valid_set_path'] = './dataset/ag-news/valid.csv'
        args['test_set_path'] = './dataset/ag-news/test.csv'
        args['vocab_path']  = ""
        args['k_fold'] = 1
    elif args['dataset'] == "ag-news-bert-whitening-eigen":
        args['task'] = 'sequence_classification'
        args['num_labels'] = BertWhiteningDataset.NUM_LABELS
        args['dataset_class'] = BertWhiteningDataset
        args['dataloader_class'] = BertWhiteningDataLoader
        args['extract_model'] = 'bert-base-uncased'
        args['dim_technique'] = 'eigen'
        args['forward_fn'] = modified_forward_word_classification
        args['metrics_fn'] = news_categorization_metrics_fn
        args['valid_criterion'] = 'F1'
        args['train_set_path'] = './dataset/ag-news/train.csv'
        args['valid_set_path'] = './dataset/ag-news/valid.csv'
        args['test_set_path'] = './dataset/ag-news/test.csv'
        args['vocab_path']  = ""
        args['k_fold'] = 1
    elif args['dataset'] == "ag-news-bert-whitening-pca":
        args['task'] = 'sequence_classification'
        args['num_labels'] = BertWhiteningDataset.NUM_LABELS
        args['dataset_class'] = BertWhiteningDataset
        args['dataloader_class'] = BertWhiteningDataLoader
        args['extract_model'] = 'bert-base-uncased'
        args['dim_technique'] = 'pca'
        args['forward_fn'] = modified_forward_word_classification
        args['metrics_fn'] = news_categorization_metrics_fn
        args['valid_criterion'] = 'F1'
        args['train_set_path'] = './dataset/ag-news/train.csv'
        args['valid_set_path'] = './dataset/ag-news/valid.csv'
        args['test_set_path'] = './dataset/ag-news/test.csv'
        args['vocab_path']  = ""
        args['k_fold'] = 1
    elif args['dataset'] == "ag-news-bert-whitening-pca-svd":
        args['task'] = 'sequence_classification'
        args['num_labels'] = BertWhiteningDataset.NUM_LABELS
        args['dataset_class'] = BertWhiteningDataset
        args['dataloader_class'] = BertWhiteningDataLoader
        args['extract_model'] = 'bert-base-uncased'
        args['dim_technique'] = 'pca-svd'
        args['forward_fn'] = modified_forward_word_classification
        args['metrics_fn'] = news_categorization_metrics_fn
        args['valid_criterion'] = 'F1'
        args['train_set_path'] = './dataset/ag-news/train.csv'
        args['valid_set_path'] = './dataset/ag-news/valid.csv'
        args['test_set_path'] = './dataset/ag-news/test.csv'
        args['vocab_path']  = ""
        args['k_fold'] = 1
    elif args['dataset'] == "ag-news-bert-whitening-zca":
        args['task'] = 'sequence_classification'
        args['num_labels'] = BertWhiteningDataset.NUM_LABELS
        args['dataset_class'] = BertWhiteningDataset
        args['dataloader_class'] = BertWhiteningDataLoader
        args['extract_model'] = 'bert-base-uncased'
        args['dim_technique'] = 'zca'
        args['forward_fn'] = modified_forward_word_classification
        args['metrics_fn'] = news_categorization_metrics_fn
        args['valid_criterion'] = 'F1'
        args['train_set_path'] = './dataset/ag-news/train.csv'
        args['valid_set_path'] = './dataset/ag-news/valid.csv'
        args['test_set_path'] = './dataset/ag-news/test.csv'
        args['vocab_path']  = ""
        args['k_fold'] = 1
        # args['word_tokenizer_class'] = TweetTokenizer
    elif args['dataset'] == "ag-news-bert-whitening-zca-svd":
        args['task'] = 'sequence_classification'
        args['num_labels'] = BertWhiteningDataset.NUM_LABELS
        args['dataset_class'] = BertWhiteningDataset
        args['dataloader_class'] = BertWhiteningDataLoader
        args['extract_model'] = 'bert-base-uncased'
        args['dim_technique'] = 'zca-svd'
        args['forward_fn'] = modified_forward_word_classification
        args['metrics_fn'] = news_categorization_metrics_fn
        args['valid_criterion'] = 'F1'
        args['train_set_path'] = './dataset/ag-news/train.csv'
        args['valid_set_path'] = './dataset/ag-news/valid.csv'
        args['test_set_path'] = './dataset/ag-news/test.csv'
        args['vocab_path']  = ""
        args['k_fold'] = 1
        # args['word_tokenizer_class'] = TweetTokenizer
    elif args['dataset'] == "ag-news-roberta-whitening-svd":
        args['task'] = 'sequence_classification'
        args['num_labels'] = BertWhiteningDataset.NUM_LABELS
        args['dataset_class'] = BertWhiteningDataset
        args['dataloader_class'] = BertWhiteningDataLoader
        args['extract_model'] = 'roberta-base'
        args['dim_technique'] = 'svd'
        args['forward_fn'] = modified_forward_word_classification
        args['metrics_fn'] = news_categorization_metrics_fn
        args['valid_criterion'] = 'F1'
        args['train_set_path'] = './dataset/ag-news/train.csv'
        args['valid_set_path'] = './dataset/ag-news/valid.csv'
        args['test_set_path'] = './dataset/ag-news/test.csv'
        args['vocab_path']  = ""
        args['k_fold'] = 1
    elif args['dataset'] == "ag-news-roberta-whitening-eigen":
        args['task'] = 'sequence_classification'
        args['num_labels'] = BertWhiteningDataset.NUM_LABELS
        args['dataset_class'] = BertWhiteningDataset
        args['dataloader_class'] = BertWhiteningDataLoader
        args['extract_model'] = 'roberta-base'
        args['dim_technique'] = 'eigen'
        args['forward_fn'] = modified_forward_word_classification
        args['metrics_fn'] = news_categorization_metrics_fn
        args['valid_criterion'] = 'F1'
        args['train_set_path'] = './dataset/ag-news/train.csv'
        args['valid_set_path'] = './dataset/ag-news/valid.csv'
        args['test_set_path'] = './dataset/ag-news/test.csv'
        args['vocab_path']  = ""
        args['k_fold'] = 1
    elif args['dataset'] == "ag-news-roberta-whitening-pca":
        args['task'] = 'sequence_classification'
        args['num_labels'] = BertWhiteningDataset.NUM_LABELS
        args['dataset_class'] = BertWhiteningDataset
        args['dataloader_class'] = BertWhiteningDataLoader
        args['extract_model'] = 'roberta-base'
        args['dim_technique'] = 'pca'
        args['forward_fn'] = modified_forward_word_classification
        args['metrics_fn'] = news_categorization_metrics_fn
        args['valid_criterion'] = 'F1'
        args['train_set_path'] = './dataset/ag-news/train.csv'
        args['valid_set_path'] = './dataset/ag-news/valid.csv'
        args['test_set_path'] = './dataset/ag-news/test.csv'
        args['vocab_path']  = ""
        args['k_fold'] = 1
    elif args['dataset'] == "ag-news-roberta-whitening-pca-svd":
        args['task'] = 'sequence_classification'
        args['num_labels'] = BertWhiteningDataset.NUM_LABELS
        args['dataset_class'] = BertWhiteningDataset
        args['dataloader_class'] = BertWhiteningDataLoader
        args['extract_model'] = 'roberta-base'
        args['dim_technique'] = 'pca-svd'
        args['forward_fn'] = modified_forward_word_classification
        args['metrics_fn'] = news_categorization_metrics_fn
        args['valid_criterion'] = 'F1'
        args['train_set_path'] = './dataset/ag-news/train.csv'
        args['valid_set_path'] = './dataset/ag-news/valid.csv'
        args['test_set_path'] = './dataset/ag-news/test.csv'
        args['vocab_path']  = ""
        args['k_fold'] = 1
    elif args['dataset'] == "ag-news-roberta-whitening-zca":
        args['task'] = 'sequence_classification'
        args['num_labels'] = BertWhiteningDataset.NUM_LABELS
        args['dataset_class'] = BertWhiteningDataset
        args['dataloader_class'] = BertWhiteningDataLoader
        args['extract_model'] = 'roberta-base'
        args['dim_technique'] = 'zca'
        args['forward_fn'] = modified_forward_word_classification
        args['metrics_fn'] = news_categorization_metrics_fn
        args['valid_criterion'] = 'F1'
        args['train_set_path'] = './dataset/ag-news/train.csv'
        args['valid_set_path'] = './dataset/ag-news/valid.csv'
        args['test_set_path'] = './dataset/ag-news/test.csv'
        args['vocab_path']  = ""
        args['k_fold'] = 1
        # args['word_tokenizer_class'] = TweetTokenizer
    elif args['dataset'] == "ag-news-roberta-whitening-zca-svd":
        args['task'] = 'sequence_classification'
        args['num_labels'] = BertWhiteningDataset.NUM_LABELS
        args['dataset_class'] = BertWhiteningDataset
        args['dataloader_class'] = BertWhiteningDataLoader
        args['extract_model'] = 'roberta-base'
        args['dim_technique'] = 'zca-svd'
        args['forward_fn'] = modified_forward_word_classification
        args['metrics_fn'] = news_categorization_metrics_fn
        args['valid_criterion'] = 'F1'
        args['train_set_path'] = './dataset/ag-news/train.csv'
        args['valid_set_path'] = './dataset/ag-news/valid.csv'
        args['test_set_path'] = './dataset/ag-news/test.csv'
        args['vocab_path']  = ""
        args['k_fold'] = 1
        # args['word_tokenizer_class'] = TweetTokenizer
    # elif args['dataset'] == "absa-airy":
    #     args['task'] = 'multi_label_classification'
    #     args['num_labels'] = AspectBasedSentimentAnalysisAiryDataset.NUM_LABELS
    #     args['dataset_class'] = AspectBasedSentimentAnalysisAiryDataset
    #     args['dataloader_class'] = AspectBasedSentimentAnalysisDataLoader
    #     args['forward_fn'] = forward_sequence_multi_classification
    #     args['metrics_fn'] = absa_metrics_fn
    #     args['valid_criterion'] = 'F1'
    #     args['train_set_path'] = './dataset/hoasa_absa-airy/train_preprocess.csv'
    #     args['valid_set_path'] = './dataset/hoasa_absa-airy/valid_preprocess.csv'
    #     args['test_set_path'] = './dataset/hoasa_absa-airy/test_preprocess_masked_label.csv'
    #     args['vocab_path'] = "./dataset/hoasa_absa-airy/vocab_uncased.txt"
    #     args['embedding_path'] = {
    #         'fasttext-cc-id-300-no-oov-uncased': './embeddings/fasttext-cc-id/cc.id.300_no-oov_absa-airy_uncased.txt',
    #         'fasttext-4B-id-300-no-oov-uncased': './embeddings/fasttext-4B-id-uncased/fasttext.4B.id.300.epoch5_uncased_no-oov_absa-airy_uncased.txt'
    #     }
    #     args['k_fold'] = 1
    #     args['word_tokenizer_class'] = TweetTokenizer
    # elif args['dataset'] == "term-extraction-airy":
    #     args['task'] = 'token_classification'
    #     args['num_labels'] = AspectExtractionDataset.NUM_LABELS
    #     args['dataset_class'] = AspectExtractionDataset
    #     args['dataloader_class'] = AspectExtractionDataLoader
    #     args['forward_fn'] = forward_word_classification
    #     args['metrics_fn'] = aspect_extraction_metrics_fn
    #     args['valid_criterion'] = 'F1'
    #     args['train_set_path'] = './dataset/terma_term-extraction-airy/train_preprocess.txt'
    #     args['valid_set_path'] = './dataset/terma_term-extraction-airy/valid_preprocess.txt'
    #     args['test_set_path'] = './dataset/terma_term-extraction-airy/test_preprocess_masked_label.txt'
    #     args['vocab_path'] = "./dataset/terma_term-extraction-airy/vocab_uncased.txt"
    #     args['embedding_path'] = {
    #         'fasttext-cc-id-300-no-oov-uncased': './embeddings/fasttext-cc-id/cc.id.300_no-oov_term-extraction-airy_uncased.txt',
    #         'fasttext-4B-id-300-no-oov-uncased': './embeddings/fasttext-4B-id-uncased/fasttext.4B.id.300.epoch5_uncased_no-oov_term-extraction-airy_uncased.txt'
    #     }
    #     args['k_fold'] = 1
    #     args['word_tokenizer_class'] = TweetTokenizer
    # elif args['dataset'] == "ner-grit":
    #     args['task'] = 'token_classification'
    #     args['num_labels'] = NerGritDataset.NUM_LABELS
    #     args['dataset_class'] = NerGritDataset
    #     args['dataloader_class'] = NerDataLoader
    #     args['forward_fn'] = forward_word_classification
    #     args['metrics_fn'] = ner_metrics_fn
    #     args['valid_criterion'] = 'F1'
    #     args['train_set_path'] = './dataset/nergrit_ner-grit/train_preprocess.txt'
    #     args['valid_set_path'] = './dataset/nergrit_ner-grit/valid_preprocess.txt'
    #     args['test_set_path'] = './dataset/nergrit_ner-grit/test_preprocess_masked_label.txt'
    #     args['vocab_path'] = "./dataset/nergrit_ner-grit/vocab_uncased.txt"
    #     args['embedding_path'] = {
    #         'fasttext-cc-id-300-no-oov-uncased': './embeddings/fasttext-cc-id/cc.id.300_no-oov_ner-grit_uncased.txt',
    #         'fasttext-4B-id-300-no-oov-uncased': './embeddings/fasttext-4B-id-uncased/fasttext.4B.id.300.epoch5_uncased_no-oov_ner-grit_uncased.txt'
    #     }
    #     args['k_fold'] = 1
    #     args['word_tokenizer_class'] = TweetTokenizer
    # elif args['dataset'] == "pos-idn":
    #     args['task'] = 'token_classification'
    #     args['num_labels'] = PosTagIdnDataset.NUM_LABELS
    #     args['dataset_class'] = PosTagIdnDataset
    #     args['dataloader_class'] = PosTagDataLoader
    #     args['forward_fn'] = forward_word_classification
    #     args['metrics_fn'] = pos_tag_metrics_fn
    #     args['valid_criterion'] = 'F1'
    #     args['train_set_path'] = './dataset/bapos_pos-idn/train_preprocess.txt'
    #     args['valid_set_path'] = './dataset/bapos_pos-idn/valid_preprocess.txt'
    #     args['test_set_path'] = './dataset/bapos_pos-idn/test_preprocess_masked_label.txt'
    #     args['vocab_path'] = "./dataset/bapos_pos-idn/vocab_uncased.txt"
    #     args['embedding_path'] = {
    #         'fasttext-cc-id-300-no-oov-uncased': './embeddings/fasttext-cc-id/cc.id.300_no-oov_pos-idn_uncased.txt',
    #         'fasttext-4B-id-300-no-oov-uncased': './embeddings/fasttext-4B-id-uncased/fasttext.4B.id.300.epoch5_uncased_no-oov_pos-idn_uncased.txt'
    #     }
    #     args['k_fold'] = 1
    #     args['word_tokenizer_class'] = TweetTokenizer
    # elif args['dataset'] == "entailment-ui":
    #     args['task'] = 'sequence_classification'
    #     args['num_labels'] = EntailmentDataset.NUM_LABELS
    #     args['dataset_class'] = EntailmentDataset
    #     args['dataloader_class'] = EntailmentDataLoader
    #     args['forward_fn'] = forward_sequence_classification
    #     args['metrics_fn'] = entailment_metrics_fn
    #     args['valid_criterion'] = 'F1'
    #     args['train_set_path'] = './dataset/wrete_entailment-ui/train_preprocess.csv'
    #     args['valid_set_path'] = './dataset/wrete_entailment-ui/valid_preprocess.csv'
    #     args['test_set_path'] = './dataset/wrete_entailment-ui/test_preprocess_masked_label.csv'
    #     args['vocab_path'] = "./dataset/wrete_entailment-ui/vocab_uncased.txt"
    #     args['embedding_path'] = {
    #         'fasttext-cc-id-300-no-oov-uncased': './embeddings/fasttext-cc-id/cc.id.300_no-oov_entailment-ui_uncased.txt',
    #         'fasttext-4B-id-300-no-oov-uncased': './embeddings/fasttext-4B-id-uncased/fasttext.4B.id.300.epoch5_uncased_no-oov_entailment-ui_uncased.txt'
    #     }
    #     args['k_fold'] = 1
    #     args['word_tokenizer_class'] = TweetTokenizer
    # elif args['dataset'] == "doc-sentiment-prosa":
    #     args['task'] = 'sequence_classification'
    #     args['num_labels'] = DocumentSentimentDataset.NUM_LABELS
    #     args['dataset_class'] = DocumentSentimentDataset
    #     args['dataloader_class'] = DocumentSentimentDataLoader
    #     args['forward_fn'] = forward_sequence_classification
    #     args['metrics_fn'] = document_sentiment_metrics_fn
    #     args['valid_criterion'] = 'F1'
    #     args['train_set_path'] = './dataset/smsa_doc-sentiment-prosa/train_preprocess.tsv'
    #     args['valid_set_path'] = './dataset/smsa_doc-sentiment-prosa/valid_preprocess.tsv'
    #     args['test_set_path'] = './dataset/smsa_doc-sentiment-prosa/test_preprocess_masked_label.tsv'
    #     args['vocab_path'] = "./dataset/smsa_doc-sentiment-prosa/vocab_uncased.txt"
    #     args['embedding_path'] = {
    #         'fasttext-cc-id-300-no-oov-uncased': './embeddings/fasttext-cc-id/cc.id.300_no-oov_doc-sentiment-prosa_uncased.txt',
    #         'fasttext-4B-id-300-no-oov-uncased': './embeddings/fasttext-4B-id-uncased/fasttext.4B.id.300.epoch5_uncased_no-oov_doc-sentiment-prosa_uncased.txt'
    #     }
    #     args['k_fold'] = 1
    #     args['word_tokenizer_class'] = TweetTokenizer
    # elif args['dataset'] == "keyword-extraction-prosa":
    #     args['task'] = 'token_classification'
    #     args['num_labels'] = KeywordExtractionDataset.NUM_LABELS
    #     args['dataset_class'] = KeywordExtractionDataset
    #     args['dataloader_class'] = KeywordExtractionDataLoader
    #     args['forward_fn'] = forward_word_classification
    #     args['metrics_fn'] = keyword_extraction_metrics_fn
    #     args['valid_criterion'] = 'F1'
    #     args['train_set_path'] = './dataset/keps_keyword-extraction-prosa/train_preprocess.txt'
    #     args['valid_set_path'] = './dataset/keps_keyword-extraction-prosa/valid_preprocess.txt'
    #     args['test_set_path'] = './dataset/keps_keyword-extraction-prosa/test_preprocess_masked_label.txt'
    #     args['vocab_path'] = "./dataset/keps_keyword-extraction-prosa/vocab_uncased.txt"
    #     args['embedding_path'] = {
    #         'fasttext-cc-id-300-no-oov-uncased': './embeddings/fasttext-cc-id/cc.id.300_no-oov_keyword-extraction-prosa_uncased.txt',
    #         'fasttext-4B-id-300-no-oov-uncased': './embeddings/fasttext-4B-id-uncased/fasttext.4B.id.300.epoch5_uncased_no-oov_keyword-extraction-prosa_uncased.txt'
    #     }
    #     args['k_fold'] = 1
    #     args['word_tokenizer_class'] = WordSplitTokenizer
    # elif args['dataset'] == "qa-factoid-itb":
    #     args['task'] = 'token_classification'
    #     args['num_labels'] = QAFactoidDataset.NUM_LABELS
    #     args['dataset_class'] = QAFactoidDataset
    #     args['dataloader_class'] = QAFactoidDataLoader
    #     args['forward_fn'] = forward_word_classification
    #     args['metrics_fn'] = qa_factoid_metrics_fn
    #     args['valid_criterion'] = 'F1'
    #     args['train_set_path'] = './dataset/facqa_qa-factoid-itb/train_preprocess.csv'
    #     args['valid_set_path'] = './dataset/facqa_qa-factoid-itb/valid_preprocess.csv'
    #     args['test_set_path'] = './dataset/facqa_qa-factoid-itb/test_preprocess_masked_label.csv'
    #     args['vocab_path'] = "./dataset/facqa_qa-factoid-itb/vocab_uncased.txt"
    #     args['embedding_path'] = {
    #         'fasttext-cc-id-300-no-oov-uncased': './embeddings/fasttext-cc-id/cc.id.300_no-oov_qa-factoid-itb_uncased.txt',
    #         'fasttext-4B-id-300-no-oov-uncased': './embeddings/fasttext-4B-id-uncased/fasttext.4B.id.300.epoch5_uncased_no-oov_qa-factoid-itb_uncased.txt'
    #     }
    #     args['k_fold'] = 1
    #     args['word_tokenizer_class'] = TweetTokenizer
    # elif args['dataset'] == "ner-prosa":
    #     args['task'] = 'token_classification'
    #     args['num_labels'] = NerProsaDataset.NUM_LABELS
    #     args['dataset_class'] = NerProsaDataset
    #     args['dataloader_class'] = NerDataLoader
    #     args['forward_fn'] = forward_word_classification
    #     args['metrics_fn'] = ner_metrics_fn
    #     args['valid_criterion'] = 'F1'
    #     args['train_set_path'] = './dataset/nerp_ner-prosa/train_preprocess.txt'
    #     args['valid_set_path'] = './dataset/nerp_ner-prosa/valid_preprocess.txt'
    #     args['test_set_path'] = './dataset/nerp_ner-prosa/test_preprocess_masked_label.txt'
    #     args['vocab_path'] = "./dataset/nerp_ner-prosa/vocab_uncased.txt"
    #     args['embedding_path'] = {
    #         'fasttext-cc-id-300-no-oov-uncased': './embeddings/fasttext-cc-id/cc.id.300_no-oov_ner-prosa_uncased.txt',
    #         'fasttext-4B-id-300-no-oov-uncased': './embeddings/fasttext-4B-id-uncased/fasttext.4B.id.300.epoch5_uncased_no-oov_ner-prosa_uncased.txt'
    #     }
    #     args['k_fold'] = 1
    #     args['word_tokenizer_class'] = TweetTokenizer
    # elif args['dataset'] == "pos-prosa":
    #     args['task'] = 'token_classification'
    #     args['num_labels'] = PosTagProsaDataset.NUM_LABELS
    #     args['dataset_class'] = PosTagProsaDataset
    #     args['dataloader_class'] = PosTagDataLoader
    #     args['forward_fn'] = forward_word_classification
    #     args['metrics_fn'] = pos_tag_metrics_fn
    #     args['valid_criterion'] = 'F1'
    #     args['train_set_path'] = './dataset/posp_pos-prosa/train_preprocess.txt'
    #     args['valid_set_path'] = './dataset/posp_pos-prosa/valid_preprocess.txt'
    #     args['test_set_path'] = './dataset/posp_pos-prosa/test_preprocess_masked_label.txt'
    #     args['vocab_path'] = "./dataset/posp_pos-prosa/vocab_uncased.txt"
    #     args['embedding_path'] = {
    #         'fasttext-cc-id-300-no-oov-uncased': './embeddings/fasttext-cc-id/cc.id.300_no-oov_pos-prosa_uncased.txt',
    #         'fasttext-4B-id-300-no-oov-uncased': './embeddings/fasttext-4B-id-uncased/fasttext.4B.id.300.epoch5_uncased_no-oov_pos-prosa_uncased.txt'
    #     }
    #     args['k_fold'] = 1
    #     args['word_tokenizer_class'] = TweetTokenizer
    # elif args['dataset'] == "absa-prosa":
    #     args['task'] = 'multi_label_classification'
    #     args['num_labels'] = AspectBasedSentimentAnalysisProsaDataset.NUM_LABELS
    #     args['dataset_class'] = AspectBasedSentimentAnalysisProsaDataset
    #     args['dataloader_class'] = AspectBasedSentimentAnalysisDataLoader
    #     args['forward_fn'] = forward_sequence_multi_classification
    #     args['metrics_fn'] = absa_metrics_fn
    #     args['valid_criterion'] = 'F1'
    #     args['train_set_path'] = './dataset/casa_absa-prosa/train_preprocess.csv'
    #     args['valid_set_path'] = './dataset/casa_absa-prosa/valid_preprocess.csv'
    #     args['test_set_path'] = './dataset/casa_absa-prosa/test_preprocess_masked_label.csv'
    #     args['vocab_path'] = "./dataset/casa_absa-prosa/vocab_uncased.txt"
    #     args['embedding_path'] = {
    #         'fasttext-cc-id-300-no-oov-uncased': './embeddings/fasttext-cc-id/cc.id.300_no-oov_absa-prosa_uncased.txt',
    #         'fasttext-4B-id-300-no-oov-uncased': './embeddings/fasttext-4B-id-uncased/fasttext.4B.id.300.epoch5_uncased_no-oov_absa-prosa_uncased.txt'
    #     }
    #     args['k_fold'] = 1
    #     args['word_tokenizer_class'] = TweetTokenizer
    else:
        raise ValueError(f'Unknown dataset name `{args["dataset"]}`')
    return args