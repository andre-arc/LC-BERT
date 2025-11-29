import os
import shutil
from copy import deepcopy
import random
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from transformers import AdamW

from utils.functions import load_model, load_extraction_model
from utils.args_helper import get_parser, append_dataset_args
from torch.utils.data import DataLoader, random_split, Subset
import time

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

###
# modelling functions
###
def get_lr(args, optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def metrics_to_string(metric_dict):
    string_list = []
    for key, value in metric_dict.items():
        string_list.append('{}:{:.2f}'.format(key, value))
    return ' '.join(string_list)

def elapsed_timestamp_to_detail(elapsed_time):
    # Calculate hours, minutes, and seconds
    hours, remainder = divmod(int(elapsed_time), 3600)  # 3600 seconds in an hour
    minutes, seconds = divmod(remainder, 60)
    # Display the detailed elapsed time
    return f"{hours} hours, {minutes} minutes, {seconds} seconds"

def efficiency_metrics_wrapper(function):
    def wrapper(*args, **kwargs):
        # Reset GPU memory stats BEFORE the function runs for accurate isolated measurement
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()  # Clear cache for clean baseline

        start_time = time.time()  # Record the start time
        result = function(*args, **kwargs)
        elapsed_time = time.time() - start_time  # Calculate the elapsed time

        # Now measure peak memory for THIS function only
        gpu_used = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0  # in MB

        return result, elapsed_time, gpu_used

    return wrapper

def get_subset_data(path, subset_size):
    df = pd.read_csv(path)
    df = df.sample(frac=1, random_state=args['seed']).reset_index(drop=True)

    # Calculate the number of samples to take (10% of the total)
    num_samples = int(len(df) * (subset_size/100))

    # Take a 10% sample of the data
    sampled_df = df.iloc[:num_samples]
    return sampled_df


def setup_dataloaders(args, tokenizer=None):
    subset_percentage = int(args['subset_percentage'])


    if(not 'extract_model' in args):
        train_dataset_path = args['train_set_path']
        sampled_train_df = get_subset_data(train_dataset_path, subset_percentage)
        train_dataset = args['dataset_class'](sampled_train_df, tokenizer, lowercase=args["lower"], no_special_token=args['no_special_token'])
        train_loader = args['dataloader_class'](dataset=train_dataset, max_seq_len=args['max_seq_len'], batch_size=args['train_batch_size'], shuffle=False)
        # if(int(subset_percentage) < 100):
        #     subset_size = int(len(train_dataset) * int(subset_percentage) / 100)
        #     new_train_dataset, b = random_split(train_dataset, [subset_size, len(full_train_loader.dataset) - subset_size])
        #     train_loader = args['dataloader_class'](dataset=new_train_dataset, max_seq_len=args['max_seq_len'], batch_size=args['train_batch_size'], shuffle=False) 
        # else:
        #     train_loader = args['dataloader_class'](dataset=train_dataset, max_seq_len=args['max_seq_len'], batch_size=args['train_batch_size'], shuffle=False)
          

        valid_dataset_path = args['valid_set_path']
        sampled_valid_df = get_subset_data(valid_dataset_path, subset_percentage)
        valid_dataset = args['dataset_class'](sampled_valid_df, tokenizer, lowercase=args["lower"], no_special_token=args['no_special_token'])
        valid_loader = args['dataloader_class'](dataset=valid_dataset, max_seq_len=args['max_seq_len'], batch_size=args['valid_batch_size'], shuffle=False)

        test_dataset_path = args['test_set_path']
        sampled_test_df = get_subset_data(test_dataset_path, subset_percentage)
        test_dataset = args['dataset_class'](sampled_test_df, tokenizer, lowercase=args["lower"], no_special_token=args['no_special_token'])
        test_loader = args['dataloader_class'](dataset=test_dataset, max_seq_len=args['max_seq_len'], batch_size=args['valid_batch_size'], shuffle=False)
    else:
        extract_model, extract_tokenizer, a, b_ = load_extraction_model(args)
        train_dataset_path = args['train_set_path']
        sampled_train_df = get_subset_data(train_dataset_path, subset_percentage)
        train_dataset = args['dataset_class'](args['device'], sampled_train_df, extract_tokenizer, extract_model, args['max_seq_len'],  lowercase=args["lower"], no_special_token=args['no_special_token'])
        train_loader = args['dataloader_class'](dataset=train_dataset, batch_size=args['train_batch_size'], shuffle=False)
        # if(int(subset_percentage) < 100):
        #     subset_size = int(len(train_dataset) * int(subset_percentage) / 100)
        #     new_train_dataset, b = random_split(train_dataset, [subset_size, len(full_train_loader.dataset) - subset_size])
        #     train_loader = args['dataloader_class'](dataset=new_train_dataset, batch_size=args['train_batch_size'], shuffle=False)  
        # else:
        #     train_loader = args['dataloader_class'](dataset=train_dataset, batch_size=args['train_batch_size'], shuffle=False)  


        valid_dataset_path = args['valid_set_path']
        sampled_valid_df = get_subset_data(valid_dataset_path, subset_percentage)
        valid_dataset = args['dataset_class'](args['device'], sampled_valid_df, extract_tokenizer, extract_model, args['max_seq_len'], lowercase=args["lower"], no_special_token=args['no_special_token'])
        valid_loader = args['dataloader_class'](dataset=valid_dataset, batch_size=args['valid_batch_size'], shuffle=False)

        test_dataset_path = args['test_set_path']
        sampled_test_df = get_subset_data(test_dataset_path, subset_percentage)
        test_dataset = args['dataset_class'](args['device'], sampled_test_df, extract_tokenizer, extract_model, args['max_seq_len'], lowercase=args["lower"], no_special_token=args['no_special_token'])
        test_loader = args['dataloader_class'](dataset=test_dataset, batch_size=args['valid_batch_size'], shuffle=False)
    
    return train_loader, valid_loader, test_loader

###
# Training & Evaluation Function
###

# Evaluate function for validation and test
def evaluate(model, data_loader, forward_fn, metrics_fn, i2w, is_test=False):
    model.eval()
    total_loss, total_correct, total_labels = 0, 0, 0

    list_hyp, list_label, list_seq = [], [], []

    pbar = tqdm(iter(data_loader), leave=True, total=len(data_loader))
    for i, batch_data in enumerate(pbar):
        batch_seq = batch_data[-1]        
        loss, batch_hyp, batch_label = forward_fn(model, batch_data, i2w=i2w, device=args['device'])

        
        # Calculate total loss
        test_loss = loss.item()
        total_loss = total_loss + test_loss

        # Calculate evaluation metrics
        list_hyp += batch_hyp
        list_label += batch_label
        list_seq += batch_seq
        metrics = metrics_fn(list_hyp, list_label)

        if not is_test:
            pbar.set_description("VALID LOSS:{:.4f} {}".format(total_loss/(i+1), metrics_to_string(metrics)))
        else:
            pbar.set_description("TEST LOSS:{:.4f} {}".format(total_loss/(i+1), metrics_to_string(metrics)))
    
    if is_test:
        return total_loss, metrics, list_hyp, list_label, list_seq
    else:
        return total_loss, metrics

# Training function and trainer
def train(model, train_loader, valid_loader, optimizer, forward_fn, metrics_fn, valid_criterion, i2w, n_epochs, evaluate_every=1, early_stop=3, step_size=1, gamma=0.5, model_dir="", exp_id=None):
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_val_metric = -100
    count_stop = 0

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0
        list_hyp, list_label = [], []
        
        # print(len(train_loader))
        # exit()

        train_pbar = tqdm(iter(train_loader), leave=True, total=len(train_loader))
        for i, batch_data in enumerate(train_pbar):
            # print(batch_data[:-1])
            # exit()

            loss, batch_hyp, batch_label = forward_fn(model, batch_data, i2w=i2w, device=args['device'])

            optimizer.zero_grad()
            if args['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args['max_norm'])
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_norm'])
                optimizer.step()

            tr_loss = loss.item()
            total_train_loss = total_train_loss + tr_loss

            # Calculate metrics
            list_hyp += batch_hyp
            list_label += batch_label
            
            train_pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} LR:{:.8f}".format((epoch+1),
                total_train_loss/(i+1), get_lr(args, optimizer)))
                        
        metrics = metrics_fn(list_hyp, list_label)
        print("(Epoch {}) TRAIN LOSS:{:.4f} {} LR:{:.8f}".format((epoch+1),
            total_train_loss/(i+1), metrics_to_string(metrics), get_lr(args, optimizer)))
        
        # Decay Learning Rate
        scheduler.step()

        # evaluate
        evaluate(model, valid_loader, forward_fn, metrics_fn, i2w, is_test=False)

if __name__ == "__main__":
    # Make sure CUDA is deterministic
    torch.backends.cudnn.deterministic = True
    
    # Parse args
    args = get_parser()
    args = append_dataset_args(args)

    # Create directory
    model_dir = '{}/{}/{}'.format(args["model_dir"], args["dataset"], args['experiment_name'])
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    elif args['force']:
        print(f'Overwriting model directory `{model_dir}`')
    else:
        raise Exception(f'Model directory `{model_dir}` already exists. Use --force if you want to overwrite the folder.')

    # Set random seed
    set_seed(args['seed'])  # Added here for reproducibility

    # Initial GPU memory reset for clean start
    if args['device'] == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("GPU memory tracking initialized\n")

    w2i, i2w = args['dataset_class'].LABEL2INDEX, args['dataset_class'].INDEX2LABEL
    metrics_scores = []
    result_dfs = []
    efficiency_metrics = []

    # Load model
    model, tokenizer, vocab_path, config_path = load_model(args)

    # for percent in range(10, 101, 10):  # Iterate over percentages from 10% to 100%
    percent = args['subset_percentage']
    print(f"Training on {percent}% of the dataset")

    setup_dataloaders = efficiency_metrics_wrapper(setup_dataloaders)
    result, elapsed_time, gpu_used = setup_dataloaders(args, tokenizer=tokenizer)
    train_loader, valid_loader, test_loader = result

    if args['device'] == "cuda":
        model = model.cuda()

    optimizer = optim.Adam(params=model.parameters(), lr=args['lr'], eps=args['eps'])

    print("=========== TRAINING PHASE ===========")

    efficiency_metrics.append({'percentage': percent, 'ket':'Dataloader', 'elapsed_time':elapsed_time, 'gpu_used': gpu_used})

    train = efficiency_metrics_wrapper(train)
    result, elapsed_time, gpu_used = train(model, train_loader=train_loader, valid_loader=valid_loader, optimizer=optimizer, forward_fn=args['forward_fn'], metrics_fn=args['metrics_fn'], valid_criterion=args['valid_criterion'], i2w=i2w, n_epochs=args['n_epochs'], evaluate_every=1, early_stop=args['early_stop'], step_size=args['step_size'], gamma=args['gamma'], model_dir=model_dir, exp_id=0)

    efficiency_metrics.append({'percentage': percent, 'ket':'Train', 'elapsed_time':elapsed_time, 'gpu_used': gpu_used})

    # Evaluate
    print("=========== EVALUATION PHASE ===========")
    evaluate = efficiency_metrics_wrapper(evaluate)
    result, elapsed_time, gpu_used = evaluate(model, data_loader=test_loader, forward_fn=args['forward_fn'], metrics_fn=args['metrics_fn'], i2w=i2w, is_test=True)
    test_loss, test_metrics, test_hyp, test_label, test_seq = result

    efficiency_metrics.append({'percentage': percent, 'ket':'Test', 'elapsed_time':elapsed_time, 'gpu_used': gpu_used})

    metrics_scores.append({'percentage': percent, 'metrics': test_metrics})
    result_dfs.append(pd.DataFrame({
        'percentage': percent,
        'seq': test_seq, 
        'hyp': test_hyp, 
        'label': test_label
    }))
    

    result_df = pd.concat(result_dfs)
    metric_df = pd.DataFrame(metrics_scores)
    
    print('== Model Efficiency Summary ==')
    summary_efficiency = pd.DataFrame(efficiency_metrics)
    sum = summary_efficiency.groupby('percentage')['elapsed_time'].sum()

    print(sum)
    summary_efficiency = summary_efficiency.groupby('percentage').apply(lambda x: x.append({'ket':'Total', 'elapsed_time': sum[x.name], 'gpu_used': x['gpu_used'].iloc[-1]}, ignore_index=True))
    print(summary_efficiency)

    # Save to model directory
    summary_efficiency.to_csv(f"{model_dir}/summary_efficiency_{percent}.csv")

    # Also save to centralized efficiency_analysis directory for easy aggregation
    centralized_dir = "efficiency_analysis/raw_results"
    os.makedirs(centralized_dir, exist_ok=True)
    centralized_filename = f"{args['dataset']}_{args['experiment_name']}_percent{percent}.csv"
    summary_efficiency.to_csv(f"{centralized_dir}/{centralized_filename}")
    print(f"Results saved to: {model_dir}/summary_efficiency_{percent}.csv")
    print(f"Also saved to: {centralized_dir}/{centralized_filename}")

    # result_df.to_csv(model_dir + "/prediction_result.csv")
    # metric_df.describe().to_csv(model_dir + "/evaluation_result.csv")

# #load bert model & tokenizer
# bert_base_model = BertModel.from_pretrained('bert-base-uncased').to(device)
# tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, truncating = True, padding_side='left')

# #load roberta model & tokenizer
# roberta_model = RobertaModel.from_pretrained('roberta-base', output_hidden_states=True).to(device)
# tokenizer_roberta = RobertaTokenizer.from_pretrained('roberta-base')

