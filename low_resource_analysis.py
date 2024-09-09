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

# Set GPU memory limit to 50%
torch.cuda.set_per_process_memory_fraction(0.15)

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

def get_subset_data(path, subset_size):
    df = pd.read_csv(path)
    df = df.sample(frac=1, random_state=args['seed']).reset_index(drop=True)

    # Calculate the number of samples to take (10% of the total)
    num_samples = int(len(df) * (subset_size / 100))

    # Take a 10% sample of the data
    sampled_df = df.iloc[:num_samples]
    return sampled_df

def setup_dataloaders(args, tokenizer=None):
    subset_percentage = int(args['subset_percentage'])

    if not 'extract_model' in args:
        train_dataset_path = args['train_set_path']
        sampled_train_df = get_subset_data(train_dataset_path, subset_percentage)
        train_dataset = args['dataset_class'](sampled_train_df, tokenizer, lowercase=args["lower"], no_special_token=args['no_special_token'])
        train_loader = args['dataloader_class'](dataset=train_dataset, max_seq_len=args['max_seq_len'], batch_size=args['train_batch_size'], shuffle=False)

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
        train_dataset = args['dataset_class'](args['device'], sampled_train_df, extract_tokenizer, extract_model, args['max_seq_len'], dim_technique=args['dim_technique'], lowercase=args["lower"], no_special_token=args['no_special_token'])
        train_loader = args['dataloader_class'](dataset=train_dataset, batch_size=args['train_batch_size'], shuffle=False)

        valid_dataset_path = args['valid_set_path']
        sampled_valid_df = get_subset_data(valid_dataset_path, subset_percentage)
        valid_dataset = args['dataset_class'](args['device'], sampled_valid_df, extract_tokenizer, extract_model, args['max_seq_len'], dim_technique=args['dim_technique'], lowercase=args["lower"], no_special_token=args['no_special_token'])
        valid_loader = args['dataloader_class'](dataset=valid_dataset, batch_size=args['valid_batch_size'], shuffle=False)

        test_dataset_path = args['test_set_path']
        sampled_test_df = get_subset_data(test_dataset_path, subset_percentage)
        test_dataset = args['dataset_class'](args['device'], sampled_test_df, extract_tokenizer, extract_model, args['max_seq_len'], dim_technique=args['dim_technique'], lowercase=args["lower"], no_special_token=args['no_special_token'])
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
        return (total_loss, metrics, list_hyp, list_label, list_seq), "success"
    else:
        return (total_loss, metrics), "success"

# Training function and trainer
def train(model, train_loader, valid_loader, optimizer, forward_fn, metrics_fn, valid_criterion, i2w, n_epochs, evaluate_every=1, early_stop=3, step_size=1, gamma=0.5, model_dir="", exp_id=None):
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_val_metric = -100
    count_stop = 0

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0
        list_hyp, list_label = [], []

        train_pbar = tqdm(iter(train_loader), leave=True, total=len(train_loader))
        for i, batch_data in enumerate(train_pbar):

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

    return True, "success"

###
# Main execution
###
if __name__ == "__main__":
    args = get_parser()
    args = append_dataset_args(args)

    # Check if the status file exists
    status_file = 'training_evaluation_status.csv'
    if os.path.exists(status_file):
        status_df = pd.read_csv(status_file)
    else:
        status_df = pd.DataFrame(columns=['model_name', 'status', 'feature_extraction', 'dim_reduction'])


    # Set random seed
    set_seed(args['seed'])  # Added here for reproductibility    
    
    w2i, i2w = args['dataset_class'].LABEL2INDEX, args['dataset_class'].INDEX2LABEL
    metrics_scores = []


    # Load and prepare the model and data
    model, tokenizer, vocab_path, config_path = load_model(args)

    try:
        if args['device'] == "cuda":
            model = model.cuda()

        train_loader, valid_loader, test_loader = setup_dataloaders(args, tokenizer)

        # Training phase
        optimizer = AdamW(model.parameters(), lr=args['lr'])
        status, train_status =  train(model, 
                                        train_loader=train_loader, 
                                        valid_loader=valid_loader, 
                                        optimizer=optimizer, 
                                        forward_fn=args['forward_fn'], 
                                        metrics_fn=args['metrics_fn'], 
                                        valid_criterion=args['valid_criterion'], 
                                        i2w=i2w, n_epochs=args['n_epochs'], 
                                        evaluate_every=1, 
                                        early_stop=args['early_stop'], 
                                        step_size=args['step_size'], 
                                        gamma=args['gamma'], 
                                        model_dir="",
                                        exp_id=0)

        # Evaluation phase
        (test_loss, test_metrics, test_hyp, test_label, test_seq), test_status = evaluate(
            model, test_loader, forward_fn=args['forward_fn'], metrics_fn=args['metrics_fn'], i2w=i2w, is_test=True
        )
    except RuntimeError as e:
        if 'out of memory' in str(e):
            torch.cuda.empty_cache()
            print("Out of memory error caught during evaluation!")
            test_status = False
    if(not 'extract_model' in args):
        args['extract_model'] = None

    if(not 'dim_technique' in args):
        args['dim_technique'] = None

    status_df = status_df.append({'model_name': args['model_name'], 'feature_extraction': args['extract_model'], 'dim_reduction': args['dim_technique'], 'status': test_status}, ignore_index=True)

    # print(status_df)
    # Save the status DataFrame to CSV
    # status_df.to_csv(status_file, index=False, header=['model_name', 'status', 'feature_extraction', 'dim_reduction'] )
