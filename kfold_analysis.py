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
from sklearn.model_selection import KFold

from utils.functions import load_model, load_extraction_model, load_tokenizer
from utils.args_helper import get_parser, append_dataset_args
from torch.utils.data import DataLoader
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

        start_time = time.time()  # Record the start time
        result = function(*args, **kwargs)
        elapsed_time = time.time() - start_time  # Calculate the elapsed time
        gpu_used = torch.cuda.max_memory_allocated() / (1024 * 1024) # in MB

        return result, elapsed_time, gpu_used
    
    return wrapper
def efficiency_kfold_metrics_wrapper(function):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time before the generator starts
        generator = function(*args, **kwargs)  # Get the generator object
        
        for result in generator:
            elapsed_time = time.time() - start_time  # Calculate elapsed time for this step
            gpu_used = torch.cuda.max_memory_allocated() / (1024 * 1024)  # GPU memory used in MB
            start_time = time.time()  # Reset start time for the next step
            
            yield result, elapsed_time, gpu_used  # Yield result along with timing and GPU usage
    return wrapper


def get_subset_data(path, subset_size):
    df = pd.read_csv(path)

    if(subset_size < 100):
        df = df.sample(frac=1, random_state=args['seed']).reset_index(drop=True)
        # Calculate the number of samples to take (10% of the total)
        num_samples = int(len(df) * (subset_size/100))

        # Take a 10% sample of the data
        sampled_df = df.iloc[:num_samples]
        return sampled_df
    else:
        return df
        


def setup_dataloaders(args, tokenizer=None):
    subset_percentage = int(args['subset_percentage'])
    
    if(not 'extract_model' in args):
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

def setup_kfold_dataloaders(args, tokenizer=None, k=5):
    subset_percentage = int(args['subset_percentage'])
    
    # Load train dataset
    train_dataset_path = args['train_set_path']
    sampled_train_df = get_subset_data(train_dataset_path, subset_percentage)
    
    # Load validation dataset
    valid_dataset_path = args['valid_set_path']
    sampled_valid_df = get_subset_data(valid_dataset_path, subset_percentage)
    
    # Merge the train and validation datasets
    merged_df = pd.concat([sampled_train_df, sampled_valid_df], ignore_index=True)
    
    if 'extract_model' not in args:
        # Standard mode without feature extraction
        dataset = args['dataset_class'](merged_df, tokenizer, lowercase=args["lower"], no_special_token=args['no_special_token'])
    else:
        # Feature extraction mode
        extract_model, extract_tokenizer, a, b_ = load_extraction_model(args)
        dataset = args['dataset_class'](args['device'], merged_df, extract_tokenizer, extract_model, args['max_seq_len'], dim_technique=args['dim_technique'], lowercase=args["lower"], no_special_token=args['no_special_token'])

    # Prepare the K-Fold cross-validation splits
    kfold = KFold(n_splits=k, shuffle=True, random_state=args['seed'])

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(dataset)):
        print(f"=========== Fold {fold + 1}/{k} ===========")
        
        # Subset the dataset for training and validation based on indices
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        valid_subset = torch.utils.data.Subset(dataset, valid_idx)
        
        # Create DataLoaders for this fold
        train_loader = args['dataloader_class'](dataset=train_subset, batch_size=args['train_batch_size'], shuffle=True)
        valid_loader = args['dataloader_class'](dataset=valid_subset, batch_size=args['valid_batch_size'], shuffle=False)
        
        # Test set remains the same for all folds
        if 'extract_model' not in args:
            test_dataset_path = args['test_set_path']
            sampled_test_df = get_subset_data(test_dataset_path, subset_percentage)
            test_dataset = args['dataset_class'](sampled_test_df, tokenizer, lowercase=args["lower"], no_special_token=args['no_special_token'])
        else:
            test_dataset_path = args['test_set_path']
            sampled_test_df = get_subset_data(test_dataset_path, subset_percentage)
            test_dataset = args['dataset_class'](args['device'], sampled_test_df, extract_tokenizer, extract_model, args['max_seq_len'], dim_technique=args['dim_technique'], lowercase=args["lower"], no_special_token=args['no_special_token'])
        
        test_loader = args['dataloader_class'](dataset=test_dataset, batch_size=args['valid_batch_size'], shuffle=False)

        # Yield loaders for this fold
        yield fold + 1, train_loader, valid_loader, test_loader

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

    train_accuracies, val_accuracies = [], []
    train_losses, val_losses = [], []

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0
        list_hyp, list_label = [], []

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
        train_accuracies.append(metrics["ACC"])
        train_losses.append(total_train_loss / len(train_loader))

        print("(Epoch {}) TRAIN LOSS:{:.4f} {} LR:{:.8f}".format((epoch+1),
            total_train_loss/(i+1), metrics_to_string(metrics), get_lr(args, optimizer)))
        
        # Decay Learning Rate
        scheduler.step()

        # evaluate
        if ((epoch+1) % evaluate_every) == 0:
            val_loss, val_metrics = evaluate(model, valid_loader, forward_fn, metrics_fn, i2w, is_test=False)

            val_accuracies.append(val_metrics["ACC"])
            val_losses.append(val_loss)

            # Early stopping
            val_metric = val_metrics[valid_criterion]
            if best_val_metric < val_metric:
                best_val_metric = val_metric
                # save model
                if exp_id is not None:
                    torch.save(model.state_dict(), model_dir + "/best_model_" + str(exp_id) + ".th")
                else:
                    torch.save(model.state_dict(), model_dir + "/best_model.th")
                count_stop = 0
            else:
                count_stop += 1
                print("count stop:", count_stop)
                if count_stop == early_stop:
                    break

    return train_accuracies, val_accuracies, train_losses, val_losses

if __name__ == "__main__":
    # Make sure cuda is deterministic
    torch.backends.cudnn.deterministic = True
    
    # Parse args
    args = get_parser()
    args = append_dataset_args(args)

    # create directory
    model_dir = '{}/{}/{}'.format(args["model_dir"],args["dataset"],args['experiment_name'])
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    elif args['force']:
        print(f'overwriting model directory `{model_dir}`')
    else:
        raise Exception(f'model directory `{model_dir}` already exists, use --force if you want to overwrite the folder')

    # Set random seed
    set_seed(args['seed'])  # Added here for reproductibility    
    
    w2i, i2w = args['dataset_class'].LABEL2INDEX, args['dataset_class'].INDEX2LABEL
    metrics_scores = []
    result_dfs = []
    # efficiency_metrics = []

    # load tokenizer
    tokenizer = load_tokenizer(args)
    setup_kfold_dataloaders = efficiency_kfold_metrics_wrapper(setup_kfold_dataloaders)

    # Initialize lists to store efficiency metrics
    all_efficiency_metrics = []
    # Initialize lists to store metrics from each fold
    all_train_accuracies = []
    all_val_accuracies = []
    all_train_losses = []
    all_val_losses = []
    all_f1_scores = []
    all_elapsed_times = []
    all_gpu_usages = []


        # print(tes1)
        # exit()
    for (fold, train_loader, valid_loader, test_loader), elapsed_time, gpu_used in setup_kfold_dataloaders(args, tokenizer=tokenizer, k=5):
        # fold, train_loader, valid_loader, test_loader = (result)
        print(f"\nFold {fold} - Elapsed time: {elapsed_time} seconds, GPU used: {gpu_used} MB")

        print(f"=========== Fold {fold + 1}/{5} ===========")

        # Capture efficiency metrics for data loading
        all_efficiency_metrics.append({
            'fold': fold + 1,
            'dataloader_elapsed_time': elapsed_time,
            'dataloader_gpu_used': gpu_used
        })


        print("\n=========== TRAINING PHASE ===========")

        # load model
        model, tokenizer, vocab_path, config_path = load_model(args)
        if args['device'] == "cuda":
            model = model.cuda()

        optimizer = optim.AdamW(params=model.parameters(), lr=args['lr'], eps=args['eps'])
        # optimizer = optim.RAdam(params=model.parameters(), lr=args['lr'])


        train = efficiency_metrics_wrapper(train)
        result, elapsed_time, gpu_used = train(model, train_loader=train_loader, valid_loader=valid_loader, optimizer=optimizer, forward_fn=args['forward_fn'], metrics_fn=args['metrics_fn'], valid_criterion=args['valid_criterion'], i2w=i2w, n_epochs=args['n_epochs'], evaluate_every=1, early_stop=args['early_stop'], step_size=args['step_size'], gamma=args['gamma'], model_dir=model_dir, exp_id=0)
        train_losses, val_losses, train_accuracies, val_accuracies= result

        # # Log fold metrics
        # print(f"Fold {fold + 1} - Dataloader elapsed time: {elapsed_time_dataloader:.2f} seconds, GPU used: {gpu_used_dataloader:.2f} MB")
        # print(f"Fold {fold + 1} - Training elapsed time: {elapsed_time_train:.2f} seconds, GPU used: {gpu_used_train:.2f} MB")


        # Save Meta
        if vocab_path:
            shutil.copyfile(vocab_path, f'{model_dir}/vocab.txt')
        if config_path:
            shutil.copyfile(config_path, f'{model_dir}/config.json')
            
        # Load best model
        model.load_state_dict(torch.load(model_dir + "/best_model_0.th"))

        # Evaluate
        print("=========== EVALUATION PHASE ===========")
        evaluate = efficiency_metrics_wrapper(evaluate)
        test_result, test_elapsed_time, test_gpu_used = evaluate(model, data_loader=test_loader, forward_fn=args['forward_fn'], metrics_fn=args['metrics_fn'], i2w=i2w, is_test=True)
        test_loss, test_metrics, test_hyp, test_label, test_seq = result

        # efficiency_metrics.append({'ket':'Test', 'elapsed_time':elapsed_time, 'gpu_used': gpu_used})

        metrics_scores.append(test_metrics)
        result_dfs.append(pd.DataFrame({
            'seq':test_seq, 
            'hyp': test_hyp, 
            'label': test_label
        }))
        
        result_df = pd.concat(result_dfs)
        metric_df = pd.DataFrame.from_records(metrics_scores)

        # Store metrics from each fold
        all_train_accuracies.append(train_accuracies[-1])  # Append last epoch's accuracy
        all_val_accuracies.append(val_accuracies[-1])      # Append last epoch's accuracy
        all_train_losses.append(train_losses[-1])          # Append last epoch's loss
        all_val_losses.append(val_losses[-1])              # Append last epoch's loss
        all_f1_scores.append(test_metrics['F1'])           # Append from testing F1-score
        all_elapsed_times.append(elapsed_time_train)       # Append training elapsed time
        all_gpu_usages.append(gpu_used_train)              # Append GPU usage
        
        # print('== Prediction Result ==')
        # print(result_df.head())
        # print()
        
        # print('== Model Performance ==')
        # print(metric_df.describe())

        # print('== Model Efficiency Summary ==')
        # summary_efficiency = pd.DataFrame(efficiency_metrics)
        # sum = elapsed_timestamp_to_detail(summary_efficiency['elapsed_time'].sum())
        # summary_efficiency = summary_efficiency.append({'ket':'Total', 'elapsed_time':sum, 'gpu_used': summary_efficiency['gpu_used'].iloc[-1]}, ignore_index=True)
        # print(summary_efficiency)
        
        # result_df.to_csv(model_dir + "/prediction_result.csv")
        # metric_df.describe().to_csv(model_dir + "/evaluation_result.csv")
        # summary_efficiency.to_csv(model_dir + "/summary_efficiency.csv")

    # Combine metrics from all folds
    combined_metrics = {
        'avg_train_accuracy': sum(all_train_accuracies) / len(all_train_accuracies),
        'avg_val_accuracy': sum(all_val_accuracies) / len(all_val_accuracies),
        'avg_train_loss': sum(all_train_losses) / len(all_train_losses),
        'avg_val_loss': sum(all_val_losses) / len(all_val_losses),
        'f1_score': all_f1_scores[-1],  # Take F1-score from the last fold
        'max_gpu_usage': max(all_gpu_usages),
        'avg_elapsed_time': sum(all_elapsed_times) / len(all_elapsed_times)
    }

    print(combined_metrics)
# #load bert model & tokenizer
# bert_base_model = BertModel.from_pretrained('bert-base-uncased').to(device)
# tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, truncating = True, padding_side='left')

# #load roberta model & tokenizer
# roberta_model = RobertaModel.from_pretrained('roberta-base', output_hidden_states=True).to(device)
# tokenizer_roberta = RobertaTokenizer.from_pretrained('roberta-base')



