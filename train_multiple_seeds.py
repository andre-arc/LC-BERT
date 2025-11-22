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

from utils.functions import load_model, load_extraction_model, load_tokenizer
from utils.args_helper import get_parser, append_dataset_args
from torch.utils.data import DataLoader
import time

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

###
# modelling functions
###
def get_lr(args, optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def metrics_to_string(metric_dict):
    string_list = []
    for key, value in metric_dict.items():
        if key != 'seed':  # Skip seed in display
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

def get_subset_data(path, subset_size, seed):
    print(os.getcwd())
    df = pd.read_csv(path)

    if(subset_size < 100):
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        # Calculate the number of samples to take (10% of the total)
        num_samples = int(len(df) * (subset_size/100))

        # Take a sample of the data
        sampled_df = df.iloc[:num_samples]
        return sampled_df
    else:
        return df
        


def setup_dataloaders(args, tokenizer=None, seed=None):
    subset_percentage = int(args['subset_percentage'])
    current_seed = seed if seed is not None else args['seed']
    
    if(not 'extract_model' in args):
        train_dataset_path = args['train_set_path']
        sampled_train_df = get_subset_data(train_dataset_path, subset_percentage, current_seed)
        train_dataset = args['dataset_class'](sampled_train_df, tokenizer, lowercase=args["lower"], no_special_token=args['no_special_token'])
        train_loader = args['dataloader_class'](dataset=train_dataset, max_seq_len=args['max_seq_len'], batch_size=args['train_batch_size'], shuffle=False) 

        valid_dataset_path = args['valid_set_path']
        sampled_valid_df = get_subset_data(valid_dataset_path, subset_percentage, current_seed)
        valid_dataset = args['dataset_class'](sampled_valid_df, tokenizer, lowercase=args["lower"], no_special_token=args['no_special_token'])
        valid_loader = args['dataloader_class'](dataset=valid_dataset, max_seq_len=args['max_seq_len'], batch_size=args['valid_batch_size'], shuffle=False)

        test_dataset_path = args['test_set_path']
        sampled_test_df = get_subset_data(test_dataset_path, subset_percentage, current_seed)
        test_dataset = args['dataset_class'](sampled_test_df, tokenizer, lowercase=args["lower"], no_special_token=args['no_special_token'])
        test_loader = args['dataloader_class'](dataset=test_dataset, max_seq_len=args['max_seq_len'], batch_size=args['valid_batch_size'], shuffle=False)
    else:
        extract_model, extract_tokenizer, a, b_ = load_extraction_model(args)
        train_dataset_path = args['train_set_path']
        sampled_train_df = get_subset_data(train_dataset_path, subset_percentage, current_seed)
        train_dataset = args['dataset_class'](args['device'], sampled_train_df, extract_tokenizer, extract_model, args['max_seq_len'], dim_technique=args['dim_technique'], lowercase=args["lower"], no_special_token=args['no_special_token'])
        train_loader = args['dataloader_class'](dataset=train_dataset, batch_size=args['train_batch_size'], shuffle=False)

        valid_dataset_path = args['valid_set_path']
        sampled_valid_df = get_subset_data(valid_dataset_path, subset_percentage, current_seed)
        valid_dataset = args['dataset_class'](args['device'], sampled_valid_df, extract_tokenizer, extract_model, args['max_seq_len'], dim_technique=args['dim_technique'], lowercase=args["lower"], no_special_token=args['no_special_token'])
        valid_loader = args['dataloader_class'](dataset=valid_dataset, batch_size=args['valid_batch_size'], shuffle=False)

        test_dataset_path = args['test_set_path']
        sampled_test_df = get_subset_data(test_dataset_path, subset_percentage, current_seed)
        test_dataset = args['dataset_class'](args['device'], sampled_test_df, extract_tokenizer, extract_model, args['max_seq_len'], dim_technique=args['dim_technique'], lowercase=args["lower"], no_special_token=args['no_special_token'])
        test_loader = args['dataloader_class'](dataset=test_dataset, batch_size=args['valid_batch_size'], shuffle=False)
    
    return train_loader, valid_loader, test_loader

###
# Training & Evaluation Function
###

# Evaluate function for validation and test
def evaluate(model, data_loader, forward_fn, metrics_fn, i2w, device, is_test=False):
    model.eval()
    total_loss, total_correct, total_labels = 0, 0, 0

    list_hyp, list_label, list_seq = [], [], []

    pbar = tqdm(iter(data_loader), leave=True, total=len(data_loader))
    for i, batch_data in enumerate(pbar):
        batch_seq = batch_data[-1]        
        loss, batch_hyp, batch_label = forward_fn(model, batch_data, i2w=i2w, device=device)

        
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
def train(model, train_loader, valid_loader, optimizer, forward_fn, metrics_fn, valid_criterion, i2w, device, n_epochs, max_norm, fp16=False, evaluate_every=1, early_stop=3, step_size=1, gamma=0.5, model_dir="", exp_id=None):
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_val_metric = -100
    count_stop = 0

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0
        list_hyp, list_label = [], []

        train_pbar = tqdm(iter(train_loader), leave=True, total=len(train_loader))
        for i, batch_data in enumerate(train_pbar):
            loss, batch_hyp, batch_label = forward_fn(model, batch_data, i2w=i2w, device=device)

            optimizer.zero_grad()
            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()

            tr_loss = loss.item()
            total_train_loss = total_train_loss + tr_loss

            # Calculate metrics
            list_hyp += batch_hyp
            list_label += batch_label
            
            train_pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} LR:{:.8f}".format((epoch+1),
                total_train_loss/(i+1), scheduler.get_last_lr()[0]))
                        
        metrics = metrics_fn(list_hyp, list_label)
        print("(Epoch {}) TRAIN LOSS:{:.4f} {} LR:{:.8f}".format((epoch+1),
            total_train_loss/(i+1), metrics_to_string(metrics), scheduler.get_last_lr()[0]))
        
        # Decay Learning Rate
        scheduler.step()

        # evaluate
        if ((epoch+1) % evaluate_every) == 0:
            val_loss, val_metrics = evaluate(model, valid_loader, forward_fn, metrics_fn, i2w, device, is_test=False)

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

    # Define multiple random seeds (at least 3-5 as recommended by reviewers)
    random_seeds = [42, 88, 456]  # 5 different seeds
    num_seeds = len(random_seeds)
    
    print(f"\n{'='*60}")
    print(f"Running experiments with {num_seeds} different random seeds: {random_seeds}")
    print(f"{'='*60}\n")
    
    w2i, i2w = args['dataset_class'].LABEL2INDEX, args['dataset_class'].INDEX2LABEL
    
    # Store results across all seeds
    all_metrics_scores = []
    all_result_dfs = []
    all_efficiency_metrics = []
    
    # Run experiment for each seed
    for seed_idx, seed in enumerate(random_seeds):
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {seed_idx + 1}/{num_seeds} - Random Seed: {seed}")
        print(f"{'='*60}\n")
        
        # Set random seed
        set_seed(seed)
        
        # Reset CUDA memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        metrics_scores = []
        result_dfs = []
        efficiency_metrics = []

        # load tokenizer
        tokenizer = load_tokenizer(args)
        setup_dataloaders_wrapped = efficiency_metrics_wrapper(setup_dataloaders)
        result, elapsed_time, gpu_used = setup_dataloaders_wrapped(args, tokenizer=tokenizer, seed=seed)
        train_loader, valid_loader, test_loader = result

        efficiency_metrics.append({'ket':'Dataloader', 'elapsed_time':elapsed_time, 'gpu_used': gpu_used, 'seed': seed})

        print(f"\n=========== TRAINING PHASE (Seed {seed}) ===========")

        # load model (fresh initialization for each seed)
        model, tokenizer, vocab_path, config_path = load_model(args)
        if args['device'] == "cuda":
            model = model.cuda()

        optimizer = optim.AdamW(params=model.parameters(), lr=args['lr'], eps=args['eps'])

        train_wrapped = efficiency_metrics_wrapper(train)
        result, elapsed_time, gpu_used = train_wrapped(
            model, 
            train_loader=train_loader, 
            valid_loader=valid_loader, 
            optimizer=optimizer, 
            forward_fn=args['forward_fn'], 
            metrics_fn=args['metrics_fn'], 
            valid_criterion=args['valid_criterion'], 
            i2w=i2w, 
            device=args['device'],
            n_epochs=args['n_epochs'], 
            max_norm=args['max_norm'],
            fp16=args.get('fp16', False),
            evaluate_every=1, 
            early_stop=args['early_stop'], 
            step_size=args['step_size'], 
            gamma=args['gamma'], 
            model_dir=model_dir, 
            exp_id=seed_idx
        )

        efficiency_metrics.append({'ket':'Train', 'elapsed_time':elapsed_time, 'gpu_used': gpu_used, 'seed': seed})

        # Save Meta (only once)
        if seed_idx == 0:
            if vocab_path:
                shutil.copyfile(vocab_path, f'{model_dir}/vocab.txt')
            if config_path:
                shutil.copyfile(config_path, f'{model_dir}/config.json')
            
        # Load best model
        model.load_state_dict(torch.load(model_dir + f"/best_model_{seed_idx}.th"))

        # Evaluate
        print(f"=========== EVALUATION PHASE (Seed {seed}) ===========")
        evaluate_wrapped = efficiency_metrics_wrapper(evaluate)
        result, elapsed_time, gpu_used = evaluate_wrapped(
            model, 
            data_loader=test_loader, 
            forward_fn=args['forward_fn'], 
            metrics_fn=args['metrics_fn'], 
            i2w=i2w, 
            device=args['device'],
            is_test=True
        )
        test_loss, test_metrics, test_hyp, test_label, test_seq = result

        efficiency_metrics.append({'ket':'Test', 'elapsed_time':elapsed_time, 'gpu_used': gpu_used, 'seed': seed})

        # Add seed information to metrics
        test_metrics['seed'] = seed
        metrics_scores.append(test_metrics)
        
        result_dfs.append(pd.DataFrame({
            'seq':test_seq, 
            'hyp': test_hyp, 
            'label': test_label,
            'seed': seed
        }))
        
        # Store results for this seed
        all_metrics_scores.append(test_metrics)
        all_result_dfs.extend(result_dfs)
        all_efficiency_metrics.extend(efficiency_metrics)
        
        print(f"\nSeed {seed} Results:")
        for key, value in test_metrics.items():
            if key != 'seed':
                print(f"  {key}: {value:.4f}")
    
    # Calculate statistics across all seeds
    print(f"\n{'='*60}")
    print("FINAL RESULTS - AGGREGATED ACROSS ALL SEEDS")
    print(f"{'='*60}\n")
    
    # Combine all metrics
    all_result_df = pd.concat(all_result_dfs, ignore_index=True)
    all_metric_df = pd.DataFrame.from_records(all_metrics_scores)
    
    # Calculate mean and std for each metric (excluding 'seed' column)
    metric_columns = [col for col in all_metric_df.columns if col != 'seed']
    metrics_summary = pd.DataFrame({
        'metric': metric_columns,
        'mean': [all_metric_df[col].mean() for col in metric_columns],
        'std': [all_metric_df[col].std() for col in metric_columns],
        'min': [all_metric_df[col].min() for col in metric_columns],
        'max': [all_metric_df[col].max() for col in metric_columns]
    })
    
    print('== Model Performance Summary (Mean ± Std) ==')
    print(metrics_summary.to_string(index=False))
    print()
    
    # Print formatted results
    print('== Detailed Metrics (Mean ± Std) ==')
    for _, row in metrics_summary.iterrows():
        print(f"{row['metric']}: {row['mean']:.4f} ± {row['std']:.4f} (min: {row['min']:.4f}, max: {row['max']:.4f})")
    print()
    
    # Efficiency metrics summary
    all_efficiency_df = pd.DataFrame(all_efficiency_metrics)
    
    # Group by 'ket' (phase) and calculate statistics
    efficiency_summary = all_efficiency_df.groupby('ket').agg({
        'elapsed_time': ['mean', 'std', 'min', 'max'],
        'gpu_used': ['mean', 'std', 'min', 'max']
    }).reset_index()
    
    print('== Model Efficiency Summary (Mean ± Std) ==')
    print(efficiency_summary)
    print()
    
    # Print formatted efficiency results
    print('== Detailed Efficiency Metrics ==')
    for phase in ['Dataloader', 'Train', 'Test']:
        phase_data = all_efficiency_df[all_efficiency_df['ket'] == phase]
        if not phase_data.empty:
            time_mean = phase_data['elapsed_time'].mean()
            time_std = phase_data['elapsed_time'].std()
            gpu_mean = phase_data['gpu_used'].mean()
            gpu_std = phase_data['gpu_used'].std()
            print(f"{phase}:")
            print(f"  Time: {elapsed_timestamp_to_detail(time_mean)} ± {time_std:.2f}s")
            print(f"  GPU Memory: {gpu_mean:.2f} ± {gpu_std:.2f} MB")
    print()
    
    # Save all results
    all_result_df.to_csv(model_dir + "/prediction_result_all_seeds.csv", index=False)
    all_metric_df.to_csv(model_dir + "/evaluation_result_all_seeds.csv", index=False)
    metrics_summary.to_csv(model_dir + "/metrics_summary_mean_std.csv", index=False)
    all_efficiency_df.to_csv(model_dir + "/efficiency_all_seeds.csv", index=False)
    efficiency_summary.to_csv(model_dir + "/efficiency_summary.csv", index=False)
    
    print(f"\n{'='*60}")
    print("All results saved to:")
    print(f"  - {model_dir}/prediction_result_all_seeds.csv")
    print(f"  - {model_dir}/evaluation_result_all_seeds.csv")
    print(f"  - {model_dir}/metrics_summary_mean_std.csv")
    print(f"  - {model_dir}/efficiency_all_seeds.csv")
    print(f"  - {model_dir}/efficiency_summary.csv")
    print(f"{'='*60}\n")