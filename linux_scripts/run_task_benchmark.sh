#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root directory (parent of linux_scripts)
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Load Telegram configuration if exists
if [ -f "$SCRIPT_DIR/telegram_config.sh" ]; then
    source "$SCRIPT_DIR/telegram_config.sh"
    echo "Telegram notifications enabled"
else
    echo "Telegram configuration not found. Notifications disabled."
fi

export CUDA_VISIBLE_DEVICES=$1

# Adam optimizer
# "$SCRIPT_DIR/notify_command.sh" "BERT-Benchmark-Adam" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name bert-base-uncased --step_size 1 --gamma 0.9 --experiment_name bert_benchmark_b${3}_step1_gamma0.9_lr1e-4_early${2}_layer2_lowerTrue --lr 1e-4 --early_stop $2 --dataset ag-news-normal --lower --num_layers 2 --force
# "$SCRIPT_DIR/notify_command.sh" "RoBERTa-Benchmark-Adam" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name roberta-base --step_size 1 --gamma 0.9 --experiment_name roberta_benchmark_b${3}_step1_gamma0.9_lr1e-4_early${2}_layer2_lowerTrue --lr 1e-4 --early_stop $2 --dataset ag-news-normal --lower --num_layers 2 --force
# "$SCRIPT_DIR/notify_command.sh" "DistilBERT-Benchmark-Adam" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name distilbert-base-uncased --step_size 1 --gamma 0.9 --experiment_name distilbert_benchmark_b${3}_step1_gamma0.9_lr1e-4_early${2}_layer2_lowerTrue --lr 1e-4 --early_stop $2 --dataset ag-news-normal --lower --num_layers 2 --force

# AdamW Optimizer
"$SCRIPT_DIR/notify_command.sh" "BERT-Benchmark-AdamW" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name bert-base-uncased --step_size 1 --gamma 0.9 --experiment_name bert_benchmark_b${3}_step1_gamma0.9_adamW_lr1e-4_early${2}_layer2_lowerTrue --lr 1e-4 --early_stop $2 --dataset ag-news-normal --lower --num_layers 2 --force
"$SCRIPT_DIR/notify_command.sh" "RoBERTa-Benchmark-AdamW" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name roberta-base --step_size 1 --gamma 0.9 --experiment_name roberta_benchmark_b${3}_step1_gamma0.9_adamW_lr1e-4_early${2}_layer2_lowerTrue --lr 1e-4 --early_stop $2 --dataset ag-news-normal --lower --num_layers 2 --force
"$SCRIPT_DIR/notify_command.sh" "DistilBERT-Benchmark-AdamW" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name distilbert-base-uncased --step_size 1 --gamma 0.9 --experiment_name distilbert_benchmark_b${3}_step1_gamma0.9_adamW_lr1e-4_early${2}_layer2_lowerTrue --lr 1e-4 --early_stop $2 --dataset ag-news-normal --lower --num_layers 2 --force
