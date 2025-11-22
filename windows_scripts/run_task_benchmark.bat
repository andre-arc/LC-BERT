@echo off
setlocal enabledelayedexpansion

REM Load Telegram configuration if exists
if exist "%~dp0telegram_config.bat" (
    call "%~dp0telegram_config.bat"
    echo Telegram notifications enabled
) else (
    echo Telegram configuration not found. Notifications disabled.
)

set CUDA_VISIBLE_DEVICES=%1

::adam optimizer
@REM call "%~dp0notify_command.bat" "BERT-Benchmark-Adam" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name bert-base-uncased --step_size 1 --gamma 0.9 --experiment_name bert_benchmark_b%3_step1_gamma0.9_lr1e-4_early%2_layer2_lowerTrue --lr 1e-4 --early_stop %2 --dataset ag-news-normal --lower --num_layers 2 --force
@REM call "%~dp0notify_command.bat" "RoBERTa-Benchmark-Adam" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name roberta-base --step_size 1 --gamma 0.9 --experiment_name roberta_benchmark_b%3_step1_gamma0.9_lr1e-4_early%2_layer2_lowerTrue --lr 1e-4 --early_stop %2 --dataset ag-news-normal --lower --num_layers 2 --force
@REM call "%~dp0notify_command.bat" "DistilBERT-Benchmark-Adam" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name distilbert-base-uncased --step_size 1 --gamma 0.9 --experiment_name distilbert_benchmark_b%3_step1_gamma0.9_lr1e-4_early%2_layer2_lowerTrue --lr 1e-4 --early_stop %2 --dataset ag-news-normal --lower --num_layers 2 --force

::adamW Optimizer
call "%~dp0notify_command.bat" "BERT-Benchmark-AdamW" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name bert-base-uncased --step_size 1 --gamma 0.9 --experiment_name bert_benchmark_b%3_step1_gamma0.9_adamW_lr1e-4_early%2_layer2_lowerTrue --lr 1e-4 --early_stop %2 --dataset ag-news-normal --lower --num_layers 2 --force
call "%~dp0notify_command.bat" "RoBERTa-Benchmark-AdamW" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name roberta-base --step_size 1 --gamma 0.9 --experiment_name roberta_benchmark_b%3_step1_gamma0.9_adamW_lr1e-4_early%2_layer2_lowerTrue --lr 1e-4 --early_stop %2 --dataset ag-news-normal --lower --num_layers 2 --force
call "%~dp0notify_command.bat" "DistilBERT-Benchmark-AdamW" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name distilbert-base-uncased --step_size 1 --gamma 0.9 --experiment_name distilbert_benchmark_b%3_step1_gamma0.9_adamW_lr1e-4_early%2_layer2_lowerTrue --lr 1e-4 --early_stop %2 --dataset ag-news-normal --lower --num_layers 2 --force