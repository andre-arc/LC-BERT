@echo off
setlocal enabledelayedexpansion

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"

REM Load Telegram configuration if exists
if exist "%SCRIPT_DIR%telegram_config.bat" (
    call "%SCRIPT_DIR%telegram_config.bat"
    echo Telegram notifications enabled
) else (
    echo Telegram configuration not found. Notifications disabled.
)

REM Change to the project root directory (parent of script directory)
cd /d "%SCRIPT_DIR%.."

REM Set CUDA device
set CUDA_VISIBLE_DEVICES=%1

REM Define the Python script location (now in project root)
set "PYTHON_SCRIPT=train_multiple_seeds.py"

::adam optimizer
@REM call "%SCRIPT_DIR%notify_command.bat" "BERT-Benchmark-Adam" python %PYTHON_SCRIPT% --n_epochs 5 --train_batch_size %3 --model_name bert-base-uncased --step_size 1 --gamma 0.9 --experiment_name bert_benchmark_b%3_step1_gamma0.9_lr1e-4_early%2_layer2_lowerTrue --lr 1e-4 --early_stop %2 --dataset ag-news-normal --lower --num_layers 2 --force
@REM call "%SCRIPT_DIR%notify_command.bat" "RoBERTa-Benchmark-Adam" python %PYTHON_SCRIPT% --n_epochs 5 --train_batch_size %3 --model_name roberta-base --step_size 1 --gamma 0.9 --experiment_name roberta_benchmark_b%3_step1_gamma0.9_lr1e-4_early%2_layer2_lowerTrue --lr 1e-4 --early_stop %2 --dataset ag-news-normal --lower --num_layers 2 --force
@REM call "%SCRIPT_DIR%notify_command.bat" "DistilBERT-Benchmark-Adam" python %PYTHON_SCRIPT% --n_epochs 5 --train_batch_size %3 --model_name distilbert-base-uncased --step_size 1 --gamma 0.9 --experiment_name distilbert_benchmark_b%3_step1_gamma0.9_lr1e-4_early%2_layer2_lowerTrue --lr 1e-4 --early_stop %2 --dataset ag-news-normal --lower --num_layers 2 --force

::adamW Optimizer
call "%SCRIPT_DIR%notify_command.bat" "BERT-Benchmark-AdamW" python %PYTHON_SCRIPT% --n_epochs 5 --train_batch_size %3 --model_name bert-base-uncased --step_size 1 --gamma 0.9 --experiment_name bert_benchmark_b%3_step1_gamma0.9_adamW_lr1e-4_early%2_layer2_lowerTrue --lr 1e-4 --early_stop %2 --dataset ag-news-normal --lower --num_layers 2 --force
call "%SCRIPT_DIR%notify_command.bat" "RoBERTa-Benchmark-AdamW" python %PYTHON_SCRIPT% --n_epochs 5 --train_batch_size %3 --model_name roberta-base --step_size 1 --gamma 0.9 --experiment_name roberta_benchmark_b%3_step1_gamma0.9_adamW_lr1e-4_early%2_layer2_lowerTrue --lr 1e-4 --early_stop %2 --dataset ag-news-normal --lower --num_layers 2 --force
call "%SCRIPT_DIR%notify_command.bat" "DistilBERT-Benchmark-AdamW" python %PYTHON_SCRIPT% --n_epochs 5 --train_batch_size %3 --model_name distilbert-base-uncased --step_size 1 --gamma 0.9 --experiment_name distilbert_benchmark_b%3_step1_gamma0.9_adamW_lr1e-4_early%2_layer2_lowerTrue --lr 1e-4 --early_stop %2 --dataset ag-news-normal --lower --num_layers 2 --force

endlocal