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
@REM call "%~dp0notify_command.bat" "BERT-Extraction-BiLSTM" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name bilstm --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-extraction%3_step1_gamma0.9_lr1e-4_early%2_layer2_lowerTrue --lr 3e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-bert-extraction --lower --num_layers 2 --force
@REM call "%~dp0notify_command.bat" "RoBERTa-Extraction-BiLSTM" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name bilstm --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-extraction%3_step1_gamma0.9_lr1e-4_early%2_layer2_lowerTrue --lr 3e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-roberta-extraction --lower --num_layers 2 --force

::SVD
@REM call "%~dp0notify_command.bat" "SVD-BERT-BiLSTM" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-whitening-svd-bilstm --lr 2e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-bert-whitening-svd --lower --num_layers 2 --force
@REM call "%~dp0notify_command.bat" "SVD-BERT-MLP" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name mlp-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-whitening-svd-mlp --lr 2e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-bert-whitening-svd --lower --num_layers 2 --force
@REM call "%~dp0notify_command.bat" "SVD-RoBERTa-BiLSTM" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-whitening-svd-bilstm --lr 2e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-roberta-whitening-svd --lower --num_layers 2 --force
@REM call "%~dp0notify_command.bat" "SVD-RoBERTa-MLP" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name mlp-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-whitening-svd-mlp --lr 2e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-roberta-whitening-svd --lower --num_layers 2 --force

::ZCA
@REM call "%~dp0notify_command.bat" "ZCA-BERT-BiLSTM" python train_multiple_seed.py --n_epochs 1 --train_batch_size %3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-whitening-zca-bilstm --lr 1.8e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-bert-whitening-zca --lower --num_layers 2 --force
@REM call "%~dp0notify_command.bat" "ZCA-BERT-MLP" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name mlp-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-whitening-zca-mlp --lr 1.8e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-bert-whitening-zca --lower --num_layers 2 --force
call "%~dp0notify_command.bat" "ZCA-RoBERTa-BiLSTM" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-whitening-zca-bilstm --lr 2e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-roberta-whitening-zca --lower --num_layers 2 --force
call "%~dp0notify_command.bat" "ZCA-RoBERTa-MLP" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name mlp-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-whitening-zca-mlp --lr 2e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-roberta-whitening-zca --lower --num_layers 2 --force

::ZCA-SVD
@REM call "%~dp0notify_command.bat" "ZCA-SVD-BERT-BiLSTM" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-whitening-zca-svd-bilstm --lr 1.8e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-bert-whitening-zca-svd --lower --num_layers 2 --force
@REM call "%~dp0notify_command.bat" "ZCA-SVD-BERT-MLP" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name mlp-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-whitening-zca-svd-mlp --lr 1.8e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-bert-whitening-zca-svd --lower --num_layers 2 --force
@REM call "%~dp0notify_command.bat" "ZCA-SVD-RoBERTa-BiLSTM" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-whitening-zca-svd-bilstm --lr 2e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-roberta-whitening-zca-svd --lower --num_layers 2 --force
@REM call "%~dp0notify_command.bat" "ZCA-SVD-RoBERTa-MLP" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name mlp-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-whitening-zca-svd-mlp --lr 2e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-roberta-whitening-zca-svd --lower --num_layers 2 --force

::PCA
call "%~dp0notify_command.bat" "PCA-BERT-BiLSTM" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-whitening-pca-bilstm --lr 2e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-bert-whitening-pca --lower --num_layers 2 --force
call "%~dp0notify_command.bat" "PCA-BERT-MLP" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name mlp-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-whitening-pca-mlp --lr 2e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-bert-whitening-pca --lower --num_layers 2 --force
call "%~dp0notify_command.bat" "PCA-RoBERTa-BiLSTM" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-whitening-pca-bilstm --lr 2e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-roberta-whitening-pca --lower --num_layers 2 --force
call "%~dp0notify_command.bat" "PCA-RoBERTa-MLP" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name mlp-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-whitening-pca-mlp --lr 2e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-roberta-whitening-pca --lower --num_layers 2 --force

::PCA-SVD
@REM call "%~dp0notify_command.bat" "PCA-SVD-BERT-BiLSTM" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-whitening-pca-svd-bilstm --lr 2e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-bert-whitening-pca-svd --lower --num_layers 2 --force
@REM call "%~dp0notify_command.bat" "PCA-SVD-BERT-MLP" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name mlp-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-whitening-pca-svd-mlp --lr 2e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-bert-whitening-pca-svd --lower --num_layers 2 --force
@REM call "%~dp0notify_command.bat" "PCA-SVD-RoBERTa-BiLSTM" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-whitening-pca-svd-bilstm --lr 2e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-roberta-whitening-pca-svd --lower --num_layers 2 --force
@REM call "%~dp0notify_command.bat" "PCA-SVD-RoBERTa-MLP" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name mlp-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-whitening-pca-svd-mlp --lr 2e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-roberta-whitening-pca-svd --lower --num_layers 2 --force

::EIGEN
@REM call "%~dp0notify_command.bat" "EIGEN-BERT-BiLSTM" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-whitening-eigen-bilstm --lr 2e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-bert-whitening-eigen --lower --num_layers 2 --force
@REM call "%~dp0notify_command.bat" "EIGEN-BERT-MLP" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name mlp-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-whitening-eigen-mlp --lr 2e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-bert-whitening-eigen --lower --num_layers 2 --force
@REM call "%~dp0notify_command.bat" "EIGEN-RoBERTa-BiLSTM" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-whitening-eigen-bilstm --lr 2e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-roberta-whitening-eigen --lower --num_layers 2 --force
@REM call "%~dp0notify_command.bat" "EIGEN-RoBERTa-MLP" python train_multiple_seed.py --n_epochs 5 --train_batch_size %3 --model_name mlp-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-whitening-eigen-mlp --lr 2e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-roberta-whitening-eigen --lower --num_layers 2 --force

@REM call "%~dp0notify_command.bat" "BERT-Whitening-Custom" python train_multiple_seed.py --n_epochs 2 --train_batch_size %3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-whitening-pca%3_step1_gamma0.9_adagrad_lr0.002_eps1e-8_early%2_layer2_lowerTrue --lr 1e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-bert-whitening --lower --num_layers 2 --force
@REM call "%~dp0notify_command.bat" "RoBERTa-Whitening-Custom" python train_multiple_seed.py --n_epochs 2 --train_batch_size %3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-whitening-pca%3_step1_gamma0.9_adagrad_lr0.002_eps1e-8_early%2_layer2_lowerTrue --lr 1e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-roberta-whitening --lower --num_layers 2 --force
