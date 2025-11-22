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

# Feature Extraction
# "$SCRIPT_DIR/notify_command.sh" "BERT-Extraction-BiLSTM" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name bilstm --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-extraction${3}_step1_gamma0.9_lr1e-4_early${2}_layer2_lowerTrue --lr 3e-3 --eps 1e-8 --early_stop $2 --dataset ag-news-bert-extraction --lower --num_layers 2 --force
# "$SCRIPT_DIR/notify_command.sh" "RoBERTa-Extraction-BiLSTM" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name bilstm --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-extraction${3}_step1_gamma0.9_lr1e-4_early${2}_layer2_lowerTrue --lr 3e-3 --eps 1e-8 --early_stop $2 --dataset ag-news-roberta-extraction --lower --num_layers 2 --force

# SVD
"$SCRIPT_DIR/notify_command.sh" "SVD-BERT-BiLSTM" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-whitening-svd-bilstm --lr 2e-3 --eps 1e-8 --early_stop $2 --dataset ag-news-bert-whitening-svd --lower --num_layers 2 --force
"$SCRIPT_DIR/notify_command.sh" "SVD-BERT-MLP" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name mlp-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-whitening-svd-mlp --lr 2e-3 --eps 1e-8 --early_stop $2 --dataset ag-news-bert-whitening-svd --lower --num_layers 2 --force
"$SCRIPT_DIR/notify_command.sh" "SVD-RoBERTa-BiLSTM" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-whitening-svd-bilstm --lr 2e-3 --eps 1e-8 --early_stop $2 --dataset ag-news-roberta-whitening-svd --lower --num_layers 2 --force
"$SCRIPT_DIR/notify_command.sh" "SVD-RoBERTa-MLP" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name mlp-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-whitening-svd-mlp --lr 2e-3 --eps 1e-8 --early_stop $2 --dataset ag-news-roberta-whitening-svd --lower --num_layers 2 --force

# ZCA
"$SCRIPT_DIR/notify_command.sh" "ZCA-BERT-BiLSTM" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 1 --train_batch_size $3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-whitening-zca-bilstm --lr 1.8e-3 --eps 1e-8 --early_stop $2 --dataset ag-news-bert-whitening-zca --lower --num_layers 2 --force
"$SCRIPT_DIR/notify_command.sh" "ZCA-BERT-MLP" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name mlp-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-whitening-zca-mlp --lr 1.8e-3 --eps 1e-8 --early_stop $2 --dataset ag-news-bert-whitening-zca --lower --num_layers 2 --force
"$SCRIPT_DIR/notify_command.sh" "ZCA-RoBERTa-BiLSTM" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-whitening-zca-bilstm --lr 2e-3 --eps 1e-8 --early_stop $2 --dataset ag-news-roberta-whitening-zca --lower --num_layers 2 --force
"$SCRIPT_DIR/notify_command.sh" "ZCA-RoBERTa-MLP" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name mlp-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-whitening-zca-mlp --lr 2e-3 --eps 1e-8 --early_stop $2 --dataset ag-news-roberta-whitening-zca --lower --num_layers 2 --force

# ZCA-SVD
# "$SCRIPT_DIR/notify_command.sh" "ZCA-SVD-BERT-BiLSTM" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-whitening-zca-svd-bilstm --lr 1.8e-3 --eps 1e-8 --early_stop $2 --dataset ag-news-bert-whitening-zca-svd --lower --num_layers 2 --force
# "$SCRIPT_DIR/notify_command.sh" "ZCA-SVD-BERT-MLP" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name mlp-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-whitening-zca-svd-mlp --lr 1.8e-3 --eps 1e-8 --early_stop $2 --dataset ag-news-bert-whitening-zca-svd --lower --num_layers 2 --force
# "$SCRIPT_DIR/notify_command.sh" "ZCA-SVD-RoBERTa-BiLSTM" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-whitening-zca-svd-bilstm --lr 2e-3 --eps 1e-8 --early_stop $2 --dataset ag-news-roberta-whitening-zca-svd --lower --num_layers 2 --force
# "$SCRIPT_DIR/notify_command.sh" "ZCA-SVD-RoBERTa-MLP" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name mlp-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-whitening-zca-svd-mlp --lr 2e-3 --eps 1e-8 --early_stop $2 --dataset ag-news-roberta-whitening-zca-svd --lower --num_layers 2 --force

# PCA
"$SCRIPT_DIR/notify_command.sh" "PCA-BERT-BiLSTM" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-whitening-pca-bilstm --lr 2e-3 --eps 1e-8 --early_stop $2 --dataset ag-news-bert-whitening-pca --lower --num_layers 2 --force
"$SCRIPT_DIR/notify_command.sh" "PCA-BERT-MLP" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name mlp-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-whitening-pca-mlp --lr 2e-3 --eps 1e-8 --early_stop $2 --dataset ag-news-bert-whitening-pca --lower --num_layers 2 --force
"$SCRIPT_DIR/notify_command.sh" "PCA-RoBERTa-BiLSTM" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-whitening-pca-bilstm --lr 2e-3 --eps 1e-8 --early_stop $2 --dataset ag-news-roberta-whitening-pca --lower --num_layers 2 --force
"$SCRIPT_DIR/notify_command.sh" "PCA-RoBERTa-MLP" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name mlp-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-whitening-pca-mlp --lr 2e-3 --eps 1e-8 --early_stop $2 --dataset ag-news-roberta-whitening-pca --lower --num_layers 2 --force

# PCA-SVD
# "$SCRIPT_DIR/notify_command.sh" "PCA-SVD-BERT-BiLSTM" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-whitening-pca-svd-bilstm --lr 2e-3 --eps 1e-8 --early_stop $2 --dataset ag-news-bert-whitening-pca-svd --lower --num_layers 2 --force
# "$SCRIPT_DIR/notify_command.sh" "PCA-SVD-BERT-MLP" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name mlp-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-whitening-pca-svd-mlp --lr 2e-3 --eps 1e-8 --early_stop $2 --dataset ag-news-bert-whitening-pca-svd --lower --num_layers 2 --force
# "$SCRIPT_DIR/notify_command.sh" "PCA-SVD-RoBERTa-BiLSTM" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-whitening-pca-svd-bilstm --lr 2e-3 --eps 1e-8 --early_stop $2 --dataset ag-news-roberta-whitening-pca-svd --lower --num_layers 2 --force
# "$SCRIPT_DIR/notify_command.sh" "PCA-SVD-RoBERTa-MLP" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name mlp-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-whitening-pca-svd-mlp --lr 2e-3 --eps 1e-8 --early_stop $2 --dataset ag-news-roberta-whitening-pca-svd --lower --num_layers 2 --force

# EIGEN
# "$SCRIPT_DIR/notify_command.sh" "EIGEN-BERT-BiLSTM" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-whitening-eigen-bilstm --lr 2e-3 --eps 1e-8 --early_stop $2 --dataset ag-news-bert-whitening-eigen --lower --num_layers 2 --force
# "$SCRIPT_DIR/notify_command.sh" "EIGEN-BERT-MLP" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name mlp-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-whitening-eigen-mlp --lr 2e-3 --eps 1e-8 --early_stop $2 --dataset ag-news-bert-whitening-eigen --lower --num_layers 2 --force
# "$SCRIPT_DIR/notify_command.sh" "EIGEN-RoBERTa-BiLSTM" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-whitening-eigen-bilstm --lr 2e-3 --eps 1e-8 --early_stop $2 --dataset ag-news-roberta-whitening-eigen --lower --num_layers 2 --force
# "$SCRIPT_DIR/notify_command.sh" "EIGEN-RoBERTa-MLP" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 5 --train_batch_size $3 --model_name mlp-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-whitening-eigen-mlp --lr 2e-3 --eps 1e-8 --early_stop $2 --dataset ag-news-roberta-whitening-eigen --lower --num_layers 2 --force

# Custom Whitening
# "$SCRIPT_DIR/notify_command.sh" "BERT-Whitening-Custom" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 2 --train_batch_size $3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-whitening-pca${3}_step1_gamma0.9_adagrad_lr0.002_eps1e-8_early${2}_layer2_lowerTrue --lr 1e-3 --eps 1e-8 --early_stop $2 --dataset ag-news-bert-whitening --lower --num_layers 2 --force
# "$SCRIPT_DIR/notify_command.sh" "RoBERTa-Whitening-Custom" python "$PROJECT_DIR/train_multiple_seeds.py" --n_epochs 2 --train_batch_size $3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-whitening-pca${3}_step1_gamma0.9_adagrad_lr0.002_eps1e-8_early${2}_layer2_lowerTrue --lr 1e-3 --eps 1e-8 --early_stop $2 --dataset ag-news-roberta-whitening --lower --num_layers 2 --force
