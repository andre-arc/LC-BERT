@echo off
set CUDA_VISIBLE_DEVICES=%1
@REM python main.py --n_epochs 5 --train_batch_size %3 --model_name bilstm --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-extraction%3_step1_gamma0.9_lr1e-4_early%2_layer2_lowerTrue --lr 3e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-bert-extraction --lower --num_layers 2 --force
@REM python main.py --n_epochs 5 --train_batch_size %3 --model_name bilstm --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-extraction%3_step1_gamma0.9_lr1e-4_early%2_layer2_lowerTrue --lr 3e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-roberta-extraction --lower --num_layers 2 --force

::ZCA
@REM python main.py --n_epochs 5 --train_batch_size %3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-whitening-zca-bilstm --lr 1.8e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-bert-whitening-zca --lower --num_layers 2 --force
python main.py --n_epochs 1 --train_batch_size %3 --model_name mlp-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-whitening-zca-mlp --lr 1.8e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-bert-whitening-zca --lower --num_layers 2 --force
@REM python main.py --n_epochs 5 --train_batch_size %3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 42 --experiment_name ag-news-roberta-whitening-zca-bilstm --lr 2e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-roberta-whitening-zca --lower --num_layers 2 --force
@REM python main.py --n_epochs 5 --train_batch_size %3 --model_name mlp-dim-reduction --step_size 1 --gamma 0.9 --seed 42 --experiment_name ag-news-roberta-whitening-zca-mlp --lr 2e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-roberta-whitening-zca --lower --num_layers 2 --force

@REM python main.py --n_epochs 5 --train_batch_size %3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-whitening-pca%3_step1_gamma0.9_adamW_lr2e-3_eps1e-8_early%2_layer2_lowerTrue --lr 2e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-bert-whitening --lower --num_layers 2 --force
@REM python main.py --n_epochs 5 --train_batch_size %3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-whitening-pca%3_step1_gamma0.9_adamW_lr2e-3_eps1e-8_early%2_layer2_lowerTrue --lr 2e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-roberta-whitening --lower --num_layers 2 --force

@REM python main.py --n_epochs 2 --train_batch_size %3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-bert-whitening-pca%3_step1_gamma0.9_adagrad_lr0.002_eps1e-8_early%2_layer2_lowerTrue --lr 1e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-bert-whitening --lower --num_layers 2 --force
@REM python main.py --n_epochs 2 --train_batch_size %3 --model_name bilstm-dim-reduction --step_size 1 --gamma 0.9 --seed 88 --experiment_name ag-news-roberta-whitening-pca%3_step1_gamma0.9_adagrad_lr0.002_eps1e-8_early%2_layer2_lowerTrue --lr 1e-3 --eps 1e-8 --early_stop %2 --dataset ag-news-roberta-whitening --lower --num_layers 2 --force
