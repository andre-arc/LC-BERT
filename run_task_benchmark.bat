@echo off
set CUDA_VISIBLE_DEVICES=%1

::adam optimizer
python main.py --n_epochs 5 --train_batch_size %3 --model_name bert-base-uncased --step_size 1 --gamma 0.9 --experiment_name bert_benchmark_b%3_step1_gamma0.9_lr1e-4_early%2_layer2_lowerTrue --lr 1e-4 --early_stop %2 --dataset ag-news-normal --lower --num_layers 2 --force
python main.py --n_epochs 5 --train_batch_size %3 --model_name roberta-base --step_size 1 --gamma 0.9 --experiment_name roberta_benchmark_b%3_step1_gamma0.9_lr1e-4_early%2_layer2_lowerTrue --lr 1e-4 --early_stop %2 --dataset ag-news-normal --lower --num_layers 2 --force

::adamW Optimizer
@REM python main.py --n_epochs 5 --train_batch_size %3 --model_name bert-base-uncased --step_size 1 --gamma 0.9 --experiment_name bert_benchmark_b%3_step1_gamma0.9_adamW_lr1e-4_early%2_layer2_lowerTrue --lr 1e-4 --early_stop %2 --dataset ag-news-normal --lower --num_layers 2 --force
@REM python main.py --n_epochs 5 --train_batch_size %3 --model_name roberta-base --step_size 1 --gamma 0.9 --experiment_name roberta_benchmark_b%3_step1_gamma0.9_adamW_lr1e-4_early%2_layer2_lowerTrue --lr 1e-4 --early_stop %2 --dataset ag-news-normal --lower --num_layers 2 --force