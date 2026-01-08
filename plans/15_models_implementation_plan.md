# 15 Models Implementation Plan

## Overview
This document provides a detailed implementation plan for 15 model scenarios defined in research project. The plan maps each scenario to existing code capabilities and identifies any gaps or missing implementations.

## Model Scenarios Mapping

### Skenario 1: Benchmark (Baseline Models)

#### A01: BERT Benchmark
- **Feature Extraction**: None (standard BERT fine-tuning)
- **Classifier**: BERT (frozen base + classifier head)
- **Dimensionality Reduction**: None
- **Status**: ✅ Already Implemented
- **Implementation**: `ag-news-normal` with `bert-base-uncased`
- **Command**: 
  ```bash
  python main.py --dataset ag-news-normal --model_name bert-base-uncased --experiment_name A01_bert_benchmark
  ```

#### A02: RoBERTa Benchmark
- **Feature Extraction**: None (standard RoBERTa fine-tuning)
- **Classifier**: RoBERTa (frozen base + classifier head)
- **Dimensionality Reduction**: None
- **Status**: ✅ Already Implemented
- **Implementation**: `ag-news-normal` with `roberta-base`
- **Command**: 
  ```bash
  python main.py --dataset ag-news-normal --model_name roberta-base --experiment_name A02_roberta_benchmark
  ```

#### A03: DistilBERT Benchmark
- **Feature Extraction**: None (standard DistilBERT fine-tuning)
- **Classifier**: DistilBERT (frozen base + classifier head)
- **Dimensionality Reduction**: None
- **Status**: ✅ Already Implemented
- **Implementation**: `ag-news-normal` with `distilbert-base-uncased`
- **Command**: 
  ```bash
  python main.py --dataset ag-news-normal --model_name distilbert-base-uncased --experiment_name A03_distilbert_benchmark
  ```

---

### Skenario 2: Modified Models with BiLSTM Classifier

#### B01: BERT + SVD Whitening + BiLSTM
- **Feature Extraction**: BERT Attention Layer
- **Classifier**: Custom Bi-LSTM
- **Dimensionality Reduction**: J.Su Whitening (SVD)
- **Status**: ✅ Already Implemented
- **Implementation**: `ag-news-bert-whitening-svd` with `bilstm-dim-reduction`
- **Command**: 
  ```bash
  python main.py --dataset ag-news-bert-whitening-svd --model_name bilstm-dim-reduction --experiment_name B01_bert_svd_bilstm
  ```

#### B02: BERT + PCA Whitening + BiLSTM
- **Feature Extraction**: BERT Attention Layer
- **Classifier**: Custom Bi-LSTM (inferred from pattern)
- **Dimensionality Reduction**: PCA Whitening
- **Status**: ✅ Already Implemented
- **Implementation**: `ag-news-bert-whitening-pca` with `bilstm-dim-reduction`
- **Command**: 
  ```bash
  python main.py --dataset ag-news-bert-whitening-pca --model_name bilstm-dim-reduction --experiment_name B02_bert_pca_bilstm
  ```

#### B03: BERT + ZCA Whitening + BiLSTM
- **Feature Extraction**: BERT Attention Layer
- **Classifier**: Custom Bi-LSTM (inferred from pattern)
- **Dimensionality Reduction**: ZCA Whitening
- **Status**: ✅ Already Implemented
- **Implementation**: `ag-news-bert-whitening-zca` with `bilstm-dim-reduction`
- **Command**: 
  ```bash
  python main.py --dataset ag-news-bert-whitening-zca --model_name bilstm-dim-reduction --experiment_name B03_bert_zca_bilstm
  ```

#### B04: RoBERTa + SVD Whitening + BiLSTM
- **Feature Extraction**: RoBERTa Attention Layer
- **Classifier**: Custom Bi-LSTM
- **Dimensionality Reduction**: J.Su Whitening (SVD)
- **Status**: ✅ Already Implemented
- **Implementation**: `ag-news-roberta-whitening-svd` with `bilstm-dim-reduction`
- **Command**: 
  ```bash
  python main.py --dataset ag-news-roberta-whitening-svd --model_name bilstm-dim-reduction --experiment_name B04_roberta_svd_bilstm
  ```

#### B05: RoBERTa + PCA Whitening + BiLSTM
- **Feature Extraction**: RoBERTa Attention Layer
- **Classifier**: Custom Bi-LSTM (inferred from pattern)
- **Dimensionality Reduction**: PCA Whitening
- **Status**: ✅ Already Implemented
- **Implementation**: `ag-news-roberta-whitening-pca` with `bilstm-dim-reduction`
- **Command**: 
  ```bash
  python main.py --dataset ag-news-roberta-whitening-pca --model_name bilstm-dim-reduction --experiment_name B05_roberta_pca_bilstm
  ```

#### B06: RoBERTa + ZCA Whitening + BiLSTM
- **Feature Extraction**: RoBERTa Attention Layer
- **Classifier**: Custom Bi-LSTM (inferred from pattern)
- **Dimensionality Reduction**: ZCA Whitening
- **Status**: ✅ Already Implemented
- **Implementation**: `ag-news-roberta-whitening-zca` with `bilstm-dim-reduction`
- **Command**: 
  ```bash
  python main.py --dataset ag-news-roberta-whitening-zca --model_name bilstm-dim-reduction --experiment_name B06_roberta_zca_bilstm
  ```

---

### Skenario 3: Modified Models with MLP Classifier

#### C01: BERT + SVD Whitening + MLP
- **Feature Extraction**: BERT Attention Layer
- **Classifier**: BERT Classifier (MLP)
- **Dimensionality Reduction**: J.Su Whitening (SVD)
- **Status**: ✅ Already Implemented
- **Implementation**: `ag-news-bert-whitening-svd` with `mlp-dim-reduction`
- **Command**: 
  ```bash
  python main.py --dataset ag-news-bert-whitening-svd --model_name mlp-dim-reduction --experiment_name C01_bert_svd_mlp
  ```

#### C02: BERT + PCA Whitening + MLP
- **Feature Extraction**: BERT Attention Layer
- **Classifier**: BERT Classifier (MLP) (inferred from pattern)
- **Dimensionality Reduction**: PCA Whitening
- **Status**: ✅ Already Implemented
- **Implementation**: `ag-news-bert-whitening-pca` with `mlp-dim-reduction`
- **Command**: 
  ```bash
  python main.py --dataset ag-news-bert-whitening-pca --model_name mlp-dim-reduction --experiment_name C02_bert_pca_mlp
  ```

#### C03: BERT + ZCA Whitening + MLP
- **Feature Extraction**: BERT Attention Layer
- **Classifier**: BERT Classifier (MLP) (inferred from pattern)
- **Dimensionality Reduction**: ZCA Whitening
- **Status**: ✅ Already Implemented
- **Implementation**: `ag-news-bert-whitening-zca` with `mlp-dim-reduction`
- **Command**: 
  ```bash
  python main.py --dataset ag-news-bert-whitening-zca --model_name mlp-dim-reduction --experiment_name C03_bert_zca_mlp
  ```

#### C04: RoBERTa + SVD Whitening + MLP
- **Feature Extraction**: RoBERTa Attention Layer
- **Classifier**: Roberta Classifier (MLP)
- **Dimensionality Reduction**: J.Su Whitening (SVD)
- **Status**: ✅ Already Implemented
- **Implementation**: `ag-news-roberta-whitening-svd` with `mlp-dim-reduction`
- **Command**: 
  ```bash
  python main.py --dataset ag-news-roberta-whitening-svd --model_name mlp-dim-reduction --experiment_name C04_roberta_svd_mlp
  ```

#### C05: RoBERTa + PCA Whitening + MLP
- **Feature Extraction**: RoBERTa Attention Layer
- **Classifier**: Roberta Classifier (MLP) (inferred from pattern)
- **Dimensionality Reduction**: PCA Whitening
- **Status**: ✅ Already Implemented
- **Implementation**: `ag-news-roberta-whitening-pca` with `mlp-dim-reduction`
- **Command**: 
  ```bash
  python main.py --dataset ag-news-roberta-whitening-pca --model_name mlp-dim-reduction --experiment_name C05_roberta_pca_mlp
  ```

#### C06: RoBERTa + ZCA Whitening + MLP
- **Feature Extraction**: RoBERTa Attention Layer
- **Classifier**: Roberta Classifier (MLP) (inferred from pattern)
- **Dimensionality Reduction**: ZCA Whitening
- **Status**: ✅ Already Implemented
- **Implementation**: `ag-news-roberta-whitening-zca` with `mlp-dim-reduction`
- **Command**: 
  ```bash
  python main.py --dataset ag-news-roberta-whitening-zca --model_name mlp-dim-reduction --experiment_name C06_roberta_zca_mlp
  ```

---

## Summary Table

| Code | Feature Extraction | Classifier | Dim Reduction | Dataset | Model Name | Status |
|------|-------------------|------------|--------------|----------|------------|--------|
| A01 | None | BERT | None | ag-news-normal | bert-base-uncased | ✅ Implemented |
| A02 | None | RoBERTa | None | ag-news-normal | roberta-base | ✅ Implemented |
| A03 | None | DistilBERT | None | ag-news-normal | distilbert-base-uncased | ✅ Implemented |
| B01 | BERT Attention | BiLSTM | SVD | ag-news-bert-whitening-svd | bilstm-dim-reduction | ✅ Implemented |
| B02 | BERT Attention | BiLSTM | PCA | ag-news-bert-whitening-pca | bilstm-dim-reduction | ✅ Implemented |
| B03 | BERT Attention | BiLSTM | ZCA | ag-news-bert-whitening-zca | bilstm-dim-reduction | ✅ Implemented |
| B04 | RoBERTa Attention | BiLSTM | SVD | ag-news-roberta-whitening-svd | bilstm-dim-reduction | ✅ Implemented |
| B05 | RoBERTa Attention | BiLSTM | PCA | ag-news-roberta-whitening-pca | bilstm-dim-reduction | ✅ Implemented |
| B06 | RoBERTa Attention | BiLSTM | ZCA | ag-news-roberta-whitening-zca | bilstm-dim-reduction | ✅ Implemented |
| C01 | BERT Attention | MLP | SVD | ag-news-bert-whitening-svd | mlp-dim-reduction | ✅ Implemented |
| C02 | BERT Attention | MLP | PCA | ag-news-bert-whitening-pca | mlp-dim-reduction | ✅ Implemented |
| C03 | BERT Attention | MLP | ZCA | ag-news-bert-whitening-zca | mlp-dim-reduction | ✅ Implemented |
| C04 | RoBERTa Attention | MLP | SVD | ag-news-roberta-whitening-svd | mlp-dim-reduction | ✅ Implemented |
| C05 | RoBERTa Attention | MLP | PCA | ag-news-roberta-whitening-pca | mlp-dim-reduction | ✅ Implemented |
| C06 | RoBERTa Attention | MLP | ZCA | ag-news-roberta-whitening-zca | mlp-dim-reduction | ✅ Implemented |

---

## Key Findings

### ✅ All 15 Models Are Already Implemented
The project already has complete implementations for all 15 model scenarios. The existing codebase supports:

1. **Benchmark Models (Skenario 1)**: Standard BERT, RoBERTa, and DistilBERT fine-tuning
2. **Modified Models with BiLSTM (Skenario 2)**: BERT/RoBERTa + SVD/PCA/ZCA + BiLSTM
3. **Modified Models with MLP (Skenario 3)**: BERT/RoBERTa + SVD/PCA/ZCA + MLP

### 📝 Notes on Original Table
The original code scenario table had empty classifier cells for models B02, B03, B05, B06, C02, C03, C05, and C06. Based on pattern in table and existing implementations, these have been inferred as:
- Skenario 2 (B-series): BiLSTM classifier
- Skenario 3 (C-series): MLP classifier

### ➕ Added Model
A03 (DistilBERT Benchmark) has been added to benchmark scenario to provide an additional baseline comparison with a more efficient transformer model.

### 🔧 Existing Code Components

#### Feature Extraction
- **Location**: `data_utils/ag_news/extraction.py` and `data_utils/ag_news/whitening.py`
- **Methods**: BERT/RoBERTa attention layer extraction with first+last layer averaging

#### Dimensionality Reduction
- **Location**: `data_utils/ag_news/whitening.py`
- **Techniques**:
  - SVD (J.Su Whitening)
  - PCA Whitening
  - ZCA Whitening
  - Eigen (experimental)
  - PCA-SVD (experimental)
  - ZCA-SVD (experimental)

#### Classifiers
- **Location**: `modules/word_classification.py` and `modules/modified_word_classification.py`
- **Types**:
  - BERT/RoBERTa/DistilBERT with frozen base + classifier head
  - BiLSTM (for 768-dim and 256-dim features)
  - MLP (for 256-dim features)

#### Dataset Configurations
- **Location**: `utils/args_helper.py`
- **Supported Datasets**:
  - `ag-news-normal`: Standard fine-tuning (supports bert-base-uncased, roberta-base, distilbert-base-uncased)
  - `ag-news-bert-extraction`: BERT extraction (768-dim)
  - `ag-news-roberta-extraction`: RoBERTa extraction (768-dim)
  - `ag-news-bert-whitening-{technique}`: BERT + whitening (256-dim)
  - `ag-news-roberta-whitening-{technique}`: RoBERTa + whitening (256-dim)

---

## Next Steps

### Option 1: Run All 15 Models
Create a comprehensive batch script to run all 15 models with consistent hyperparameters.

### Option 2: Create Comparison Analysis
Generate a unified analysis script to compare results across all 15 models.

### Option 3: Optimize Existing Implementations
Review and potentially optimize existing code for better performance or efficiency.

### Option 4: Extend with Additional Models
Add new model variants or techniques beyond current 15 scenarios.

---

## Recommended Hyperparameters

Based on existing batch scripts, following hyperparameters are recommended:

### For Benchmark Models (A01, A02, A03)
```bash
--n_epochs 5 --train_batch_size 32 --lr 6.25e-5 --eps 6.25e-5 --early_stop 3 --step_size 1 --gamma 0.5 --seed 88 --lower --force
```

### For Modified Models with BiLSTM (B01-B06)
```bash
--n_epochs 5 --train_batch_size 32 --lr 2e-3 --eps 1e-8 --early_stop 3 --step_size 1 --gamma 0.9 --seed 88 --num_layers 2 --lower --force
```

### For Modified Models with MLP (C01-C06)
```bash
--n_epochs 5 --train_batch_size 32 --lr 2e-3 --eps 1e-8 --early_stop 3 --step_size 1 --gamma 0.9 --seed 88 --num_layers 2 --lower --force
```

---

## Batch Script Template

A Windows batch script to run all 15 models:

```batch
@echo off
setlocal enabledelayedexpansion

set "GPU_ID=%1"
set "EARLY_STOP=%2"
set "BATCH_SIZE=%3"

REM Skenario 1: Benchmark
call python main.py --dataset ag-news-normal --model_name bert-base-uncased --experiment_name A01_bert_benchmark --n_epochs 5 --train_batch_size %BATCH_SIZE% --lr 6.25e-5 --eps 6.25e-5 --early_stop %EARLY_STOP% --step_size 1 --gamma 0.5 --seed 88 --lower --force
call python main.py --dataset ag-news-normal --model_name roberta-base --experiment_name A02_roberta_benchmark --n_epochs 5 --train_batch_size %BATCH_SIZE% --lr 6.25e-5 --eps 6.25e-5 --early_stop %EARLY_STOP% --step_size 1 --gamma 0.5 --seed 88 --lower --force
call python main.py --dataset ag-news-normal --model_name distilbert-base-uncased --experiment_name A03_distilbert_benchmark --n_epochs 5 --train_batch_size %BATCH_SIZE% --lr 6.25e-5 --eps 6.25e-5 --early_stop %EARLY_STOP% --step_size 1 --gamma 0.5 --seed 88 --lower --force

REM Skenario 2: BiLSTM Classifier
call python main.py --dataset ag-news-bert-whitening-svd --model_name bilstm-dim-reduction --experiment_name B01_bert_svd_bilstm --n_epochs 5 --train_batch_size %BATCH_SIZE% --lr 2e-3 --eps 1e-8 --early_stop %EARLY_STOP% --step_size 1 --gamma 0.9 --seed 88 --num_layers 2 --lower --force
call python main.py --dataset ag-news-bert-whitening-pca --model_name bilstm-dim-reduction --experiment_name B02_bert_pca_bilstm --n_epochs 5 --train_batch_size %BATCH_SIZE% --lr 2e-3 --eps 1e-8 --early_stop %EARLY_STOP% --step_size 1 --gamma 0.9 --seed 88 --num_layers 2 --lower --force
call python main.py --dataset ag-news-bert-whitening-zca --model_name bilstm-dim-reduction --experiment_name B03_bert_zca_bilstm --n_epochs 5 --train_batch_size %BATCH_SIZE% --lr 2e-3 --eps 1e-8 --early_stop %EARLY_STOP% --step_size 1 --gamma 0.9 --seed 88 --num_layers 2 --lower --force
call python main.py --dataset ag-news-roberta-whitening-svd --model_name bilstm-dim-reduction --experiment_name B04_roberta_svd_bilstm --n_epochs 5 --train_batch_size %BATCH_SIZE% --lr 2e-3 --eps 1e-8 --early_stop %EARLY_STOP% --step_size 1 --gamma 0.9 --seed 88 --num_layers 2 --lower --force
call python main.py --dataset ag-news-roberta-whitening-pca --model_name bilstm-dim-reduction --experiment_name B05_roberta_pca_bilstm --n_epochs 5 --train_batch_size %BATCH_SIZE% --lr 2e-3 --eps 1e-8 --early_stop %EARLY_STOP% --step_size 1 --gamma 0.9 --seed 88 --num_layers 2 --lower --force
call python main.py --dataset ag-news-roberta-whitening-zca --model_name bilstm-dim-reduction --experiment_name B06_roberta_zca_bilstm --n_epochs 5 --train_batch_size %BATCH_SIZE% --lr 2e-3 --eps 1e-8 --early_stop %EARLY_STOP% --step_size 1 --gamma 0.9 --seed 88 --num_layers 2 --lower --force

REM Skenario 3: MLP Classifier
call python main.py --dataset ag-news-bert-whitening-svd --model_name mlp-dim-reduction --experiment_name C01_bert_svd_mlp --n_epochs 5 --train_batch_size %BATCH_SIZE% --lr 2e-3 --eps 1e-8 --early_stop %EARLY_STOP% --step_size 1 --gamma 0.9 --seed 88 --num_layers 2 --lower --force
call python main.py --dataset ag-news-bert-whitening-pca --model_name mlp-dim-reduction --experiment_name C02_bert_pca_mlp --n_epochs 5 --train_batch_size %BATCH_SIZE% --lr 2e-3 --eps 1e-8 --early_stop %EARLY_STOP% --step_size 1 --gamma 0.9 --seed 88 --num_layers 2 --lower --force
call python main.py --dataset ag-news-bert-whitening-zca --model_name mlp-dim-reduction --experiment_name C03_bert_zca_mlp --n_epochs 5 --train_batch_size %BATCH_SIZE% --lr 2e-3 --eps 1e-8 --early_stop %EARLY_STOP% --step_size 1 --gamma 0.9 --seed 88 --num_layers 2 --lower --force
call python main.py --dataset ag-news-roberta-whitening-svd --model_name mlp-dim-reduction --experiment_name C04_roberta_svd_mlp --n_epochs 5 --train_batch_size %BATCH_SIZE% --lr 2e-3 --eps 1e-8 --early_stop %EARLY_STOP% --step_size 1 --gamma 0.9 --seed 88 --num_layers 2 --lower --force
call python main.py --dataset ag-news-roberta-whitening-pca --model_name mlp-dim-reduction --experiment_name C05_roberta_pca_mlp --n_epochs 5 --train_batch_size %BATCH_SIZE% --lr 2e-3 --eps 1e-8 --early_stop %EARLY_STOP% --step_size 1 --gamma 0.9 --seed 88 --num_layers 2 --lower --force
call python main.py --dataset ag-news-roberta-whitening-zca --model_name mlp-dim-reduction --experiment_name C06_roberta_zca_mlp --n_epochs 5 --train_batch_size %BATCH_SIZE% --lr 2e-3 --eps 1e-8 --early_stop %EARLY_STOP% --step_size 1 --gamma 0.9 --seed 88 --num_layers 2 --lower --force

endlocal
```

---

## Conclusion

All 15 model scenarios from the research design are already implemented in the LC-BERT project. The codebase provides a comprehensive framework for:

1. **Benchmark comparisons** (A01, A02, A03)
2. **Dimensionality reduction experiments** (B01-B06, C01-C06)
3. **Classifier comparisons** (BiLSTM vs MLP)
4. **Model comparisons** (BERT vs RoBERTa vs DistilBERT)
5. **Whitening technique comparisons** (SVD, PCA, ZCA)

The project is ready for comprehensive experimentation and analysis across all 15 scenarios.
