# LC-BERT Project Context Memory

**Last Updated**: 2026-01-02  
**Project**: LC-BERT - Lightweight Classification with BERT Embeddings  
**Purpose**: Comprehensive context memory for understanding and working with this project

---

## Executive Summary

LC-BERT is a research project that explores dimensionality reduction techniques applied to BERT/RoBERTa embeddings for efficient text classification. The project implements a two-stage pipeline:

1. **Feature Extraction**: Extract embeddings from pretrained BERT/RoBERTa models and apply dimensionality reduction (768 → 256 dimensions) using whitening techniques
2. **Classification**: Train lightweight classifiers (BiLSTM or MLP) on the reduced embeddings

**Goal**: Reduce computational costs while maintaining competitive performance compared to full BERT fine-tuning.

**Dataset**: AG News (text classification with 4 categories)

---

## Architecture Overview

### Two-Stage Pipeline

```
Input Text → BERT/RoBERTa (frozen) → Embeddings (768-dim) → Whitening (ZCA/PCA/SVD) → Reduced Embeddings (256-dim) → BiLSTM/MLP → Classification
```

### Stage 1: Feature Extraction
- **Location**: [`data_utils/ag_news/whitening.py`](data_utils/ag_news/whitening.py)
- **Models**: BERT, RoBERTa, DistilBERT, ALBERT (all frozen)
- **Dimensionality Reduction**: 768 → 256 dimensions
- **Whitening Techniques**:
  - **Recommended (Production-Ready)**:
    - `zca`: ZCA whitening using eigendecomposition
    - `pca`: PCA whitening using eigendecomposition
    - `svd`: Pure SVD-based reduction
  - **Experimental (Not Recommended)**:
    - `zca-svd`: ZCA whitening using SVD (under development)
    - `pca-svd`: PCA whitening using SVD (under development)
    - `eigen`: Eigendecomposition-based reduction (under development)
- **Pooling Strategy**: `first_last_avg` (average of first and last hidden states)
- **Numerical Stability**: Controlled by `EPSILON = 1e-5` (configurable)

### Stage 2: Classification
- **Location**: [`modules/modified_word_classification.py`](modules/modified_word_classification.py)
- **Classifiers**:
  - **BiLSTM**: 256 input → 32 hidden → num_labels output
  - **MLP**: 256 input → 32 hidden → num_labels output
- **Training**: Only classification heads are trained; BERT/RoBERTa parameters remain frozen

---

## Key Features

1. **Multiple Whitening Techniques**: ZCA, PCA, and SVD transformations for dimensionality reduction
2. **Flexible Preprocessing**: Support for three preprocessing methods (Gensim, NLTK, Stanza) with different speed/accuracy tradeoffs
3. **Model Variety**: Compare BERT, RoBERTa, DistilBERT, and ALBERT embeddings
4. **Lightweight Classifiers**: BiLSTM and MLP architectures for efficient training
5. **K-Fold Cross-Validation**: Built-in support for robust evaluation (5-fold by default)
6. **Telegram Notifications**: Get notified when training tasks start, succeed, or fail
7. **Comprehensive Logging**: Track efficiency metrics (GPU usage, training time)
8. **Batch Processing**: Automated scripts for running multiple experiments
9. **Efficiency Analysis**: Automated workflow for running experiments across different data subset sizes (10%-100%) with visualization

---

## Project Structure

```
LC-BERT/
├── data_utils/              # Dataset loaders and preprocessing
│   └── ag_news/
│       ├── normal.py        # Standard BERT fine-tuning
│       ├── extraction.py    # Feature extraction only
│       └── whitening.py     # Whitening transformations
├── modules/                 # Model architectures
│   ├── word_classification.py         # Standard BERT classifier
│   └── modified_word_classification.py # Lightweight classifiers
├── utils/                   # Utility functions
│   ├── preprocessing.py     # Text preprocessing (3 methods)
│   ├── args_helper.py       # Command-line arguments
│   ├── functions.py         # Model loading utilities
│   ├── forward_fn.py        # Forward pass functions
│   └── metrics.py           # Evaluation metrics
├── windows_scripts/         # Windows batch scripts
│   ├── run_task_modified.bat       # Main training script
│   ├── run_task_benchmark.bat      # Benchmark experiments
│   ├── run_with_notification.bat   # Notification wrapper
│   ├── run_efficiency_analysis_auto.bat  # Automated efficiency analysis
│   ├── run_complete_efficiency_analysis.bat  # Complete pipeline
│   └── telegram_config.bat         # Telegram configuration
├── linux_scripts/           # Linux shell scripts
│   ├── run_task_modified.sh
│   ├── run_task_benchmark.sh
│   └── telegram_config.sh
├── main.py                  # Main training script
├── kfold_analysis.py        # K-fold cross-validation script
├── train_multiple_seeds.py  # Multi-seed training
├── telegram_notifier.py     # Telegram notification utility
├── efficient_analysis.py    # Core efficiency analysis script
├── aggregate_efficiency_results.py  # Aggregation script
├── visualize_efficiency.py  # Visualization script
├── efficiency_config.txt    # Configuration for efficiency analysis
├── environment.yml          # Conda environment (full)
├── requirements.txt         # Pip requirements
├── .env.example             # Example Telegram config
├── README.md                # Main project documentation
├── CLAUDE.md                # Guidance for Claude Code
├── CLEAN_ENVIRONMENT.md     # Environment cleanup guide
├── EFFICIENCY_ANALYSIS_GUIDE.md  # Efficiency analysis documentation
├── EFFICIENCY_QUICK_REFERENCE.md  # Quick reference for efficiency
├── EPSILON_USAGE.md         # Epsilon parameter documentation
└── TELEGRAM_SETUP.md        # Telegram notification setup
```

---

## Installation & Setup

### Quick Start

```bash
# 1. Clone the repository
git clone <repository-url>
cd LC-BERT

# 2. Create conda environment
conda env create -f environment.yml

# 3. Activate environment
conda activate lc-bert

# 4. Install PyTorch (CUDA 11.3)
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# 5. Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# 6. (Optional) Download Stanza model for advanced preprocessing
python -c "import stanza; stanza.download('en')"

# 7. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Environment Cleanup

If you need to clean and recreate the environment (see [`CLEAN_ENVIRONMENT.md`](CLEAN_ENVIRONMENT.md)):

```bash
# Remove and recreate (recommended)
conda deactivate
conda env remove -n lc-bert
conda env create -f environment.yml
conda activate lc-bert
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

**Benefits of clean environment**:
- Reduces from 367+ packages to ~18 packages
- Reduces size from ~10-15 GB to ~3-4 GB
- Faster install time (5-10 min vs 30-60 min)
- Lower risk of conflicts

---

## Usage

### Training Models

#### Basic Training

```bash
python main.py \
  --n_epochs 5 \
  --train_batch_size 32 \
  --model_name bilstm-dim-reduction \
  --experiment_name my_experiment \
  --dataset ag-news-bert-whitening-zca \
  --lr 1.8e-3 \
  --early_stop 3 \
  --lower \
  --force
```

#### K-Fold Cross-Validation

```bash
python kfold_analysis.py \
  --n_epochs 5 \
  --train_batch_size 32 \
  --model_name bilstm-dim-reduction \
  --experiment_name my_kfold_experiment \
  --dataset ag-news-bert-whitening-pca \
  --lr 2e-3 \
  --early_stop 3 \
  --lower \
  --force
```

#### Subset Training (for quick experiments)

```bash
python main.py \
  --n_epochs 5 \
  --train_batch_size 32 \
  --model_name bilstm-dim-reduction \
  --experiment_name quick_test \
  --dataset ag-news-bert-whitening-svd \
  --lr 2e-3 \
  --subset_percentage 10 \
  --force
```

### Batch Scripts (Windows)

All batch scripts accept three parameters:
1. **GPU_ID** (CUDA_VISIBLE_DEVICES)
2. **EARLY_STOP** (patience)
3. **BATCH_SIZE**

```batch
# Standard training run
windows_scripts\run_task_modified.bat 0 3 32

# K-fold cross-validation
windows_scripts\run_task_modified_kfold.bat 0 3 32

# Benchmark baselines
windows_scripts\run_task_benchmark.bat 0 3 32

# With Telegram notifications
windows_scripts\run_with_notification.bat run_task_modified.bat 0 3 32
```

**Note**: Batch scripts contain many commented experiment configurations. Uncomment specific lines to run different whitening techniques or model combinations.

### Efficiency Analysis

The project includes an automated efficiency analysis workflow for comparing experiments across different data subset sizes (10%-100%).

#### Complete Pipeline (Recommended)

```batch
# Run everything: experiments + aggregation + visualization
windows_scripts\run_complete_efficiency_analysis.bat 0 3 32
```

#### Individual Steps

```batch
# 1. Run experiments only
windows_scripts\run_efficiency_analysis_auto.bat 0 3 32

# 2. Aggregate results
python aggregate_efficiency_results.py --verbose

# 3. Generate plots
python visualize_efficiency.py
```

#### Configuration

Edit [`efficiency_config.txt`](efficiency_config.txt) to enable/disable experiments:

```text
# Format: ENABLED|MODEL_NAME|DATASET|EXPERIMENT_NAME|LR|SEED|OTHER_ARGS
# Set ENABLED to 1 to run, 0 to skip

# BERT + ZCA Whitening
1|bilstm-dim-reduction|ag-news-bert-whitening-zca|ag-news-bert-whitening-zca-modified|1.8e-3|88|

# RoBERTa + ZCA Whitening
1|bilstm-dim-reduction|ag-news-roberta-whitening-zca|ag-news-roberta-whitening-zca-modified|1.8e-3|42|
```

#### Output Locations

```
efficiency_analysis/
├── raw_results/                    # Individual run results
├── all_efficiency_results_*.csv    # Full aggregated data
├── summary_by_experiment_*.csv     # Stats by experiment
├── summary_by_percentage_*.csv     # Stats by subset %
├── experiment_comparison_*.csv     # Direct comparison
└── plots/                          # All visualizations
    ├── time_vs_percentage_*.png
    ├── gpu_usage_*.png
    ├── time_breakdown_*.png
    ├── efficiency_scatter_*.png
    └── scaling_efficiency_*.png
```

### Preprocessing Methods

The project supports three text preprocessing methods with different speed/accuracy tradeoffs:

1. **Gensim** (default) - Fast, manual lemmatization
2. **NLTK** - Balanced speed/accuracy with WordNetLemmatizer
3. **Stanza** - Most accurate with full NLP pipeline (slowest)

#### Setting the Method

**Via Environment Variable (before running)**:

Windows:
```batch
set PREPROCESSING_METHOD=nltk
python main.py --experiment_name my_exp ...
```

Linux/Mac:
```bash
export PREPROCESSING_METHOD=nltk
python main.py --experiment_name my_exp ...
```

**In Python Code**:
```python
import utils.preprocessing as preprocessing

# Set method
preprocessing.set_preprocessing_method('stanza')

# Use preprocessing
cleaned = preprocessing.clean(["Sample text to preprocess"])
```

**Testing Methods**:
```bash
python test_preprocessing.py
```

---

## Configuration

### Available Models

- `bert-base-uncased`: Full BERT with classification head (frozen BERT)
- `roberta-base`: Full RoBERTa with classification head (frozen)
- `distilbert-base-uncased`: DistilBERT variant
- `albert-base-v2`: ALBERT variant
- `bilstm`: BiLSTM classifier for 768-dim features (extraction only)
- `bilstm-dim-reduction`: BiLSTM for 256-dim features (whitening)
- `mlp-dim-reduction`: MLP for 256-dim features (whitening)

### Available Datasets

**Baseline**:
- `ag-news-normal`: Standard BERT fine-tuning

**Feature Extraction (768-dim)**:
- `ag-news-bert-extraction`: BERT features only
- `ag-news-roberta-extraction`: RoBERTa features only

**Whitening (256-dim)**:
- `ag-news-bert-whitening-{technique}`: BERT + whitening
- `ag-news-roberta-whitening-{technique}`: RoBERTa + whitening

Where `{technique}` is one of: `zca`, `pca`, `svd` (recommended) or `zca-svd`, `pca-svd`, `eigen` (experimental)

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_epochs` | 10 | Number of training epochs |
| `--train_batch_size` | 4 | Training batch size |
| `--valid_batch_size` | 4 | Validation batch size |
| `--lr` | 6.25e-5 | Learning rate (use ~1-2e-3 for BiLSTM/MLP) |
| `--eps` | 6.25e-5 | Epsilon for optimizer |
| `--early_stop` | 3 | Early stopping patience |
| `--step_size` | 1 | LR scheduler step size |
| `--gamma` | 0.5 | LR scheduler decay factor |
| `--seed` | 42 | Random seed (use 88 for consistency) |
| `--max_seq_len` | 512 | Maximum sequence length |
| `--num_layers` | 12 | Number of layers |
| `--subset_percentage` | 100 | Percentage of data to use (1-100) |
| `--force` | False | Overwrite existing experiment directory |
| `--lower` | False | Lowercase input text |

---

## Telegram Notifications

Get automatic notifications when training tasks start, succeed, or fail.

### Quick Setup

1. Create a Telegram bot via [@BotFather](https://t.me/botfather)
2. Get your chat ID via [@userinfobot](https://t.me/userinfobot)
3. Copy [`.env.example`](.env.example) to `.env` and fill in your credentials
4. Use `run_with_notification.bat` wrapper to run any script with notifications

### Usage

```batch
# Standard training with notifications
run_with_notification.bat run_task_modified.bat 0 3 32

# K-fold training with notifications
run_with_notification.bat run_task_modified_kfold.bat 0 3 32

# Efficiency analysis with notifications
run_with_notification.bat run_complete_efficiency_analysis.bat 0 3 32
```

### Notification Format

Notifications include:
- **Status** (with emoji): 🚀 STARTED, ✅ SUCCESS, ❌ FAILED
- **Task name**: The batch file or task being run
- **Experiment name**: The experiment being trained (if provided)
- **Machine**: The hostname of the machine running the task
- **Timestamp**: When the notification was sent
- **Details**: Additional information about the task

Example notification:
```
✅ SUCCESS
━━━━━━━━━━━━━━━━━━━━
Task: run_task_modified.bat
Experiment: ag-news-bert-whitening-zca-bilstm
Machine: DESKTOP-ABC123
Time: 2025-11-12 14:30:45

Details:
Task completed successfully
```

**See [`TELEGRAM_SETUP.md`](TELEGRAM_SETUP.md) for detailed setup instructions.**

---

## Epsilon Parameter

The epsilon parameter controls numerical stability in whitening transformations.

### Default Value

```python
BertWhiteningDataset.EPSILON = 1e-5  # Class-level default
```

### Usage Options

**Option 1: Use Default Epsilon (1e-5)**
```python
dataset = BertWhiteningDataset(
    device=device,
    dataset=data,
    tokenizer=tokenizer,
    model=model,
    max_len=128,
    dim_technique='zca'
)
```

**Option 2: Specify Custom Epsilon**
```python
dataset = BertWhiteningDataset(
    device=device,
    dataset=data,
    tokenizer=tokenizer,
    model=model,
    max_len=128,
    dim_technique='zca',
    epsilon=1e-6  # Custom value
)
```

**Option 3: Change Class Default Globally**
```python
BertWhiteningDataset.EPSILON = 1e-6
dataset = BertWhiteningDataset(...)
```

### Recommended Values

- **Default (1e-5)**: Good balance for most use cases
- **More stability (1e-4 or 1e-3)**: Use if you encounter numerical issues
- **Less regularization (1e-6 or 1e-7)**: Use if whitening is too conservative

### Impact on Results

- **Larger epsilon**: More regularization, less aggressive whitening, more stable
- **Smaller epsilon**: Less regularization, more aggressive whitening, may be numerically unstable

**See [`EPSILON_USAGE.md`](EPSILON_USAGE.md) for detailed documentation.**

---

## Results & Output

### Training Outputs

Training saves to `save/{dataset}/{experiment_name}/`:
- `best_model_{id}.th`: Best model checkpoint
- `prediction_result.csv`: Test set predictions
- `evaluation_result.csv`: Performance metrics (accuracy, F1 score)
- `summary_efficiency.csv`: Efficiency metrics (time, GPU memory)
- `vocab.txt` and `config.json`: Model metadata

### Evaluation Metrics

- **Primary**: F1 score (macro average)
- **Secondary**: Accuracy
- **Efficiency**: Training time, GPU memory usage

---

## Key Implementation Details

### Whitening Implementation

The [`BertWhiteningDataset`](data_utils/ag_news/whitening.py) class performs whitening transformations:
- Extracts embeddings using frozen BERT/RoBERTa during initialization
- Uses `first_last_avg` pooling: averages first and last hidden states
- All techniques center data by subtracting mean
- Transformations compute kernel and bias matrices specific to each technique
- Final embeddings are normalized (L2 normalization via `_normalize()`)
- Target dimensionality reduced from 768 → 256 via kernel slicing: `kernel[:, :target_dim]`
- Numerical stability controlled by `EPSILON = 1e-5` (or configurable via constructor)

### Data Loading Patterns

Three dataloader patterns exist:

1. **Normal** ([`BertNormalDataLoader`](data_utils/ag_news/normal.py)): 
   - On-the-fly tokenization
   - Uses `max_seq_len` for truncation
   - Requires tokenizer

2. **Extraction** ([`BertExtractionDataLoader`](data_utils/ag_news/extraction.py)): 
   - Similar to Normal but for extraction-only mode (no whitening)

3. **Whitening** ([`BertWhiteningDataLoader`](data_utils/ag_news/whitening.py)): 
   - Pre-processes all data during dataset initialization
   - Stores pre-computed whitened embeddings
   - No tokenizer needed during training

**Critical for setup_dataloaders**:
- Checks for `'extract_model'` key in args to determine which pattern to use
- Whitening mode loads extraction model first, then creates dataset with pre-computed embeddings
- Supports subset sampling via `get_subset_data()` controlled by `--subset_percentage`

### Forward Functions

Located in [`utils/forward_fn.py`](utils/forward_fn.py):
- `forward_word_classification`: For BERT/RoBERTa end-to-end training
- `modified_forward_word_classification`: For pre-extracted features with lightweight classifiers

### Training Loop

The training loop ([`main.py`](main.py) and [`kfold_analysis.py`](kfold_analysis.py)) includes:
- **Efficiency tracking**: `efficiency_metrics_wrapper()` decorator measures time and GPU memory usage
- **Early stopping**: Based on validation F1 score (configurable via `--early_stop`)
- **Learning rate scheduling**: StepLR with configurable `--step_size` and `--gamma`
- **Gradient clipping**: max_norm=10.0 to prevent exploding gradients
- **Deterministic CUDA**: `torch.backends.cudnn.deterministic = True` for reproducibility
- **Optimizer**: AdamW with configurable learning rate (`--lr`) and epsilon (`--eps`)

**K-fold differences**:
- [`kfold_analysis.py`](kfold_analysis.py) merges train and validation sets, then splits into 5 folds
- Uses `setup_kfold_dataloaders()` generator that yields fold-specific data loaders
- Tracks metrics across folds: train/val accuracy, train/val loss, F1 score, GPU usage

---

## Development Notes

- **Reproducibility**: Fixed seed (default 88, configurable via `--seed`) with deterministic CUDA operations
- **Subset sampling**: `--subset_percentage` (1-100) controls data usage for faster experimentation
- **Optimizer**: AdamW with typical learning rates: 1e-3 to 2e-3 for lightweight classifiers, 1e-5 to 1e-4 for full BERT
- **Data limitation**: The whitening dataset may limit to first 1000 samples (see [`whitening.py:59`](data_utils/ag_news/whitening.py:59) - commented line)
- **Batch scripts**: Contain many commented experiments - uncomment specific lines to run different configurations
- **Force flag**: Use `--force` to overwrite existing experiment directories
- **Lower flag**: Use `--lower` to lowercase input text
- **Preprocessing methods**: Choose between 'gensim' (default, fast), 'nltk' (balanced), or 'stanza' (accurate) via `PREPROCESSING_METHOD` environment variable

---

## Troubleshooting

### Common Issues

**CUDA out of memory**:
- Reduce batch size: `--train_batch_size 16` or `--train_batch_size 8`
- Use CPU instead: Don't set `CUDA_VISIBLE_DEVICES`

**ImportError for NLTK or Stanza**:
- Make sure you downloaded the required data (see Installation Step 4)
- For NLTK: `python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"`
- For Stanza: `python -c "import stanza; stanza.download('en')"`

**Slow preprocessing**:
- Switch to faster method: `set PREPROCESSING_METHOD=gensim`
- Use subset for testing: `--subset_percentage 10`

**Telegram notifications not working**:
- Check `.env` file exists and has correct credentials
- See [`TELEGRAM_SETUP.md`](TELEGRAM_SETUP.md) for setup instructions

**No results found during efficiency aggregation**:
- Ensure experiments have completed successfully
- Check that `summary_efficiency_*.csv` files exist in `save/` directories
- Verify centralized directory: `efficiency_analysis/raw_results/`

**Configuration file not found**:
- Ensure `efficiency_config.txt` is in the project root
- Check file path in batch script (`set CONFIG_FILE=efficiency_config.txt`)

**Plots not generating**:
- Install required packages: `pip install matplotlib seaborn pandas`
- Ensure aggregated results exist before running visualization
- Check for errors in console output

---

## Tips and Best Practices

### Efficiency Analysis

1. **Start Small**: Enable only 1-2 experiments initially to verify the setup
2. **Use Subset Percentages Strategically**: For quick testing, modify the `PERCENTAGES` variable in the batch script to use fewer percentages (e.g., `10 50 100`)
3. **Monitor Progress**: Watch console output for errors and progress updates
4. **Review Aggregated Results**: Check the aggregated CSV files before creating visualizations
5. **Customize Visualizations**: Modify [`visualize_efficiency.py`](visualize_efficiency.py) to change plot styles, add new chart types, or adjust figure sizes

### General Development

1. **Use version control**: Commit configuration changes (e.g., `efficiency_config.txt`)
2. **Document experiments**: Keep track of which configurations work best
3. **Test incrementally**: Start with small subsets before running full experiments
4. **Monitor GPU usage**: Use efficiency metrics to optimize batch sizes and model configurations
5. **Save configurations**: Document enabled experiments in config file comments

---

## Additional Documentation

- [`README.md`](README.md) - Main project documentation
- [`CLAUDE.md`](CLAUDE.md) - Guidance for Claude Code
- [`CLEAN_ENVIRONMENT.md`](CLEAN_ENVIRONMENT.md) - Environment cleanup instructions
- [`EFFICIENCY_ANALYSIS_GUIDE.md`](EFFICIENCY_ANALYSIS_GUIDE.md) - Detailed efficiency analysis documentation
- [`EFFICIENCY_QUICK_REFERENCE.md`](EFFICIENCY_QUICK_REFERENCE.md) - Quick reference for efficiency analysis
- [`EPSILON_USAGE.md`](EPSILON_USAGE.md) - Epsilon parameter documentation
- [`TELEGRAM_SETUP.md`](TELEGRAM_SETUP.md) - Telegram notification setup guide

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{lc-bert,
  title = {LC-BERT: Lightweight Classification with BERT Embeddings},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/LC-BERT}
}
```

---

## Acknowledgments

- BERT and RoBERTa models from [Hugging Face Transformers](https://huggingface.co/transformers/)
- AG News dataset from [PyTorch Text](https://pytorch.org/text/)
- Whitening techniques inspired by [BERT-whitening](https://github.com/bojone/BERT-whitening)

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**End of Context Memory Document**
