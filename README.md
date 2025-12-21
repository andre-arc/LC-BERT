# LC-BERT: Lightweight Classification with BERT Embeddings

A research project exploring dimensionality reduction techniques applied to BERT/RoBERTa embeddings for efficient text classification. This project compares various whitening transformations (ZCA, PCA, SVD) combined with lightweight classifiers (BiLSTM, MLP) against standard BERT-based approaches on the AG News dataset.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Detailed Installation](#detailed-installation)
- [Usage](#usage)
  - [Training Models](#training-models)
  - [Running Experiments](#running-experiments)
  - [Preprocessing Methods](#preprocessing-methods)
  - [Telegram Notifications](#telegram-notifications)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

LC-BERT implements a two-stage pipeline for text classification:

1. **Feature Extraction**: Extract embeddings from pretrained BERT/RoBERTa models and apply dimensionality reduction (768 → 256 dimensions) using various whitening techniques
2. **Classification**: Train lightweight classifiers (BiLSTM or MLP) on the reduced embeddings

This approach aims to reduce computational costs while maintaining competitive performance compared to full BERT fine-tuning.

## Key Features

- **Multiple Whitening Techniques**: ZCA, PCA, and SVD transformations for dimensionality reduction
- **Flexible Preprocessing**: Support for three preprocessing methods (Gensim, NLTK, Stanza) with different speed/accuracy tradeoffs
- **Model Variety**: Compare BERT, RoBERTa, DistilBERT, and ALBERT embeddings
- **Lightweight Classifiers**: BiLSTM and MLP architectures for efficient training
- **K-Fold Cross-Validation**: Built-in support for robust evaluation
- **Telegram Notifications**: Get notified when training tasks start, succeed, or fail
- **Comprehensive Logging**: Track efficiency metrics (GPU usage, training time)
- **Batch Processing**: Automated scripts for running multiple experiments
- **Efficiency Analysis**: Automated workflow for running experiments across different data subset sizes with visualization

## Installation

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

### Detailed Installation

#### Step 1: Prerequisites

- **Anaconda or Miniconda**: [Download here](https://docs.conda.io/en/latest/miniconda.html)
- **CUDA 11.3** (optional, for GPU support): [Download here](https://developer.nvidia.com/cuda-11.3.0-download-archive)
- **Git**: [Download here](https://git-scm.com/downloads)

#### Step 2: Environment Setup

The project provides two environment files:

1. **Full installation** (`environment.yml`): Includes all dependencies including Jupyter, visualization tools

**Full Installation:**
```bash
conda env create -f environment.yml
conda activate lc-bert
```

#### Step 3: Install PyTorch

PyTorch must be installed separately based on your system configuration:

**For GPU (CUDA 11.3):**
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

**For CPU only:**
```bash
pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0
```

**For other CUDA versions:**
Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) and select your configuration.

#### Step 4: Download NLP Data

**Required - NLTK data:**
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

**Optional - Stanza model (only if using advanced preprocessing):**
```bash
python -c "import stanza; stanza.download('en')"
```

#### Step 5: Verify Installation

```bash
python -c "import torch, transformers, numpy, pandas, sklearn; \
           print(f'PyTorch: {torch.__version__}'); \
           print(f'CUDA available: {torch.cuda.is_available()}'); \
           print(f'Transformers: {transformers.__version__}')"
```

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

### Running Experiments

#### Windows

The project includes batch scripts in the `windows_scripts/` directory:

```batch
REM Standard training run
windows_scripts\run_task_modified.bat 0 3 32

REM Arguments:
REM   0  - GPU ID (CUDA_VISIBLE_DEVICES)
REM   3  - Early stopping patience
REM   32 - Batch size

REM K-fold cross-validation
windows_scripts\run_task_modified_kfold.bat 0 3 32

REM Benchmark baselines
windows_scripts\run_task_benchmark.bat 0 3 32

REM With Telegram notifications
windows_scripts\run_with_notification.bat windows_scripts\run_task_modified.bat 0 3 32

REM Automated efficiency analysis (runs experiments across subset percentages)
windows_scripts\run_efficiency_analysis_auto.bat 0 3 32

REM Complete efficiency pipeline (experiments + aggregation + visualization)
windows_scripts\run_complete_efficiency_analysis.bat 0 3 32
```

**Note**: Batch scripts contain many commented experiment configurations. Uncomment specific lines to run different whitening techniques or model combinations.

#### Efficiency Analysis

The project includes an automated efficiency analysis workflow for comparing experiments across different data subset sizes (10%-100%):

```batch
REM 1. Configure which experiments to run
REM Edit efficiency_config.txt and set ENABLED=1 for desired experiments

REM 2. Run complete pipeline (experiments + aggregation + plots)
windows_scripts\run_complete_efficiency_analysis.bat 0 3 32

REM 3. Or run steps individually:

REM   a. Run experiments only
windows_scripts\run_efficiency_analysis_auto.bat 0 3 32

REM   b. Aggregate results
python aggregate_efficiency_results.py --verbose

REM   c. Generate visualizations
python visualize_efficiency.py
```

Results are automatically saved to:
- `efficiency_analysis/all_efficiency_results_*.csv` - Complete aggregated data
- `efficiency_analysis/summary_*.csv` - Summary statistics
- `efficiency_analysis/plots/` - Visualization charts

See [EFFICIENCY_ANALYSIS_GUIDE.md](EFFICIENCY_ANALYSIS_GUIDE.md) for detailed documentation.

#### Linux

```bash
# Standard training
linux_scripts/run_task_modified.sh 0 3 32

# K-fold cross-validation
linux_scripts/run_task_modified_kfold.sh 0 3 32

# Benchmark baselines
linux_scripts/run_task_benchmark.sh 0 3 32
```

### Preprocessing Methods

The project supports three text preprocessing methods with different speed/accuracy tradeoffs. See [PREPROCESSING_METHODS.md](PREPROCESSING_METHODS.md) for detailed documentation.

#### Available Methods

1. **Gensim** (default) - Fast, manual lemmatization
2. **NLTK** - Balanced speed/accuracy with WordNetLemmatizer
3. **Stanza** - Most accurate with full NLP pipeline (slowest)

#### Setting the Method

**Via Environment Variable (before running):**

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

**In Python Code:**
```python
import utils.preprocessing as preprocessing

# Set method
preprocessing.set_preprocessing_method('stanza')

# Use preprocessing
cleaned = preprocessing.clean(["Sample text to preprocess"])
```

**Testing Methods:**
```bash
python test_preprocessing.py
```

### Telegram Notifications

Get automatic notifications when training tasks start, succeed, or fail. See [TELEGRAM_SETUP.md](TELEGRAM_SETUP.md) for setup instructions.

**Quick Setup:**
1. Create a Telegram bot via [@BotFather](https://t.me/botfather)
2. Get your chat ID via [@userinfobot](https://t.me/userinfobot)
3. Copy `.env.example` to `.env` and fill in your credentials
4. Use `run_with_notification.bat` wrapper to run any script with notifications

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
│   └── telegram_config.bat         # Telegram configuration
├── linux_scripts/           # Linux shell scripts
│   ├── run_task_modified.sh
│   ├── run_task_benchmark.sh
│   └── telegram_config.sh
├── main.py                  # Main training script
├── kfold_analysis.py        # K-fold cross-validation script
├── train_multiple_seeds.py  # Multi-seed training
├── telegram_notifier.py     # Telegram notification utility
├── test_preprocessing.py    # Test preprocessing methods
├── environment.yml          # Conda environment (full)
├── requirements.txt         # Pip requirements
├── .env.example             # Example Telegram config
├── README.md                # This file
├── PREPROCESSING_METHODS.md # Preprocessing guide
├── TELEGRAM_SETUP.md        # Telegram setup guide
├── EPSILON_USAGE.md         # Epsilon parameter documentation
└── CLEAN_ENVIRONMENT.md     # Environment cleanup guide
```

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

**Baseline:**
- `ag-news-normal`: Standard BERT fine-tuning

**Feature Extraction (768-dim):**
- `ag-news-bert-extraction`: BERT features only
- `ag-news-roberta-extraction`: RoBERTa features only

**Whitening (256-dim):**
- `ag-news-bert-whitening-{technique}`: BERT + whitening
- `ag-news-roberta-whitening-{technique}`: RoBERTa + whitening

Where `{technique}` is one of: `zca`, `pca`, `svd`

<!-- Experimental techniques (not recommended for production use):
- `zca-svd`, `pca-svd`, `eigen` - Still under development
-->

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

## Results

Training outputs are saved to `save/{dataset}/{experiment_name}/`:
- `best_model_{id}.th`: Best model checkpoint
- `prediction_result.csv`: Test set predictions
- `evaluation_result.csv`: Performance metrics (accuracy, F1 score)
- `summary_efficiency.csv`: Efficiency metrics (time, GPU memory)
- `vocab.txt` and `config.json`: Model metadata

### Evaluation Metrics

- **Primary**: F1 score (macro average)
- **Secondary**: Accuracy
- **Efficiency**: Training time, GPU memory usage

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

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

## Acknowledgments

- BERT and RoBERTa models from [Hugging Face Transformers](https://huggingface.co/transformers/)
- AG News dataset from [PyTorch Text](https://pytorch.org/text/)
- Whitening techniques inspired by [BERT-whitening](https://github.com/bojone/BERT-whitening)

## Additional Documentation

- [PREPROCESSING_METHODS.md](PREPROCESSING_METHODS.md) - Detailed preprocessing documentation
- [TELEGRAM_SETUP.md](TELEGRAM_SETUP.md) - Telegram notification setup
- [EPSILON_USAGE.md](EPSILON_USAGE.md) - Epsilon parameter explanation
- [CLEAN_ENVIRONMENT.md](CLEAN_ENVIRONMENT.md) - Environment cleanup instructions

## Troubleshooting

### Common Issues

**CUDA out of memory:**
- Reduce batch size: `--train_batch_size 16` or `--train_batch_size 8`
- Use CPU instead: Don't set `CUDA_VISIBLE_DEVICES`

**ImportError for NLTK or Stanza:**
- Make sure you downloaded the required data (see Installation Step 4)
- For NLTK: `python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"`
- For Stanza: `python -c "import stanza; stanza.download('en')"`

**Slow preprocessing:**
- Switch to faster method: `set PREPROCESSING_METHOD=gensim`
- Use subset for testing: `--subset_percentage 10`

**Telegram notifications not working:**
- Check `.env` file exists and has correct credentials
- See [TELEGRAM_SETUP.md](TELEGRAM_SETUP.md) for setup instructions

### Getting Help

- Check existing documentation in the repository
- Open an issue on GitHub with error logs and system information

---

**Happy experimenting!** 🚀
