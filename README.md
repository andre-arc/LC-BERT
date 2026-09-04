# LC-BERT: Lightweight Classification with BERT Embeddings

A research project exploring dimensionality reduction techniques applied to BERT/RoBERTa embeddings for efficient text classification. This project compares various whitening transformations (ZCA, PCA, SVD) combined with lightweight classifiers (BiLSTM, MLP) against standard BERT-based approaches on the AG News dataset.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Detailed Installation](#detailed-installation)
  - [Minimal Installation](#minimal-installation)
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

This approach aims to reduce computational costs while maintaining competitive performance.

**Note on the encoder:** the transformer backbone is **frozen** in every scenario — both the
baselines and the whitening pipeline. Only the classification head is trained. This was a
hardware necessity (full fine-tuning of BERT-base needs ~8–12 GB; the original experiments ran
on a 4 GB GPU), and applying it uniformly is what makes the comparison isolate the whitening
and classifier choices. Nothing here fine-tunes BERT.

## Key Features

- **Multiple Whitening Techniques**: ZCA, PCA, and SVD transformations for dimensionality reduction
- **Flexible Preprocessing**: Support for three preprocessing methods (Gensim, NLTK, Stanza) with different speed/accuracy tradeoffs
- **Model Variety**: Compare BERT, RoBERTa, DistilBERT, and ALBERT embeddings
- **Lightweight Classifiers**: BiLSTM and MLP architectures for efficient training
- **K-Fold Cross-Validation**: Built-in support for robust evaluation
- **Telegram Notifications**: Get notified when training tasks start, succeed, or fail
- **Comprehensive Logging**: Track efficiency metrics (GPU usage, training time)
- **Batch Processing**: Automated scripts for running multiple experiments

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

# 5. (Optional) NLTK data - only needed for PREPROCESSING_METHOD=nltk.
#     Text preprocessing is OFF in the training path by default, so you can skip this.
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

```bash
conda env create -f environment.yml
conda activate lc-bert
```

`environment.yml` pins the whole stack (python 3.10, numpy 1.22.4, pandas 1.4.4, scipy 1.8.0,
scikit-learn 1.0.2, gensim 4.2.0) plus transformers 4.25.1 via pip. Three details in it are
load-bearing, so don't "clean them up":

- **`nodefaults`** in the channel list. conda ≥ 25 refuses to solve against the Anaconda
  `repo.anaconda.com` channels until their Terms of Service are accepted; every package here is
  on conda-forge, so the ToS prompt is avoided entirely. If conda still complains, an
  installer-level `.condarc` is forcing `defaults` — point it at conda-forge instead.
- **`libblas=*=*openblas`**. Left unpinned, conda-forge resolves LAPACK to an MKL 2024 build
  whose exports numpy 1.22.4 cannot load: `np.linalg.eigh` / `svd` then kill the interpreter
  with `0xC06D007F` and **no traceback**, which breaks every whitening run at kernel computation.
- **No `pathlib` pip entry.** That PyPI package is the Python-2 backport of a stdlib module
  (stdlib since 3.4); it fails to build with modern setuptools, and its failure aborts the whole
  `conda env create` and rolls the environment back.

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

#### Step 4: Download NLP Data (optional)

Text preprocessing is **disabled in the training path** (`apply_cleaning=False` in all three
dataset classes), so neither download is needed for a normal run. Get them only if you set
`PREPROCESSING_METHOD` and enable cleaning yourself:

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
python -c "import stanza; stanza.download('en')"   # only for PREPROCESSING_METHOD=stanza
```

#### Step 5: Verify Installation

```bash
python -c "import torch, transformers, numpy, pandas, sklearn; \
           print(f'PyTorch: {torch.__version__}'); \
           print(f'CUDA available: {torch.cuda.is_available()}'); \
           print(f'Transformers: {transformers.__version__}')"

# Whitening kernels: numpy only, no GPU, runs in seconds
python test_whitening_validation.py
```

#### Step 6: Check for a shadowing user site-packages (Windows especially)

Python puts the per-user site-packages directory **ahead** of the conda environment on
`sys.path`, so a stray `%APPDATA%\Python\Python310\site-packages` silently overrides the pinned
versions — `conda list` shows numpy 1.22.4 while the process actually imports 1.23.5. Check:

```bash
python -c "import numpy; print(numpy.__file__)"      # must be inside envs/lc-bert
```

If it points elsewhere, scope the fix to this environment only:

```bash
conda env config vars set -n lc-bert PYTHONNOUSERSITE=1
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
```

**Note**: Batch scripts contain many commented experiment configurations. Uncomment specific lines to run different whitening techniques or model combinations.

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

The project supports three text preprocessing methods with different speed/accuracy tradeoffs.

**They are dormant by default.** All three dataset classes call
`load_dataset(..., apply_cleaning=False)`, so `cleaned_text` is the raw `Description` column and
`utils/preprocessing.clean` never runs during training. The setting below only affects code that
calls `clean` directly, such as `test_preprocessing.py`.

That default is deliberate: preprocessing measurably *hurts* BERT features here. A linear probe
scores 0.8951 on uncleaned whitened embeddings versus 0.8821 on cleaned ones, and 0.9032 versus
0.8900 on the raw 768-dim vectors — a consistent −0.013. Stopword, punctuation and inflection
removal strips exactly the signal BERT was pretrained on, and WordPiece already handles
morphology, so lemmatisation is redundant.

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
│       ├── normal.py        # Subword tokenizer path (frozen backbone)
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
├── test_whitening_validation.py # Whitening kernel checks (numpy only)
├── environment.yml          # Conda environment
├── requirements.txt         # Pip requirements
├── .env.example             # Example Telegram config
├── README.md                # This file
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
- `ag-news-normal`: subword tokenizer path — frozen backbone plus a trainable head

**Feature Extraction (768-dim):**
- `ag-news-bert-extraction`: BERT features only
- `ag-news-roberta-extraction`: RoBERTa features only

**Whitening (256-dim):**
- `ag-news-bert-whitening-{technique}`: BERT + whitening
- `ag-news-roberta-whitening-{technique}`: RoBERTa + whitening

Where `{technique}` is one of:

| technique | kernel (row-vector convention, `z = (x - μ)·W`) | notes |
|---|---|---|
| `svd` | `U·S^(-1/2)` | BERT-Whitening (Su et al. 2021) |
| `pca` | `V·Λ^(-1/2)` | **identical to `svd`** — same transform, two decompositions |
| `pca-cor` | `V^(-1/2)·G·Θ^(-1/2)` | PCA-cor (Kessy et al. 2018 Eq. 12), scale-invariant |
| `zca` | `V·Λ^(-1/2)·Vᵀ` | symmetric; measurably worse under truncation, see Results |
| `pca-svd`, `zca-svd` | as above, via SVD | numerically equivalent alternatives |
| `eigen` | `U·Λ^(-1/2)` | equivalent to `svd`; kept for backwards compatibility |

Truncation keeps the first `target_dim` **columns** (256, hardcoded in
`BertWhiteningDataset.Dim_reduction`). That selects the top-variance components for the
PCA-type kernels, which is why they beat `zca` — the symmetric ZCA kernel's columns are the
original embedding dimensions, so slicing it keeps dims 0–255 rather than the informative ones.

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
| `--lower` | False | Lowercase input text (**no effect** on the whitening/extraction path — the dataset accepts it into `**kwargs` and never uses it) |
| `--per_split_whitening` | False | Fit a separate whitening kernel on each split. Default is to fit on train only and reuse it for valid/test, so the classifier sees every split in one basis |

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

### Current results

AG News test set, BERT features, Bi-LSTM head, full data, no text preprocessing, kernel fitted
on train only. Multi-seed is `train_multiple_seeds.py` over seeds [42, 88, 456].

| Configuration | dims | F1 (mean ± std) |
|---|---|---|
| `ag-news-bert-whitening-svd` | 256 | **0.9130 ± 0.0008** |
| `ag-news-bert-whitening-pca-cor` | 256 | **0.9129 ± 0.0012** |
| `ag-news-bert-whitening-zca` | 256 | 0.8111 ± 0.0022 |
| `ag-news-bert-extraction` (no reduction) | 768 | 0.9002 (single seed) |

`svd` and `pca-cor` differ by 0.0001 — inside one standard deviation, so they are the same
method in practice. `zca` is genuinely worse. Whitening to 256 dims beats the unreduced 768-dim
features, at roughly half the peak GPU memory of a frozen-BERT-plus-head baseline.

### Relationship to the published paper

The IEEE COMNETSAT 2025 paper reports different numbers (best F1 0.83). Those are reproducible
— check out commit `c42c5a7` and run it unmodified — but they predate three fixes in the
current code, and the differences are not small:

1. **The Bi-LSTM head applied `nn.Softmax` before `CrossEntropyLoss`** (softmax twice, capping
   the loss at 0.743) and **`nn.ReLU` to the logits** (all-negative initialisation produced an
   all-zero output and zero gradient). Results swung between 0.14 and 0.83 F1 on initialisation
   state alone. `MLPForWordClassification` was never affected, so the paper's Bi-LSTM vs MLP
   comparison is confounded.
2. **Text preprocessing was always on.** It costs ~0.013 F1 on the features themselves — it
   strips the stopwords, punctuation and inflection BERT was pretrained on — but was worth
   +0.607 to the broken head. It is off by default now; leave it off.
3. **The whitening kernels were fixed** (orientation, eigenvalue ordering, two variants that
   raised `IndexError` and could never run). See `test_whitening_validation.py`.

The paper's central claim survives and strengthens: whitening-based reduction to 256 dims gives
better accuracy than the unreduced features at a third of the size. What changes is the method
ranking — PCA-type transforms beat ZCA, the reverse of the published order.

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

- [TELEGRAM_SETUP.md](TELEGRAM_SETUP.md) - Telegram notification setup
- [EPSILON_USAGE.md](EPSILON_USAGE.md) - Epsilon parameter explanation. **Stale:** it documents
  the whitening epsilon as `1e-5`; the value in `BertWhiteningDataset.EPSILON` is `1e-8`, which
  is also what the paper reports. Note `--eps` is a different quantity — the AdamW epsilon.
- [CLEAN_ENVIRONMENT.md](CLEAN_ENVIRONMENT.md) - Environment cleanup instructions

## Troubleshooting

### Common Issues

**Python dies with no traceback during whitening (exit code `0xC06D007F` / `-1066598273`):**
- A LAPACK mismatch — `np.linalg.eigh` / `svd` cannot load their MKL exports. Buffered stdout is
  lost on the crash, so it looks like a silent hang.
- Fix: `conda install -n lc-bert "libblas=*=*openblas"`. Already pinned in `environment.yml`.

**`conda env create` fails with `CondaToSNonInteractiveError`:**
- conda ≥ 25 blocks on the Anaconda channel Terms of Service. `environment.yml` uses
  `nodefaults` to avoid them, but an installer-level `.condarc` can still force `defaults` —
  check `conda config --show-sources` and point it at conda-forge.

**Results don't match the pinned versions / behave inconsistently:**
- A user site-packages directory may be shadowing the environment. Verify with
  `python -c "import numpy; print(numpy.__file__)"` and see Installation Step 6.

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
- Run `python test_whitening_validation.py` if whitening results look wrong — it checks the
  kernels against `Wᵀ Σ W = I` in the pipeline's own row-vector convention, in seconds
- Open an issue on GitHub with error logs and system information

---

**Happy experimenting!** 🚀
