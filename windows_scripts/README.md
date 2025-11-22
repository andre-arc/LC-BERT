# Windows Scripts

This directory contains Windows batch scripts for running LC-BERT training experiments.

## Main Training Scripts

### `run_task_modified.bat`
Main training script for whitening experiments with multiple model combinations.

**Usage:**
```cmd
run_task_modified.bat <GPU_ID> <EARLY_STOP_PATIENCE> <BATCH_SIZE>
```

**Example:**
```cmd
run_task_modified.bat 0 3 32
```

**Parameters:**
- `GPU_ID`: CUDA device ID (e.g., 0, 1, 2)
- `EARLY_STOP_PATIENCE`: Number of epochs to wait before early stopping
- `BATCH_SIZE`: Training batch size

**Contains experiments for:**
- BERT/RoBERTa extraction (no whitening)
- ZCA, ZCA-SVD whitening
- PCA, PCA-SVD whitening
- SVD whitening
- Eigen whitening
- Both BiLSTM and MLP classifiers

### `run_task_benchmark.bat`
Baseline benchmark experiments using standard BERT/RoBERTa/DistilBERT models.

**Usage:**
```cmd
run_task_benchmark.bat <GPU_ID> <EARLY_STOP_PATIENCE> <BATCH_SIZE>
```

**Contains experiments for:**
- BERT, RoBERTa, DistilBERT baselines
- Adam and AdamW optimizers

### `run_task_modified_kfold.bat`
K-fold cross-validation training (5-fold by default).

**Usage:**
```cmd
run_task_modified_kfold.bat <GPU_ID> <EARLY_STOP_PATIENCE> <BATCH_SIZE>
```

### `run_efficiency_analysis.bat`
Experiments for analyzing model efficiency across different data subset sizes (10%-100%).

**Usage:**
```cmd
run_efficiency_analysis.bat <GPU_ID> <EARLY_STOP_PATIENCE> <BATCH_SIZE>
```

## Utility Scripts

### `run_with_notification.bat`
Wrapper script that adds Telegram notifications to any training script.

**Usage:**
```cmd
run_with_notification.bat <SCRIPT_NAME> <SCRIPT_ARGS...>
```

**Example:**
```cmd
run_with_notification.bat run_task_modified.bat 0 3 32
```

**Features:**
- Sends start/success/failure notifications to Telegram
- Detects interrupted runs and sends notifications
- Tracks running tasks using flag files

**Requirements:**
- Telegram bot configured (see `../TELEGRAM_SETUP.md`)
- `.env` file in project root with credentials

### `notify_command.bat`
Helper script for running individual commands with Telegram notifications.

**Usage:**
```cmd
call notify_command.bat "<EXPERIMENT_NAME>" <COMMAND> <ARGS...>
```

**Example:**
```cmd
call notify_command.bat "ZCA-BiLSTM" python main.py --experiment_name zca-test ...
```

### `telegram_config.bat`
Loads Telegram bot credentials from `.env` file in the parent directory.

**Called automatically by other scripts** - not meant to be run directly.

## Configuration

### Telegram Notifications Setup
1. Follow instructions in `../TELEGRAM_SETUP.md`
2. Create `.env` file in project root (one level up from this directory)
3. Scripts will automatically load credentials when run

### Experiment Configuration
Most scripts contain commented experiment configurations. To run specific experiments:
1. Open the desired batch script
2. Uncomment the experiment line(s) you want to run
3. Comment out any active experiments you don't want
4. Run the script with appropriate parameters

## Notes

- All scripts use `%~dp0` for relative path resolution, so they work from any directory
- Scripts expect to be run from the project root or can be called with full path
- CUDA device is set via `CUDA_VISIBLE_DEVICES` environment variable
- Failed experiments will exit with non-zero code for proper error handling
