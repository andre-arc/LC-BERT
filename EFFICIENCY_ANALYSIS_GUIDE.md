# Efficiency Analysis Guide

This guide explains the improved efficiency analysis workflow for LC-BERT experiments.

## Overview

The efficiency analysis system has been redesigned to be more automated and organized:

- **Centralized Configuration**: Use `efficiency_config.txt` to enable/disable experiments
- **Automated Execution**: One command runs all enabled experiments across all subset percentages
- **Centralized Results**: Results are saved in both original locations and a centralized directory
- **Automated Aggregation**: Automatically collect and aggregate all results
- **Visualization**: Generate comprehensive plots to compare experiments

## Quick Start

### 1. Configure Experiments

Edit [efficiency_config.txt](efficiency_config.txt) to enable/disable experiments:

```text
# Format: ENABLED|MODEL_NAME|DATASET|EXPERIMENT_NAME|LR|SEED|OTHER_ARGS
# Set ENABLED to 1 to run, 0 to skip

# BERT + ZCA Whitening
1|bilstm-dim-reduction|ag-news-bert-whitening-zca|ag-news-bert-whitening-zca-modified|1.8e-3|88|

# RoBERTa + ZCA Whitening
1|bilstm-dim-reduction|ag-news-roberta-whitening-zca|ag-news-roberta-whitening-zca-modified|1.8e-3|42|
```

### 2. Run Efficiency Analysis

Execute the automated batch script:

```bash
# Usage: run_efficiency_analysis_auto.bat GPU_ID EARLY_STOP BATCH_SIZE
windows_scripts\run_efficiency_analysis_auto.bat 0 3 32
```

This will:
- Read the configuration file
- Run all enabled experiments for subset percentages: 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%
- Save results to both individual experiment directories and centralized location
- Automatically run aggregation at the end

### 3. View Results

After completion, results are available in:

```
efficiency_analysis/
├── raw_results/                    # Centralized raw results
│   └── *.csv                       # Individual experiment results
├── all_efficiency_results_*.csv    # Complete aggregated data
├── summary_by_experiment_*.csv     # Summary statistics by experiment
├── summary_by_percentage_*.csv     # Summary by subset percentage
├── experiment_comparison_*.csv     # Direct experiment comparison
└── plots/                          # Visualizations (after running visualize_efficiency.py)
    ├── time_vs_percentage_*.png
    ├── gpu_usage_*.png
    ├── time_breakdown_*.png
    ├── efficiency_scatter_*.png
    └── scaling_efficiency_*.png
```

### 4. Generate Visualizations

Create plots from aggregated results:

```bash
python visualize_efficiency.py
```

Or specify custom directories:

```bash
python visualize_efficiency.py --input-dir efficiency_analysis --output-dir my_plots
```

## Detailed Usage

### Configuration File Format

The [efficiency_config.txt](efficiency_config.txt) file uses a simple pipe-delimited format:

```
ENABLED|MODEL_NAME|DATASET|EXPERIMENT_NAME|LR|SEED|OTHER_ARGS
```

**Fields:**
- `ENABLED`: `1` to run, `0` to skip
- `MODEL_NAME`: Model architecture (e.g., `bilstm-dim-reduction`, `mlp-dim-reduction`)
- `DATASET`: Dataset configuration (e.g., `ag-news-bert-whitening-zca`)
- `EXPERIMENT_NAME`: Experiment identifier (used in save paths)
- `LR`: Learning rate (e.g., `1.8e-3`)
- `SEED`: Random seed (e.g., `88`, `42`)
- `OTHER_ARGS`: Reserved for future use

**Example configurations:**

```text
# Full BERT baseline
1|bert-base-uncased|ag-news-normal|bert_benchmark|1e-4|88|

# BERT + ZCA + BiLSTM
1|bilstm-dim-reduction|ag-news-bert-whitening-zca|bert-zca-bilstm|1.8e-3|88|

# BERT + ZCA + MLP
1|mlp-dim-reduction|ag-news-bert-whitening-zca|bert-zca-mlp|1.8e-3|88|

# RoBERTa + PCA + BiLSTM
1|bilstm-dim-reduction|ag-news-roberta-whitening-pca|roberta-pca-bilstm|1e-3|42|
```

### Aggregation Script

Manually run aggregation anytime:

```bash
# Basic aggregation
python aggregate_efficiency_results.py

# Verbose output with statistics
python aggregate_efficiency_results.py --verbose

# Custom directories
python aggregate_efficiency_results.py --base-dir save --output-dir my_analysis

# Compare specific experiments
python aggregate_efficiency_results.py --experiments bert-zca-bilstm roberta-zca-bilstm --metric elapsed_time
```

**Options:**
- `--base-dir`: Directory containing experiment results (default: `save`)
- `--output-dir`: Output directory for aggregated results (default: `efficiency_analysis`)
- `--experiments`: Specific experiments to compare (optional)
- `--metric`: Metric to compare (`elapsed_time` or `gpu_used`)
- `--verbose`: Print detailed summary statistics

### Visualization Script

Generate plots from aggregated data:

```bash
# Generate all plots from latest aggregation
python visualize_efficiency.py

# Use specific input file
python visualize_efficiency.py --input-file efficiency_analysis/all_efficiency_results_20250128_143022.csv

# Custom output directory
python visualize_efficiency.py --output-dir custom_plots
```

**Generated plots:**
1. **Time vs Percentage**: Training time across subset sizes
2. **GPU Usage**: Maximum GPU memory by experiment
3. **Time Breakdown**: Stacked bar chart of time by phase (Dataloader, Train, Test)
4. **Efficiency Scatter**: Time vs GPU usage scatter plot
5. **Scaling Efficiency**: Time per percentage point (shows scaling behavior)

## Directory Structure

```
LC-BERT/
├── efficiency_config.txt                      # Configuration file
├── efficient_analysis.py                      # Core efficiency analysis script
├── aggregate_efficiency_results.py            # Aggregation script
├── visualize_efficiency.py                    # Visualization script
├── windows_scripts/
│   ├── run_efficiency_analysis.bat            # Old manual script (deprecated)
│   └── run_efficiency_analysis_auto.bat       # New automated script
├── save/                                       # Original experiment results
│   └── {dataset}/{experiment}/
│       └── summary_efficiency_{percent}.csv
└── efficiency_analysis/                        # Centralized analysis directory
    ├── raw_results/                            # Centralized raw results
    ├── all_efficiency_results_*.csv           # Aggregated data
    ├── summary_*.csv                          # Summary statistics
    └── plots/                                  # Visualizations
```

## Workflow Comparison

### Old Workflow

1. Manually uncomment specific lines in `run_efficiency_analysis.bat`
2. Run batch script
3. Results scattered across `save/` directory
4. Manual collection and comparison needed

### New Workflow

1. Edit `efficiency_config.txt` (toggle experiments with 1/0)
2. Run `run_efficiency_analysis_auto.bat 0 3 32`
3. Results automatically:
   - Saved to original locations
   - Copied to centralized directory
   - Aggregated into summary reports
4. Generate plots with `visualize_efficiency.py`

## Integration with Telegram Notifications

The automated script works with Telegram notifications:

```bash
# Run with Telegram notifications
windows_scripts\run_with_notification.bat run_efficiency_analysis_auto.bat 0 3 32
```

See [TELEGRAM_SETUP.md](TELEGRAM_SETUP.md) for configuration.

## Tips and Best Practices

### 1. Start Small
Enable only 1-2 experiments initially to verify the setup:

```text
1|bilstm-dim-reduction|ag-news-bert-whitening-zca|test-run|1.8e-3|88|
0|... all others ...
```

### 2. Use Subset Percentages Strategically
For quick testing, modify the `PERCENTAGES` variable in the batch script:

```batch
REM Test with fewer percentages
set PERCENTAGES=10 50 100
```

### 3. Monitor Progress
The script displays progress for each experiment and percentage:

```
========================================
Running: ag-news-bert-whitening-zca-modified
========================================
[10%] Running efficiency analysis...
Successfully completed 10% subset
[20%] Running efficiency analysis...
```

### 4. Review Aggregated Results
Check the aggregated CSV files before creating visualizations:

```bash
# Quick look at experiment comparison
type efficiency_analysis\experiment_comparison_*.csv
```

### 5. Customize Visualizations
Modify [visualize_efficiency.py](visualize_efficiency.py) to:
- Change plot styles
- Add new chart types
- Filter specific experiments
- Adjust figure sizes

## Troubleshooting

### No results found during aggregation
- Ensure experiments have completed successfully
- Check that `summary_efficiency_*.csv` files exist in `save/` directories
- Verify centralized directory: `efficiency_analysis/raw_results/`

### Configuration file not found
- Ensure `efficiency_config.txt` is in the project root
- Check file path in batch script (`set CONFIG_FILE=efficiency_config.txt`)

### Plots not generating
- Install required packages: `pip install matplotlib seaborn pandas`
- Ensure aggregated results exist before running visualization
- Check for errors in console output

### Experiment fails partway through
- Individual experiment failures don't stop the entire run
- Check error messages in console
- Review experiment-specific logs in `save/` directories

## Advanced Usage

### Custom Aggregation Queries

The aggregated CSV files can be analyzed with pandas:

```python
import pandas as pd

# Load aggregated results
df = pd.read_csv('efficiency_analysis/all_efficiency_results_latest.csv')

# Custom analysis
avg_time_by_dataset = df.groupby('dataset')['elapsed_time'].mean()
gpu_by_model = df.groupby('experiment')['gpu_used'].max()

# Filter specific experiments
bert_experiments = df[df['dataset'].str.contains('bert')]
```

### Export for LaTeX/Papers

```python
# Generate LaTeX table
summary = pd.read_csv('efficiency_analysis/summary_by_experiment_latest.csv')
latex_table = summary.to_latex(index=False)
print(latex_table)
```

## See Also

- [CLAUDE.md](CLAUDE.md) - Project overview and common commands
- [TELEGRAM_SETUP.md](TELEGRAM_SETUP.md) - Telegram notification setup
- [README.md](README.md) - Main project documentation
