# Efficiency Analysis Quick Reference

Quick reference for the automated efficiency analysis workflow.

## One-Line Commands

### Complete Pipeline (Recommended)
```batch
REM Run everything: experiments + aggregation + visualization
windows_scripts\run_complete_efficiency_analysis.bat 0 3 32
```

### Individual Steps
```batch
REM 1. Run experiments only
windows_scripts\run_efficiency_analysis_auto.bat 0 3 32

REM 2. Aggregate results
python aggregate_efficiency_results.py --verbose

REM 3. Generate plots
python visualize_efficiency.py
```

## Configuration

### Enable/Disable Experiments
Edit `efficiency_config.txt`:
```text
# Set ENABLED to 1 (run) or 0 (skip)
1|bilstm-dim-reduction|ag-news-bert-whitening-zca|bert-zca|1.8e-3|88|
0|bilstm-dim-reduction|ag-news-bert-whitening-pca|bert-pca|1e-3|88|
```

### Change Subset Percentages
Edit `windows_scripts\run_efficiency_analysis_auto.bat`, line 36:
```batch
REM Default: 10 20 30 40 50 60 70 80 90 100
set PERCENTAGES=10 50 100
```

## Output Locations

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

## Common Workflows

### Quick Test (Small Subset)
```text
# 1. Edit efficiency_config.txt
1|bilstm-dim-reduction|ag-news-bert-whitening-zca|test|1.8e-3|88|

# 2. Edit batch script PERCENTAGES to: 10 50
# 3. Run
windows_scripts\run_complete_efficiency_analysis.bat 0 3 32
```

### Full Comparison
```text
# 1. Enable multiple experiments in efficiency_config.txt
1|bilstm-dim-reduction|ag-news-bert-whitening-zca|bert-zca|1.8e-3|88|
1|bilstm-dim-reduction|ag-news-roberta-whitening-zca|roberta-zca|1.8e-3|42|

# 2. Run full pipeline
windows_scripts\run_complete_efficiency_analysis.bat 0 3 32

# 3. Check plots in efficiency_analysis/plots/
```

### Regenerate Plots Only
```batch
python visualize_efficiency.py
```

### Custom Aggregation
```batch
python aggregate_efficiency_results.py ^
  --base-dir save ^
  --output-dir my_analysis ^
  --experiments bert-zca roberta-zca ^
  --metric elapsed_time ^
  --verbose
```

## Arguments Reference

### Batch Scripts
```
Argument 1: GPU_ID (0, 1, 2, ...)
Argument 2: EARLY_STOP (patience, e.g., 3)
Argument 3: BATCH_SIZE (e.g., 32, 64)
```

### aggregate_efficiency_results.py
```
--base-dir DIR          Source directory (default: save)
--output-dir DIR        Output directory (default: efficiency_analysis)
--experiments E1 E2     Compare specific experiments
--metric METRIC         elapsed_time or gpu_used
--verbose               Print detailed stats
```

### visualize_efficiency.py
```
--input-dir DIR         Input directory (default: efficiency_analysis)
--output-dir DIR        Output for plots (default: efficiency_analysis/plots)
--input-file FILE       Use specific CSV file
```

## Troubleshooting

### No results found
```batch
REM Check if experiments completed
dir save\ag-news-bert-whitening-zca\*\summary_efficiency_*.csv

REM Check centralized location
dir efficiency_analysis\raw_results\*.csv
```

### Aggregation fails
```batch
REM Ensure results exist
python aggregate_efficiency_results.py --verbose

REM Check specific directory
python aggregate_efficiency_results.py --base-dir save --verbose
```

### Plots not generating
```batch
REM Install dependencies
pip install matplotlib seaborn pandas

REM Check aggregated data exists
dir efficiency_analysis\all_efficiency_results_*.csv

REM Try manual run
python visualize_efficiency.py
```

### Configuration not found
```batch
REM Ensure efficiency_config.txt is in project root
dir efficiency_config.txt

REM Check from correct directory
cd c:\Users\rizwa\OneDrive\Documents\Projects\LC-BERT
```

## With Telegram Notifications

```batch
REM Get notified when tasks complete
windows_scripts\run_with_notification.bat ^
  windows_scripts\run_complete_efficiency_analysis.bat 0 3 32
```

See [TELEGRAM_SETUP.md](TELEGRAM_SETUP.md) for setup.

## Tips

1. **Start small**: Test with 1-2 experiments and percentages (10, 50) first
2. **Monitor progress**: Watch console output for errors
3. **Check results incrementally**: Review aggregated CSV before plots
4. **Save configurations**: Document enabled experiments in config file comments
5. **Use version control**: Commit efficiency_config.txt changes

## See Also

- [EFFICIENCY_ANALYSIS_GUIDE.md](EFFICIENCY_ANALYSIS_GUIDE.md) - Detailed documentation
- [CLAUDE.md](CLAUDE.md) - Project overview
- [README.md](README.md) - Main documentation
