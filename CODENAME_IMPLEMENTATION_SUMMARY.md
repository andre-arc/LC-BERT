# Codename Implementation Summary

## Overview
This document summarizes the modifications made to [`comparison_performance_analysis.py`](comparison_performance_analysis.py:1) to implement codenames (A01, A02, A03, B01-B06, C01-C06) from the 15 models implementation plan.

## Changes Made

### 1. Added Codename Mappings (Lines 46-88)
Added two new dictionaries to map model configurations to codenames:

- **`CODENAME_MAPPING`**: Maps (base_model, preprocessing, classifier) tuples to codenames
  - Skenario 1 (Benchmark): A01, A02, A03
  - Skenario 2 (BiLSTM): B01-B06
  - Skenario 3 (MLP): C01-C06

- **`CODENAME_DESCRIPTIONS`**: Provides human-readable descriptions for each codename

### 2. Added `get_codename()` Function (Lines 90-92)
New helper function that:
- Takes base_model, preprocessing, and classifier as input
- Returns the corresponding codename from `CODENAME_MAPPING`
- Returns 'N/A' if no match is found

### 3. Modified `parse_model_name()` Function (Lines 94-122)
Updated to:
- Call `get_codename()` to determine the codename
- Include 'codename' in the returned dictionary

### 4. Updated `load_model_results()` Function (Line 163)
Modified the result dictionary to include:
- `'codename': model_info['codename']`

### 5. Enhanced `create_performance_comparison_table()` Function (Line 248)
Added 'Codename' column to the comparison table, appearing as the first column for easy reference.

### 6. Updated Visualization Functions

#### `create_accuracy_comparison_plot()` (Line 296)
- Modified y-axis labels to show: `"codename: model_name[:20]"`
- Provides clear identification of models in accuracy comparison

#### `create_efficiency_comparison_plot()` (Lines 365, 378, 389)
Updated all three efficiency plots:
- Training Time Comparison
- Test Time Comparison
- GPU Usage Comparison
- All now display: `"codename: model_name[:20]"` in y-axis labels

#### `create_model_ranking_dashboard()` (Lines 492, 565)
Updated two ranking visualizations:
- Top 10 Models by Accuracy
- Top 10 Models by F1 Score
- Both now show: `"codename: model_name[:15]"` in y-axis labels

### 7. Enhanced `create_detailed_analysis_report()` Function (Lines 650, 752)
Updated report sections:
- Top 10 Models by Accuracy: Shows `"[codename] model_name"`
- Top 5 Most Efficient Models: Shows `"[codename] model_name"`

### 8. Added `create_codename_reference_table()` Function (Lines 283-322)
New function that:
- Creates a comprehensive reference table mapping codenames to model configurations
- Includes: Codename, Description, Model Name, Base Model, Preprocessing, Classifier
- Saves to both CSV and TXT formats
- Filters out models with 'N/A' codename

### 9. Updated `main()` Function (Line 836)
Added call to `create_codename_reference_table(df)` in the analysis pipeline.

## Codename Mapping

### Skenario 1: Benchmark (Baseline Models)
| Codename | Base Model | Preprocessing | Classifier | Description |
|----------|-------------|---------------|-------------|-------------|
| A01 | bert | none | benchmark | BERT Benchmark |
| A02 | roberta | none | benchmark | RoBERTa Benchmark |
| A03 | distilbert | none | benchmark | DistilBERT Benchmark |

### Skenario 2: Modified Models with BiLSTM Classifier
| Codename | Base Model | Preprocessing | Classifier | Description |
|----------|-------------|---------------|-------------|-------------|
| B01 | bert | svd | bilstm | BERT + SVD Whitening + BiLSTM |
| B02 | bert | pca | bilstm | BERT + PCA Whitening + BiLSTM |
| B03 | bert | zca | bilstm | BERT + ZCA Whitening + BiLSTM |
| B04 | roberta | svd | bilstm | RoBERTa + SVD Whitening + BiLSTM |
| B05 | roberta | pca | bilstm | RoBERTa + PCA Whitening + BiLSTM |
| B06 | roberta | zca | bilstm | RoBERTa + ZCA Whitening + BiLSTM |

### Skenario 3: Modified Models with MLP Classifier
| Codename | Base Model | Preprocessing | Classifier | Description |
|----------|-------------|---------------|-------------|-------------|
| C01 | bert | svd | mlp | BERT + SVD Whitening + MLP |
| C02 | bert | pca | mlp | BERT + PCA Whitening + MLP |
| C03 | bert | zca | mlp | BERT + ZCA Whitening + MLP |
| C04 | roberta | svd | mlp | RoBERTa + SVD Whitening + MLP |
| C05 | roberta | pca | mlp | RoBERTa + PCA Whitening + MLP |
| C06 | roberta | zca | mlp | RoBERTa + ZCA Whitening + MLP |

## Output Files

The modified script now generates the following additional files:

1. **`performance_analysis/codename_reference_table.csv`** - CSV format codename reference
2. **`performance_analysis/codename_reference_table.txt`** - Text format codename reference

All existing output files now include codenames:
- `performance_comparison_table.csv/txt` - Added Codename column
- `accuracy_comparison.png` - Codenames in plot labels
- `efficiency_comparison.png` - Codenames in plot labels
- `model_ranking_dashboard.png` - Codenames in plot labels
- `detailed_analysis_report.txt` - Codenames in model listings

## Usage

Run the modified script as before:

```bash
python comparison_performance_analysis.py
```

The script will automatically:
1. Parse model directory names
2. Assign codenames based on model configuration
3. Generate all analysis outputs with codenames included
4. Create a dedicated codename reference table

## Benefits

1. **Easy Reference**: Codenames provide a quick way to identify models
2. **Consistent Naming**: Aligns with the 15 models implementation plan
3. **Better Organization**: Models are grouped by scenario (A, B, C series)
4. **Clear Documentation**: Reference table provides complete mapping
5. **Enhanced Visualizations**: Plots show both codename and model name

## Notes

- Models that don't match any codename mapping will display 'N/A'
- The script filters out 'N/A' codenames from the reference table
- Codenames appear before model names in visualizations for quick identification
- All existing functionality is preserved; codenames are additive
