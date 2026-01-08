"""
Comparison Performance Analysis Script
Aggregates and visualizes results from all models in save/non_preprocessing and save/processing directories
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configuration
NON_PREPROCESSING_DIR = Path("save/non_preprocessing")
PROCESSING_DIR = Path("save/processing")
OUTPUT_DIR = Path("performance_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model name mappings
BASE_MODEL_NAMES = {
    'bert': 'BERT',
    'roberta': 'RoBERTa',
    'distilbert': 'DistilBERT'
}

WHITENING_NAMES = {
    'bert-whitening-pca': 'BERT + PCA',
    'bert-whitening-svd': 'BERT + SVD',
    'bert-whitening-zca': 'BERT + ZCA',
    'roberta-whitening-pca': 'RoBERTa + PCA',
    'roberta-whitening-svd': 'RoBERTa + SVD',
    'roberta-whitening-zca': 'RoBERTa + ZCA'
}

CLASSIFIER_NAMES = {
    'bilstm': 'Bi-LSTM',
    'mlp': 'MLP',
    'benchmark': 'Benchmark'
}

# Codename mappings based on 15 models implementation plan
# Format: (base_model, preprocessing, classifier) -> codename
CODENAME_MAPPING = {
    # Skenario 1: Benchmark (Baseline Models)
    ('bert', 'none', 'benchmark'): 'A01',
    ('roberta', 'none', 'benchmark'): 'A02',
    ('distilbert', 'none', 'benchmark'): 'A03',
    
    # Skenario 2: Modified Models with BiLSTM Classifier
    ('bert', 'svd', 'bilstm'): 'B01',
    ('bert', 'pca', 'bilstm'): 'B02',
    ('bert', 'zca', 'bilstm'): 'B03',
    ('roberta', 'svd', 'bilstm'): 'B04',
    ('roberta', 'pca', 'bilstm'): 'B05',
    ('roberta', 'zca', 'bilstm'): 'B06',
    
    # Skenario 3: Modified Models with MLP Classifier
    ('bert', 'svd', 'mlp'): 'C01',
    ('bert', 'pca', 'mlp'): 'C02',
    ('bert', 'zca', 'mlp'): 'C03',
    ('roberta', 'svd', 'mlp'): 'C04',
    ('roberta', 'pca', 'mlp'): 'C05',
    ('roberta', 'zca', 'mlp'): 'C06',
}

# Codename descriptions
CODENAME_DESCRIPTIONS = {
    'A01': 'BERT Benchmark',
    'A02': 'RoBERTa Benchmark',
    'A03': 'DistilBERT Benchmark',
    'B01': 'BERT + SVD Whitening + BiLSTM',
    'B02': 'BERT + PCA Whitening + BiLSTM',
    'B03': 'BERT + ZCA Whitening + BiLSTM',
    'B04': 'RoBERTa + SVD Whitening + BiLSTM',
    'B05': 'RoBERTa + PCA Whitening + BiLSTM',
    'B06': 'RoBERTa + ZCA Whitening + BiLSTM',
    'C01': 'BERT + SVD Whitening + MLP',
    'C02': 'BERT + PCA Whitening + MLP',
    'C03': 'BERT + ZCA Whitening + MLP',
    'C04': 'RoBERTa + SVD Whitening + MLP',
    'C05': 'RoBERTa + PCA Whitening + MLP',
    'C06': 'RoBERTa + ZCA Whitening + MLP',
}

def get_codename(base_model, preprocessing, classifier):
    """Get codename based on model components"""
    key = (base_model, preprocessing, classifier)
    return CODENAME_MAPPING.get(key, 'N/A')

def parse_model_name(model_dir_name):
    """Parse model directory name to extract components"""
    parts = model_dir_name.split('-')
    
    # Extract base model
    base_model = 'unknown'
    for bm in BASE_MODEL_NAMES.keys():
        if bm in model_dir_name.lower():
            base_model = bm
            break
    
    # Extract preprocessing type
    preprocessing = 'none'
    for wt in WHITENING_NAMES.keys():
        if wt in model_dir_name:
            preprocessing = wt.split('-')[-1]  # Get pca, svd, or zca
            break
    
    # Extract classifier
    classifier = 'benchmark'
    for cl in CLASSIFIER_NAMES.keys():
        if cl in model_dir_name.lower():
            classifier = cl
            break
    
    # Get codename
    codename = get_codename(base_model, preprocessing, classifier)
    
    return {
        'base_model': base_model,
        'preprocessing': preprocessing,
        'classifier': classifier,
        'codename': codename,
        'full_name': model_dir_name
    }

def load_model_results(base_dir, preprocessing_type):
    """Load model results from a directory"""
    print(f"\nLoading results from {preprocessing_type}...")
    all_results = []
    
    if not base_dir.exists():
        print(f"  ✗ Directory not found: {base_dir}")
        return all_results
    
    # Iterate through all model directories
    for model_dir in base_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        # Parse model name
        model_info = parse_model_name(model_dir.name)
        
        # Look for result subdirectories
        for result_dir in model_dir.iterdir():
            if not result_dir.is_dir():
                continue
            
            # Load metrics summary
            metrics_file = result_dir / "metrics_summary_mean_std.csv"
            eval_file = result_dir / "evaluation_result_all_seeds.csv"
            efficiency_file = result_dir / "efficiency_summary.csv"
            
            if metrics_file.exists():
                try:
                    metrics_df = pd.read_csv(metrics_file)
                    
                    # Get mean values
                    mean_metrics = metrics_df[metrics_df['metric'].str.contains('mean|ACC|F1|REC|PRE', case=False, na=False)]
                    
                    if not mean_metrics.empty:
                        result = {
                            'model_name': model_dir.name,
                            'result_dir': result_dir.name,
                            'base_model': model_info['base_model'],
                            'preprocessing': model_info['preprocessing'],
                            'classifier': model_info['classifier'],
                            'codename': model_info['codename'],
                            'preprocessing_type': preprocessing_type,
                            'full_path': str(result_dir)
                        }
                        
                        # Extract metrics
                        for _, row in metrics_df.iterrows():
                            metric_name = row['metric']
                            result[f'{metric_name}_mean'] = row['mean']
                            result[f'{metric_name}_std'] = row['std']
                        
                        # Load evaluation results if available
                        if eval_file.exists():
                            eval_df = pd.read_csv(eval_file)
                            result['num_seeds'] = len(eval_df)
                            result['seeds'] = eval_df['seed'].tolist()
                        
                        # Load efficiency data if available
                        if efficiency_file.exists():
                            eff_df = pd.read_csv(efficiency_file, header=1)
                            # Extract training time
                            train_row = eff_df[eff_df.iloc[:, 0].str.contains('Train', case=False, na=False)]
                            if not train_row.empty:
                                result['train_time_mean'] = train_row.iloc[0, 1]  # mean elapsed time
                                result['train_time_std'] = train_row.iloc[0, 2]   # std elapsed time
                            
                            # Extract test time
                            test_row = eff_df[eff_df.iloc[:, 0].str.contains('Test', case=False, na=False)]
                            if not test_row.empty:
                                result['test_time_mean'] = test_row.iloc[0, 1]  # mean elapsed time
                                result['test_time_std'] = test_row.iloc[0, 2]   # std elapsed time
                            
                            # Extract GPU usage
                            gpu_row = eff_df[eff_df.iloc[:, 0].str.contains('Train', case=False, na=False)]
                            if not gpu_row.empty:
                                result['gpu_usage_mean'] = gpu_row.iloc[0, 5]  # mean GPU usage
                                result['gpu_usage_std'] = gpu_row.iloc[0, 6]   # std GPU usage
                        
                        all_results.append(result)
                        
                except Exception as e:
                    print(f"    Warning: Could not load {metrics_file}: {e}")
    
    print(f"  ✓ Loaded {len(all_results)} model results")
    return all_results

def aggregate_all_results():
    """Aggregate results from both preprocessing and non-preprocessing directories"""
    print("=" * 80)
    print("Aggregating Performance Results from All Models")
    print("=" * 80)
    
    # Load non-preprocessing results
    non_prep_results = load_model_results(NON_PREPROCESSING_DIR, 'non_preprocessing')
    
    # Load preprocessing results
    prep_results = load_model_results(PROCESSING_DIR, 'preprocessing')
    
    # Combine all results
    all_results = non_prep_results + prep_results
    
    if not all_results:
        print("\n✗ No results found")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    print(f"\n✓ Total results loaded: {len(df)}")
    print(f"  - Non-preprocessing: {len(non_prep_results)}")
    print(f"  - Preprocessing: {len(prep_results)}")
    
    return df

def create_performance_comparison_table(df):
    """Create comprehensive performance comparison table"""
    print("\nCreating performance comparison table...")
    
    # Select key metrics
    key_metrics = ['ACC_mean', 'F1_mean', 'REC_mean', 'PRE_mean']
    
    # Create comparison table
    comparison_data = []
    
    for _, row in df.iterrows():
        comparison_data.append({
            'Codename': row['codename'],
            'Model': row['model_name'],
            'Base Model': row['base_model'],
            'Preprocessing': row['preprocessing'].upper(),
            'Classifier': row['classifier'].upper(),
            'Type': row['preprocessing_type'].replace('_', ' ').title(),
            'Accuracy': row.get('ACC_mean', np.nan),
            'F1 Score': row.get('F1_mean', np.nan),
            'Recall': row.get('REC_mean', np.nan),
            'Precision': row.get('PRE_mean', np.nan),
            'Train Time (s)': row.get('train_time_mean', np.nan),
            'Test Time (s)': row.get('test_time_mean', np.nan),
            'GPU Usage (MB)': row.get('gpu_usage_mean', np.nan),
            'Seeds': row.get('num_seeds', 1)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by accuracy
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
    
    # Save to CSV
    comparison_df.to_csv(OUTPUT_DIR / 'performance_comparison_table.csv', index=False)
    print(f"  ✓ Saved: performance_comparison_table.csv")
    
    # Also save as formatted text
    with open(OUTPUT_DIR / 'performance_comparison_table.txt', 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("PERFORMANCE COMPARISON TABLE\n")
        f.write("=" * 120 + "\n\n")
        f.write(comparison_df.to_string(index=False))
    
    print(f"  ✓ Saved: performance_comparison_table.txt")
    
    return comparison_df

def create_codename_reference_table(df):
    """Create codename reference table"""
    print("\nCreating codename reference table...")
    
    # Get unique codenames
    unique_codenames = df[df['codename'] != 'N/A'][['codename', 'model_name', 'base_model', 'preprocessing', 'classifier']].drop_duplicates()
    
    if unique_codenames.empty:
        print("  ✗ No codenames found")
        return None
    
    # Sort by codename
    unique_codenames = unique_codenames.sort_values('codename')
    
    # Create reference table
    reference_data = []
    for _, row in unique_codenames.iterrows():
        reference_data.append({
            'Codename': row['codename'],
            'Description': CODENAME_DESCRIPTIONS.get(row['codename'], 'Unknown'),
            'Model Name': row['model_name'],
            'Base Model': row['base_model'].upper(),
            'Preprocessing': row['preprocessing'].upper(),
            'Classifier': row['classifier'].upper()
        })
    
    reference_df = pd.DataFrame(reference_data)
    
    # Save to CSV
    reference_df.to_csv(OUTPUT_DIR / 'codename_reference_table.csv', index=False)
    print(f"  ✓ Saved: codename_reference_table.csv")
    
    # Also save as formatted text
    with open(OUTPUT_DIR / 'codename_reference_table.txt', 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("CODENAME REFERENCE TABLE\n")
        f.write("=" * 120 + "\n\n")
        f.write(reference_df.to_string(index=False))
    
    print(f"  ✓ Saved: codename_reference_table.txt")
    
    return reference_df

def create_accuracy_comparison_plot(df):
    """Create accuracy comparison plot"""
    print("\nCreating accuracy comparison plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Overall accuracy comparison
    ax1 = axes[0, 0]
    df_sorted = df.sort_values('ACC_mean', ascending=False)
    bars = ax1.barh(range(len(df_sorted)), df_sorted['ACC_mean'], 
                    color=['#1f77b4' if t == 'non_preprocessing' else '#ff7f0e' 
                           for t in df_sorted['preprocessing_type']])
    ax1.set_yticks(range(len(df_sorted)))
    ax1.set_yticklabels([f"{row['codename']}: {row['model_name'][:20]}" for _, row in df_sorted.iterrows()], fontsize=8)
    ax1.set_xlabel('Accuracy', fontsize=11)
    ax1.set_title('Overall Accuracy Comparison', fontweight='bold', fontsize=12)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#1f77b4', label='Non-Preprocessing'),
                      Patch(facecolor='#ff7f0e', label='Preprocessing')]
    ax1.legend(handles=legend_elements, fontsize=9)
    
    # 2. Accuracy by base model
    ax2 = axes[0, 1]
    base_model_acc = df.groupby('base_model')['ACC_mean'].agg(['mean', 'std']).sort_values('mean', ascending=False)
    bars = ax2.bar(range(len(base_model_acc)), base_model_acc['mean'], 
                   yerr=base_model_acc['std'], capsize=5, alpha=0.7)
    ax2.set_xticks(range(len(base_model_acc)))
    ax2.set_xticklabels(base_model_acc.index, rotation=45, ha='right')
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('Accuracy by Base Model', fontweight='bold', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Accuracy by preprocessing type
    ax3 = axes[1, 0]
    prep_acc = df.groupby('preprocessing')['ACC_mean'].agg(['mean', 'std']).sort_values('mean', ascending=False)
    bars = ax3.bar(range(len(prep_acc)), prep_acc['mean'], 
                   yerr=prep_acc['std'], capsize=5, alpha=0.7)
    ax3.set_xticks(range(len(prep_acc)))
    ax3.set_xticklabels([p.upper() for p in prep_acc.index], rotation=45, ha='right')
    ax3.set_ylabel('Accuracy', fontsize=11)
    ax3.set_title('Accuracy by Preprocessing Type', fontweight='bold', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Accuracy by classifier
    ax4 = axes[1, 1]
    classifier_acc = df.groupby('classifier')['ACC_mean'].agg(['mean', 'std']).sort_values('mean', ascending=False)
    bars = ax4.bar(range(len(classifier_acc)), classifier_acc['mean'], 
                   yerr=classifier_acc['std'], capsize=5, alpha=0.7)
    ax4.set_xticks(range(len(classifier_acc)))
    ax4.set_xticklabels([c.upper() for c in classifier_acc.index], rotation=45, ha='right')
    ax4.set_ylabel('Accuracy', fontsize=11)
    ax4.set_title('Accuracy by Classifier Type', fontweight='bold', fontsize=12)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: accuracy_comparison.png")

def create_efficiency_comparison_plot(df):
    """Create efficiency comparison plot"""
    print("\nCreating efficiency comparison plot...")
    
    # Filter models with efficiency data
    eff_df = df[df['train_time_mean'].notna()].copy()
    
    if eff_df.empty:
        print("  ✗ No efficiency data available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Training time comparison
    ax1 = axes[0, 0]
    eff_sorted = eff_df.sort_values('train_time_mean', ascending=True)
    bars = ax1.barh(range(len(eff_sorted)), eff_sorted['train_time_mean'] / 60,
                    color=['#1f77b4' if t == 'non_preprocessing' else '#ff7f0e' 
                           for t in eff_sorted['preprocessing_type']])
    ax1.set_yticks(range(len(eff_sorted)))
    ax1.set_yticklabels([f"{row['codename']}: {row['model_name'][:20]}" for _, row in eff_sorted.iterrows()], fontsize=8)
    ax1.set_xlabel('Training Time (minutes)', fontsize=11)
    ax1.set_title('Training Time Comparison', fontweight='bold', fontsize=12)
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Test time comparison
    ax2 = axes[0, 1]
    test_sorted = eff_df.sort_values('test_time_mean', ascending=True)
    bars = ax2.barh(range(len(test_sorted)), test_sorted['test_time_mean'],
                    color=['#1f77b4' if t == 'non_preprocessing' else '#ff7f0e' 
                           for t in test_sorted['preprocessing_type']])
    ax2.set_yticks(range(len(test_sorted)))
    ax2.set_yticklabels([f"{row['codename']}: {row['model_name'][:20]}" for _, row in test_sorted.iterrows()], fontsize=8)
    ax2.set_xlabel('Test Time (seconds)', fontsize=11)
    ax2.set_title('Test Time Comparison', fontweight='bold', fontsize=12)
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. GPU usage comparison
    ax3 = axes[1, 0]
    gpu_sorted = eff_df.sort_values('gpu_usage_mean', ascending=True)
    bars = ax3.barh(range(len(gpu_sorted)), gpu_sorted['gpu_usage_mean'],
                    color=['#1f77b4' if t == 'non_preprocessing' else '#ff7f0e' 
                           for t in gpu_sorted['preprocessing_type']])
    ax3.set_yticks(range(len(gpu_sorted)))
    ax3.set_yticklabels([f"{row['codename']}: {row['model_name'][:20]}" for _, row in gpu_sorted.iterrows()], fontsize=8)
    ax3.set_xlabel('GPU Usage (MB)', fontsize=11)
    ax3.set_title('GPU Usage Comparison', fontweight='bold', fontsize=12)
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. Efficiency scatter plot (accuracy vs training time)
    ax4 = axes[1, 1]
    colors = ['#1f77b4' if t == 'non_preprocessing' else '#ff7f0e' for t in eff_df['preprocessing_type']]
    scatter = ax4.scatter(eff_df['train_time_mean'] / 60, eff_df['ACC_mean'], 
                         c=colors, s=100, alpha=0.7, edgecolors='black')
    ax4.set_xlabel('Training Time (minutes)', fontsize=11)
    ax4.set_ylabel('Accuracy', fontsize=11)
    ax4.set_title('Accuracy vs Training Time', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', 
                                  markersize=10, label='Non-Preprocessing'),
                      plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', 
                                  markersize=10, label='Preprocessing')]
    ax4.legend(handles=legend_elements, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'efficiency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: efficiency_comparison.png")

def create_preprocessing_impact_analysis(df):
    """Create preprocessing impact analysis"""
    print("\nCreating preprocessing impact analysis...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    metrics = ['ACC_mean', 'F1_mean', 'REC_mean', 'PRE_mean']
    metric_names = ['Accuracy', 'F1 Score', 'Recall', 'Precision']
    
    # 1-4. Metric comparison by preprocessing type
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]
        
        prep_comparison = df.groupby(['preprocessing', 'preprocessing_type'])[metric].mean().unstack()
        
        if not prep_comparison.empty:
            x = np.arange(len(prep_comparison.index))
            width = 0.35
            
            if 'non_preprocessing' in prep_comparison.columns:
                bars1 = ax.bar(x - width/2, prep_comparison['non_preprocessing'], width, 
                              label='Non-Preprocessing', alpha=0.8)
            
            if 'preprocessing' in prep_comparison.columns:
                bars2 = ax.bar(x + width/2, prep_comparison['preprocessing'], width, 
                              label='Preprocessing', alpha=0.8)
            
            ax.set_xticks(x)
            ax.set_xticklabels([p.upper() for p in prep_comparison.index], rotation=45, ha='right')
            ax.set_ylabel(metric_name, fontsize=11)
            ax.set_title(f'{metric_name} by Preprocessing Type', fontweight='bold', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)
    
    # 5. Overall preprocessing impact
    ax5 = axes[1, 2]
    overall_impact = df.groupby('preprocessing_type')[metrics].mean()
    
    if not overall_impact.empty:
        x = np.arange(len(metrics))
        width = 0.35
        
        if 'non_preprocessing' in overall_impact.index:
            bars1 = ax5.bar(x - width/2, overall_impact.loc['non_preprocessing'], width, 
                           label='Non-Preprocessing', alpha=0.8)
        
        if 'preprocessing' in overall_impact.index:
            bars2 = ax5.bar(x + width/2, overall_impact.loc['preprocessing'], width, 
                           label='Preprocessing', alpha=0.8)
        
        ax5.set_xticks(x)
        ax5.set_xticklabels(metric_names, rotation=45, ha='right')
        ax5.set_ylabel('Score', fontsize=11)
        ax5.set_title('Overall Preprocessing Impact', fontweight='bold', fontsize=12)
        ax5.legend(fontsize=9)
        ax5.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'preprocessing_impact_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: preprocessing_impact_analysis.png")

def create_model_ranking_dashboard(df):
    """Create comprehensive model ranking dashboard"""
    print("\nCreating model ranking dashboard...")
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    # 1. Top 10 models by accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    top_models = df.nlargest(10, 'ACC_mean')
    bars = ax1.barh(range(len(top_models)), top_models['ACC_mean'], 
                    color=['#1f77b4' if t == 'non_preprocessing' else '#ff7f0e' 
                           for t in top_models['preprocessing_type']])
    ax1.set_yticks(range(len(top_models)))
    ax1.set_yticklabels([f"{row['codename']}: {row['model_name'][:15]}" for _, row in top_models.iterrows()], fontsize=8)
    ax1.set_xlabel('Accuracy', fontsize=11)
    ax1.set_title('Top 10 Models by Accuracy', fontweight='bold', fontsize=12)
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Accuracy distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df['ACC_mean'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.axvline(df['ACC_mean'].mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {df["ACC_mean"].mean():.4f}')
    ax2.axvline(df['ACC_mean'].median(), color='green', linestyle='--', linewidth=2, 
               label=f'Median: {df["ACC_mean"].median():.4f}')
    ax2.set_xlabel('Accuracy', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Accuracy Distribution', fontweight='bold', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Base model comparison
    ax3 = fig.add_subplot(gs[0, 2])
    base_model_stats = df.groupby('base_model')['ACC_mean'].agg(['mean', 'std', 'count'])
    base_model_stats = base_model_stats.sort_values('mean', ascending=False)
    bars = ax3.bar(range(len(base_model_stats)), base_model_stats['mean'], 
                   yerr=base_model_stats['std'], capsize=5, alpha=0.7)
    ax3.set_xticks(range(len(base_model_stats)))
    ax3.set_xticklabels([bm.upper() for bm in base_model_stats.index], rotation=45, ha='right')
    ax3.set_ylabel('Accuracy', fontsize=11)
    ax3.set_title('Base Model Comparison', fontweight='bold', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Preprocessing technique comparison
    ax4 = fig.add_subplot(gs[1, 0])
    prep_stats = df.groupby('preprocessing')['ACC_mean'].agg(['mean', 'std', 'count'])
    prep_stats = prep_stats.sort_values('mean', ascending=False)
    bars = ax4.bar(range(len(prep_stats)), prep_stats['mean'], 
                   yerr=prep_stats['std'], capsize=5, alpha=0.7)
    ax4.set_xticks(range(len(prep_stats)))
    ax4.set_xticklabels([p.upper() for p in prep_stats.index], rotation=45, ha='right')
    ax4.set_ylabel('Accuracy', fontsize=11)
    ax4.set_title('Preprocessing Technique Comparison', fontweight='bold', fontsize=12)
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Classifier comparison
    ax5 = fig.add_subplot(gs[1, 1])
    classifier_stats = df.groupby('classifier')['ACC_mean'].agg(['mean', 'std', 'count'])
    classifier_stats = classifier_stats.sort_values('mean', ascending=False)
    bars = ax5.bar(range(len(classifier_stats)), classifier_stats['mean'], 
                   yerr=classifier_stats['std'], capsize=5, alpha=0.7)
    ax5.set_xticks(range(len(classifier_stats)))
    ax5.set_xticklabels([c.upper() for c in classifier_stats.index], rotation=45, ha='right')
    ax5.set_ylabel('Accuracy', fontsize=11)
    ax5.set_title('Classifier Comparison', fontweight='bold', fontsize=12)
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Preprocessing vs Non-preprocessing
    ax6 = fig.add_subplot(gs[1, 2])
    type_stats = df.groupby('preprocessing_type')['ACC_mean'].agg(['mean', 'std', 'count'])
    colors = ['#1f77b4', '#ff7f0e']
    bars = ax6.bar(range(len(type_stats)), type_stats['mean'], 
                   yerr=type_stats['std'], capsize=5, color=colors, alpha=0.7)
    ax6.set_xticks(range(len(type_stats)))
    ax6.set_xticklabels([t.replace('_', ' ').title() for t in type_stats.index], rotation=45, ha='right')
    ax6.set_ylabel('Accuracy', fontsize=11)
    ax6.set_title('Preprocessing vs Non-Preprocessing', fontweight='bold', fontsize=12)
    ax6.grid(axis='y', alpha=0.3)
    
    # 7. Top 10 models by F1 score
    ax7 = fig.add_subplot(gs[2, 0])
    f1_sorted = df.nlargest(10, 'F1_mean')
    bars = ax7.barh(range(len(f1_sorted)), f1_sorted['F1_mean'], 
                    color=['#1f77b4' if t == 'non_preprocessing' else '#ff7f0e' 
                           for t in f1_sorted['preprocessing_type']])
    ax7.set_yticks(range(len(f1_sorted)))
    ax7.set_yticklabels([f"{row['codename']}: {row['model_name'][:15]}" for _, row in f1_sorted.iterrows()], fontsize=8)
    ax7.set_xlabel('F1 Score', fontsize=11)
    ax7.set_title('Top 10 Models by F1 Score', fontweight='bold', fontsize=12)
    ax7.grid(axis='x', alpha=0.3)
    
    # 8. Training efficiency (accuracy vs time)
    ax8 = fig.add_subplot(gs[2, 1])
    eff_df = df[df['train_time_mean'].notna()]
    if not eff_df.empty:
        colors = ['#1f77b4' if t == 'non_preprocessing' else '#ff7f0e' 
                 for t in eff_df['preprocessing_type']]
        scatter = ax8.scatter(eff_df['train_time_mean'] / 60, eff_df['ACC_mean'], 
                             c=colors, s=100, alpha=0.7, edgecolors='black')
        ax8.set_xlabel('Training Time (minutes)', fontsize=11)
        ax8.set_ylabel('Accuracy', fontsize=11)
        ax8.set_title('Training Efficiency', fontweight='bold', fontsize=12)
        ax8.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#1f77b4', label='Non-Preprocessing'),
                          Patch(facecolor='#ff7f0e', label='Preprocessing')]
        ax8.legend(handles=legend_elements, fontsize=9)
    
    # 9. Summary statistics
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    summary_text = "SUMMARY STATISTICS\n\n"
    summary_text += f"Total Models: {len(df)}\n"
    summary_text += f"Non-Preprocessing: {len(df[df['preprocessing_type'] == 'non_preprocessing'])}\n"
    summary_text += f"Preprocessing: {len(df[df['preprocessing_type'] == 'preprocessing'])}\n\n"
    summary_text += f"Best Accuracy: {df['ACC_mean'].max():.4f}\n"
    summary_text += f"Worst Accuracy: {df['ACC_mean'].min():.4f}\n"
    summary_text += f"Mean Accuracy: {df['ACC_mean'].mean():.4f}\n"
    summary_text += f"Std Accuracy: {df['ACC_mean'].std():.4f}\n\n"
    summary_text += f"Best F1: {df['F1_mean'].max():.4f}\n"
    summary_text += f"Mean F1: {df['F1_mean'].mean():.4f}"
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=11,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('LC-BERT Model Performance Ranking Dashboard', fontsize=18, fontweight='bold')
    plt.savefig(OUTPUT_DIR / 'model_ranking_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: model_ranking_dashboard.png")

def create_detailed_analysis_report(df):
    """Create detailed analysis report"""
    print("\nCreating detailed analysis report...")
    
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("LC-BERT MODEL PERFORMANCE ANALYSIS REPORT")
    report_lines.append("=" * 100)
    report_lines.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"\nTotal Models Analyzed: {len(df)}")
    report_lines.append(f"Non-Preprocessing: {len(df[df['preprocessing_type'] == 'non_preprocessing'])}")
    report_lines.append(f"Preprocessing: {len(df[df['preprocessing_type'] == 'preprocessing'])}")
    
    # Overall statistics
    report_lines.append("\n" + "=" * 100)
    report_lines.append("OVERALL PERFORMANCE STATISTICS")
    report_lines.append("=" * 100)
    
    metrics = ['ACC_mean', 'F1_mean', 'REC_mean', 'PRE_mean']
    metric_names = ['Accuracy', 'F1 Score', 'Recall', 'Precision']
    
    for metric, name in zip(metrics, metric_names):
        if metric in df.columns:
            report_lines.append(f"\n{name}:")
            report_lines.append(f"  Best:    {df[metric].max():.6f}")
            report_lines.append(f"  Worst:   {df[metric].min():.6f}")
            report_lines.append(f"  Mean:    {df[metric].mean():.6f}")
            report_lines.append(f"  Std:     {df[metric].std():.6f}")
            report_lines.append(f"  Median:  {df[metric].median():.6f}")
    
    # Top models
    report_lines.append("\n" + "=" * 100)
    report_lines.append("TOP 10 MODELS BY ACCURACY")
    report_lines.append("=" * 100)
    
    top_models = df.nlargest(10, 'ACC_mean')
    for idx, (_, row) in enumerate(top_models.iterrows(), 1):
        report_lines.append(f"\n{idx}. [{row['codename']}] {row['model_name']}")
        report_lines.append(f"   Type: {row['preprocessing_type'].replace('_', ' ').title()}")
        report_lines.append(f"   Base Model: {row['base_model']}")
        report_lines.append(f"   Preprocessing: {row['preprocessing'].upper()}")
        report_lines.append(f"   Classifier: {row['classifier'].upper()}")
        report_lines.append(f"   Accuracy: {row['ACC_mean']:.6f} ± {row.get('ACC_std', 0):.6f}")
        report_lines.append(f"   F1 Score: {row['F1_mean']:.6f} ± {row.get('F1_std', 0):.6f}")
        report_lines.append(f"   Recall: {row['REC_mean']:.6f} ± {row.get('REC_std', 0):.6f}")
        report_lines.append(f"   Precision: {row['PRE_mean']:.6f} ± {row.get('PRE_std', 0):.6f}")
        
        if 'train_time_mean' in row and pd.notna(row['train_time_mean']):
            report_lines.append(f"   Training Time: {row['train_time_mean']:.2f}s ± {row.get('train_time_std', 0):.2f}s")
        
        if 'test_time_mean' in row and pd.notna(row['test_time_mean']):
            report_lines.append(f"   Test Time: {row['test_time_mean']:.2f}s ± {row.get('test_time_std', 0):.2f}s")
        
        if 'gpu_usage_mean' in row and pd.notna(row['gpu_usage_mean']):
            report_lines.append(f"   GPU Usage: {row['gpu_usage_mean']:.2f}MB ± {row.get('gpu_usage_std', 0):.2f}MB")
    
    # Base model comparison
    report_lines.append("\n" + "=" * 100)
    report_lines.append("BASE MODEL COMPARISON")
    report_lines.append("=" * 100)
    
    base_model_stats = df.groupby('base_model')[metrics].agg(['mean', 'std', 'count'])
    for base_model in base_model_stats.index:
        report_lines.append(f"\n{base_model.upper()}:")
        report_lines.append(f"  Number of Models: {int(base_model_stats.loc[base_model, ('ACC_mean', 'count')])}")
        for metric, name in zip(metrics, metric_names):
            mean_val = base_model_stats.loc[base_model, (metric, 'mean')]
            std_val = base_model_stats.loc[base_model, (metric, 'std')]
            report_lines.append(f"  {name}: {mean_val:.6f} ± {std_val:.6f}")
    
    # Preprocessing technique comparison
    report_lines.append("\n" + "=" * 100)
    report_lines.append("PREPROCESSING TECHNIQUE COMPARISON")
    report_lines.append("=" * 100)
    
    prep_stats = df.groupby('preprocessing')[metrics].agg(['mean', 'std', 'count'])
    for prep in prep_stats.index:
        report_lines.append(f"\n{prep.upper()}:")
        report_lines.append(f"  Number of Models: {int(prep_stats.loc[prep, ('ACC_mean', 'count')])}")
        for metric, name in zip(metrics, metric_names):
            mean_val = prep_stats.loc[prep, (metric, 'mean')]
            std_val = prep_stats.loc[prep, (metric, 'std')]
            report_lines.append(f"  {name}: {mean_val:.6f} ± {std_val:.6f}")
    
    # Classifier comparison
    report_lines.append("\n" + "=" * 100)
    report_lines.append("CLASSIFIER COMPARISON")
    report_lines.append("=" * 100)
    
    classifier_stats = df.groupby('classifier')[metrics].agg(['mean', 'std', 'count'])
    for classifier in classifier_stats.index:
        report_lines.append(f"\n{classifier.upper()}:")
        report_lines.append(f"  Number of Models: {int(classifier_stats.loc[classifier, ('ACC_mean', 'count')])}")
        for metric, name in zip(metrics, metric_names):
            mean_val = classifier_stats.loc[classifier, (metric, 'mean')]
            std_val = classifier_stats.loc[classifier, (metric, 'std')]
            report_lines.append(f"  {name}: {mean_val:.6f} ± {std_val:.6f}")
    
    # Preprocessing vs Non-preprocessing comparison
    report_lines.append("\n" + "=" * 100)
    report_lines.append("PREPROCESSING VS NON-PREPROCESSING COMPARISON")
    report_lines.append("=" * 100)
    
    type_stats = df.groupby('preprocessing_type')[metrics].agg(['mean', 'std', 'count'])
    for ptype in type_stats.index:
        report_lines.append(f"\n{ptype.replace('_', ' ').title()}:")
        report_lines.append(f"  Number of Models: {int(type_stats.loc[ptype, ('ACC_mean', 'count')])}")
        for metric, name in zip(metrics, metric_names):
            mean_val = type_stats.loc[ptype, (metric, 'mean')]
            std_val = type_stats.loc[ptype, (metric, 'std')]
            report_lines.append(f"  {name}: {mean_val:.6f} ± {std_val:.6f}")
    
    # Efficiency analysis
    report_lines.append("\n" + "=" * 100)
    report_lines.append("EFFICIENCY ANALYSIS")
    report_lines.append("=" * 100)
    
    eff_df = df[df['train_time_mean'].notna()]
    if not eff_df.empty:
        report_lines.append(f"\nTraining Time:")
        report_lines.append(f"  Fastest:  {eff_df['train_time_mean'].min():.2f}s")
        report_lines.append(f"  Slowest:  {eff_df['train_time_mean'].max():.2f}s")
        report_lines.append(f"  Mean:     {eff_df['train_time_mean'].mean():.2f}s")
        
        report_lines.append(f"\nTest Time:")
        report_lines.append(f"  Fastest:  {eff_df['test_time_mean'].min():.2f}s")
        report_lines.append(f"  Slowest:  {eff_df['test_time_mean'].max():.2f}s")
        report_lines.append(f"  Mean:     {eff_df['test_time_mean'].mean():.2f}s")
        
        report_lines.append(f"\nGPU Usage:")
        report_lines.append(f"  Lowest:   {eff_df['gpu_usage_mean'].min():.2f}MB")
        report_lines.append(f"  Highest:  {eff_df['gpu_usage_mean'].max():.2f}MB")
        report_lines.append(f"  Mean:     {eff_df['gpu_usage_mean'].mean():.2f}MB")
        
        # Most efficient models (best accuracy per training time)
        report_lines.append(f"\nTOP 5 MOST EFFICIENT MODELS (Accuracy/Training Time):")
        eff_df['efficiency_score'] = eff_df['ACC_mean'] / (eff_df['train_time_mean'] / 60)
        top_efficient = eff_df.nlargest(5, 'efficiency_score')
        for idx, (_, row) in enumerate(top_efficient.iterrows(), 1):
            report_lines.append(f"\n{idx}. [{row['codename']}] {row['model_name']}")
            report_lines.append(f"   Efficiency Score: {row['efficiency_score']:.6f}")
            report_lines.append(f"   Accuracy: {row['ACC_mean']:.6f}")
            report_lines.append(f"   Training Time: {row['train_time_mean']:.2f}s")
    
    # Save report
    report_text = "\n".join(report_lines)
    
    with open(OUTPUT_DIR / 'detailed_analysis_report.txt', 'w') as f:
        f.write(report_text)
    
    print(f"  ✓ Saved: detailed_analysis_report.txt")
    
    return report_text

def main():
    """Main execution function"""
    print("=" * 100)
    print("LC-BERT COMPARISON PERFORMANCE ANALYSIS")
    print("=" * 100)
    
    # Aggregate all results
    df = aggregate_all_results()
    
    if df is None or df.empty:
        print("\n✗ No data available for analysis")
        return
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Create analysis outputs
    create_performance_comparison_table(df)
    create_codename_reference_table(df)
    create_accuracy_comparison_plot(df)
    create_efficiency_comparison_plot(df)
    create_preprocessing_impact_analysis(df)
    create_model_ranking_dashboard(df)
    create_detailed_analysis_report(df)
    
    # Save aggregated data
    df.to_csv(OUTPUT_DIR / 'aggregated_results.csv', index=False)
    print(f"\n✓ Saved: aggregated_results.csv")
    
    print("\n" + "=" * 100)
    print(f"All analysis results saved to: {OUTPUT_DIR}")
    print("=" * 100)
    
    # Print summary
    print("\nSUMMARY:")
    print(f"  Total models analyzed: {len(df)}")
    print(f"  Best accuracy: {df['ACC_mean'].max():.6f}")
    print(f"  Mean accuracy: {df['ACC_mean'].mean():.6f}")
    print(f"  Best F1 score: {df['F1_mean'].max():.6f}")
    print(f"  Mean F1 score: {df['F1_mean'].mean():.6f}")

if __name__ == "__main__":
    main()
