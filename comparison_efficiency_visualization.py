"""
Comparison Efficiency Visualization Script
Visualizes aggregated results from all 15 models across different scenarios
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
EFFICIENCY_DIR = Path("efficiency_analysis")
OUTPUT_DIR = Path("efficiency_analysis/comparison_visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model definitions
MODELS = {
    'A01': {'name': 'BERT Benchmark', 'scenario': 'Skenario 1', 'type': 'Benchmark'},
    'A02': {'name': 'RoBERTa Benchmark', 'scenario': 'Skenario 1', 'type': 'Benchmark'},
    'A03': {'name': 'DistilBERT Benchmark', 'scenario': 'Skenario 1', 'type': 'Benchmark'},
    'B01': {'name': 'BERT + Bi-LSTM + J.Su', 'scenario': 'Skenario 2', 'type': 'Modified'},
    'B02': {'name': 'BERT + Bi-LSTM + PCA', 'scenario': 'Skenario 2', 'type': 'Modified'},
    'B03': {'name': 'BERT + Bi-LSTM + ZCA', 'scenario': 'Skenario 2', 'type': 'Modified'},
    'B04': {'name': 'RoBERTa + Bi-LSTM + J.Su', 'scenario': 'Skenario 2', 'type': 'Modified'},
    'B05': {'name': 'RoBERTa + Bi-LSTM + PCA', 'scenario': 'Skenario 2', 'type': 'Modified'},
    'B06': {'name': 'RoBERTa + Bi-LSTM + ZCA', 'scenario': 'Skenario 2', 'type': 'Modified'},
    'C01': {'name': 'BERT + MLP + J.Su', 'scenario': 'Skenario 3', 'type': 'Modified'},
    'C02': {'name': 'BERT + MLP + PCA', 'scenario': 'Skenario 3', 'type': 'Modified'},
    'C03': {'name': 'BERT + MLP + ZCA', 'scenario': 'Skenario 3', 'type': 'Modified'},
    'C04': {'name': 'RoBERTa + MLP + J.Su', 'scenario': 'Skenario 3', 'type': 'Modified'},
    'C05': {'name': 'RoBERTa + MLP + PCA', 'scenario': 'Skenario 3', 'type': 'Modified'},
    'C06': {'name': 'RoBERTa + MLP + ZCA', 'scenario': 'Skenario 3', 'type': 'Modified'},
}

def load_all_results():
    """Load all efficiency results from CSV files"""
    print("Loading efficiency results...")
    
    all_data = []
    
    # Load raw results
    raw_dir = EFFICIENCY_DIR / "raw_results"
    if raw_dir.exists():
        for csv_file in raw_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                df['source_file'] = csv_file.name
                all_data.append(df)
            except Exception as e:
                print(f"Warning: Could not load {csv_file}: {e}")
    
    # Load summary files
    for summary_file in EFFICIENCY_DIR.glob("summary_by_*.csv"):
        try:
            df = pd.read_csv(summary_file)
            df['source_file'] = summary_file.name
            all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not load {summary_file}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"✓ Loaded {len(combined_df)} records from {len(all_data)} files")
        return combined_df
    else:
        print("✗ No data files found")
        return pd.DataFrame()

def create_model_comparison_heatmap(df, metric='accuracy'):
    """Create heatmap comparing all models"""
    print(f"\nCreating {metric} comparison heatmap...")
    
    # Pivot data for heatmap
    if 'model' in df.columns and 'percentage' in df.columns:
        pivot_df = df.pivot(index='model', columns='percentage', values=metric)
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                    cbar_kws={'label': metric.capitalize()})
        plt.title(f'{metric.capitalize()} Comparison Across All Models', fontsize=16, fontweight='bold')
        plt.xlabel('Training Data Percentage', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'model_comparison_{metric}_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: model_comparison_{metric}_heatmap.png")

def create_scenario_comparison(df):
    """Compare performance across scenarios"""
    print("\nCreating scenario comparison...")
    
    if 'scenario' in df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            if metric in df.columns:
                scenario_data = df.groupby('scenario')[metric].mean().sort_values(ascending=False)
                bars = ax.bar(range(len(scenario_data)), scenario_data.values, 
                             color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                ax.set_xticks(range(len(scenario_data)))
                ax.set_xticklabels(scenario_data.index, rotation=45, ha='right')
                ax.set_ylabel(metric.capitalize())
                ax.set_title(f'Average {metric.capitalize()} by Scenario')
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'scenario_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: scenario_comparison.png")

def create_efficiency_curves(df):
    """Create efficiency curves showing performance vs data percentage"""
    print("\nCreating efficiency curves...")
    
    if 'model' in df.columns and 'percentage' in df.columns and 'accuracy' in df.columns:
        plt.figure(figsize=(16, 10))
        
        # Group by model and plot
        for model_id in df['model'].unique():
            model_data = df[df['model'] == model_id].sort_values('percentage')
            if len(model_data) > 0:
                model_name = MODELS.get(model_id, {}).get('name', model_id)
                scenario = MODELS.get(model_id, {}).get('scenario', 'Unknown')
                marker = 'o' if scenario == 'Skenario 1' else ('s' if scenario == 'Skenario 2' else '^')
                plt.plot(model_data['percentage'], model_data['accuracy'], 
                        marker=marker, label=model_name, alpha=0.7, linewidth=2)
        
        plt.xlabel('Training Data Percentage', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Model Accuracy vs Training Data Size', fontsize=16, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'efficiency_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: efficiency_curves.png")

def create_model_ranking(df):
    """Create ranking visualization for all models"""
    print("\nCreating model ranking...")
    
    if 'model' in df.columns and 'accuracy' in df.columns:
        # Calculate average accuracy for each model
        model_avg = df.groupby('model')['accuracy'].mean().sort_values(ascending=False)
        
        # Create ranking with model names
        ranking_data = []
        for model_id in model_avg.index:
            model_info = MODELS.get(model_id, {})
            ranking_data.append({
                'Model ID': model_id,
                'Model Name': model_info.get('name', model_id),
                'Scenario': model_info.get('scenario', 'Unknown'),
                'Type': model_info.get('type', 'Unknown'),
                'Avg Accuracy': model_avg[model_id]
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(12, 10))
        
        colors = ['#1f77b4' if t == 'Benchmark' else '#ff7f0e' for t in ranking_df['Type']]
        bars = ax.barh(range(len(ranking_df)), ranking_df['Avg Accuracy'], color=colors)
        
        ax.set_yticks(range(len(ranking_df)))
        ax.set_yticklabels([f"{row['Model ID']}: {row['Model Name']}" for _, row in ranking_df.iterrows()], 
                          fontsize=9)
        ax.set_xlabel('Average Accuracy', fontsize=12)
        ax.set_title('Model Ranking by Average Accuracy', fontsize=16, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2., 
                   f'{width:.3f}', ha='left', va='center', fontsize=8)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#1f77b4', label='Benchmark'),
                          Patch(facecolor='#ff7f0e', label='Modified')]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'model_ranking.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: model_ranking.png")
        
        # Save ranking table
        ranking_df.to_csv(OUTPUT_DIR / 'model_ranking_table.csv', index=False)
        print("✓ Saved: model_ranking_table.csv")

def create_whitening_comparison(df):
    """Compare different whitening techniques"""
    print("\nCreating whitening technique comparison...")
    
    # Extract whitening technique from model names
    whitening_map = {
        'J.Su': ['B01', 'B04', 'C01', 'C04'],
        'PCA': ['B02', 'B05', 'C02', 'C05'],
        'ZCA': ['B03', 'B06', 'C03', 'C06'],
        'None': ['A01', 'A02', 'A03']
    }
    
    whitening_data = []
    for technique, model_ids in whitening_map.items():
        for model_id in model_ids:
            if model_id in df['model'].values:
                model_data = df[df['model'] == model_id]
                avg_acc = model_data['accuracy'].mean()
                whitening_data.append({
                    'Whitening': technique,
                    'Model': model_id,
                    'Avg Accuracy': avg_acc
                })
    
    if whitening_data:
        whitening_df = pd.DataFrame(whitening_data)
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=whitening_df, x='Whitening', y='Avg Accuracy')
        plt.title('Whitening Technique Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Whitening Technique', fontsize=12)
        plt.ylabel('Average Accuracy', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'whitening_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: whitening_comparison.png")

def create_classifier_comparison(df):
    """Compare different classifiers (Bi-LSTM vs MLP)"""
    print("\nCreating classifier comparison...")
    
    classifier_map = {
        'Bi-LSTM': ['B01', 'B02', 'B03', 'B04', 'B05', 'B06'],
        'MLP': ['C01', 'C02', 'C03', 'C04', 'C05', 'C06'],
        'None (Benchmark)': ['A01', 'A02', 'A03']
    }
    
    classifier_data = []
    for classifier, model_ids in classifier_map.items():
        for model_id in model_ids:
            if model_id in df['model'].values:
                model_data = df[df['model'] == model_id]
                avg_acc = model_data['accuracy'].mean()
                classifier_data.append({
                    'Classifier': classifier,
                    'Model': model_id,
                    'Avg Accuracy': avg_acc
                })
    
    if classifier_data:
        classifier_df = pd.DataFrame(classifier_data)
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=classifier_df, x='Classifier', y='Avg Accuracy')
        plt.title('Classifier Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Classifier Type', fontsize=12)
        plt.ylabel('Average Accuracy', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'classifier_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: classifier_comparison.png")

def create_feature_extractor_comparison(df):
    """Compare BERT vs RoBERTa feature extractors"""
    print("\nCreating feature extractor comparison...")
    
    extractor_map = {
        'BERT': ['A01', 'B01', 'B02', 'B03', 'C01', 'C02', 'C03'],
        'RoBERTa': ['A02', 'B04', 'B05', 'B06', 'C04', 'C05', 'C06'],
        'DistilBERT': ['A03']
    }
    
    extractor_data = []
    for extractor, model_ids in extractor_map.items():
        for model_id in model_ids:
            if model_id in df['model'].values:
                model_data = df[df['model'] == model_id]
                avg_acc = model_data['accuracy'].mean()
                extractor_data.append({
                    'Extractor': extractor,
                    'Model': model_id,
                    'Avg Accuracy': avg_acc
                })
    
    if extractor_data:
        extractor_df = pd.DataFrame(extractor_data)
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=extractor_df, x='Extractor', y='Avg Accuracy')
        plt.title('Feature Extractor Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Feature Extractor', fontsize=12)
        plt.ylabel('Average Accuracy', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'feature_extractor_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: feature_extractor_comparison.png")

def create_summary_dashboard(df):
    """Create a comprehensive summary dashboard"""
    print("\nCreating summary dashboard...")
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Top 5 models by accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    if 'model' in df.columns and 'accuracy' in df.columns:
        top_models = df.groupby('model')['accuracy'].mean().sort_values(ascending=False).head(5)
        bars = ax1.barh(range(len(top_models)), top_models.values, color='steelblue')
        ax1.set_yticks(range(len(top_models)))
        ax1.set_yticklabels([MODELS.get(m, {}).get('name', m) for m in top_models.index], fontsize=8)
        ax1.set_xlabel('Accuracy')
        ax1.set_title('Top 5 Models by Accuracy', fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
    
    # 2. Scenario comparison
    ax2 = fig.add_subplot(gs[0, 1])
    if 'scenario' in df.columns and 'accuracy' in df.columns:
        scenario_acc = df.groupby('scenario')['accuracy'].mean()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        bars = ax2.bar(range(len(scenario_acc)), scenario_acc.values, color=colors)
        ax2.set_xticks(range(len(scenario_acc)))
        ax2.set_xticklabels(scenario_acc.index, rotation=45, ha='right')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Average Accuracy by Scenario', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Data efficiency
    ax3 = fig.add_subplot(gs[0, 2])
    if 'percentage' in df.columns and 'accuracy' in df.columns:
        efficiency = df.groupby('percentage')['accuracy'].mean()
        ax3.plot(efficiency.index, efficiency.values, marker='o', linewidth=2, markersize=8)
        ax3.set_xlabel('Data Percentage')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Data Efficiency Curve', fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # 4. Whitening comparison
    ax4 = fig.add_subplot(gs[1, 0])
    whitening_acc = {}
    for model_id, info in MODELS.items():
        if model_id in df['model'].values:
            whitening = 'J.Su' if 'J.Su' in info['name'] else ('PCA' if 'PCA' in info['name'] else ('ZCA' if 'ZCA' in info['name'] else 'None'))
            if whitening not in whitening_acc:
                whitening_acc[whitening] = []
            whitening_acc[whitening].append(df[df['model'] == model_id]['accuracy'].mean())
    
    if whitening_acc:
        whitening_avg = {k: np.mean(v) for k, v in whitening_acc.items()}
        bars = ax4.bar(range(len(whitening_avg)), whitening_avg.values(), 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax4.set_xticks(range(len(whitening_avg)))
        ax4.set_xticklabels(whitening_avg.keys())
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Whitening Techniques', fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
    
    # 5. Classifier comparison
    ax5 = fig.add_subplot(gs[1, 1])
    classifier_acc = {}
    for model_id, info in MODELS.items():
        if model_id in df['model'].values:
            classifier = 'Bi-LSTM' if 'Bi-LSTM' in info['name'] else ('MLP' if 'MLP' in info['name'] else 'None')
            if classifier not in classifier_acc:
                classifier_acc[classifier] = []
            classifier_acc[classifier].append(df[df['model'] == model_id]['accuracy'].mean())
    
    if classifier_acc:
        classifier_avg = {k: np.mean(v) for k, v in classifier_acc.items()}
        bars = ax5.bar(range(len(classifier_avg)), classifier_avg.values(), 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax5.set_xticks(range(len(classifier_avg)))
        ax5.set_xticklabels(classifier_avg.keys())
        ax5.set_ylabel('Accuracy')
        ax5.set_title('Classifier Types', fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)
    
    # 6. Feature extractor comparison
    ax6 = fig.add_subplot(gs[1, 2])
    extractor_acc = {}
    for model_id, info in MODELS.items():
        if model_id in df['model'].values:
            extractor = 'BERT' if 'BERT' in info['name'] and 'RoBERTa' not in info['name'] else ('RoBERTa' if 'RoBERTa' in info['name'] else 'DistilBERT')
            if extractor not in extractor_acc:
                extractor_acc[extractor] = []
            extractor_acc[extractor].append(df[df['model'] == model_id]['accuracy'].mean())
    
    if extractor_acc:
        extractor_avg = {k: np.mean(v) for k, v in extractor_acc.items()}
        bars = ax6.bar(range(len(extractor_avg)), extractor_avg.values(), 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax6.set_xticks(range(len(extractor_avg)))
        ax6.set_xticklabels(extractor_avg.keys())
        ax6.set_ylabel('Accuracy')
        ax6.set_title('Feature Extractors', fontweight='bold')
        ax6.grid(axis='y', alpha=0.3)
    
    # 7. Performance distribution
    ax7 = fig.add_subplot(gs[2, :])
    if 'accuracy' in df.columns:
        ax7.hist(df['accuracy'], bins=30, edgecolor='black', alpha=0.7)
        ax7.axvline(df['accuracy'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["accuracy"].mean():.3f}')
        ax7.axvline(df['accuracy'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["accuracy"].median():.3f}')
        ax7.set_xlabel('Accuracy')
        ax7.set_ylabel('Frequency')
        ax7.set_title('Accuracy Distribution Across All Experiments', fontweight='bold')
        ax7.legend()
        ax7.grid(axis='y', alpha=0.3)
    
    plt.suptitle('LC-BERT Model Comparison Summary Dashboard', fontsize=20, fontweight='bold')
    plt.savefig(OUTPUT_DIR / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: summary_dashboard.png")

def main():
    """Main execution function"""
    print("=" * 80)
    print("LC-BERT Comparison Efficiency Visualization")
    print("=" * 80)
    
    # Load data
    df = load_all_results()
    
    if df.empty:
        print("\n✗ No data available for visualization")
        return
    
    # Add model metadata
    if 'model' in df.columns:
        df['scenario'] = df['model'].map(lambda x: MODELS.get(x, {}).get('scenario', 'Unknown'))
        df['model_type'] = df['model'].map(lambda x: MODELS.get(x, {}).get('type', 'Unknown'))
        df['model_name'] = df['model'].map(lambda x: MODELS.get(x, {}).get('name', x))
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Create visualizations
    create_model_comparison_heatmap(df, 'accuracy')
    create_model_comparison_heatmap(df, 'f1')
    create_scenario_comparison(df)
    create_efficiency_curves(df)
    create_model_ranking(df)
    create_whitening_comparison(df)
    create_classifier_comparison(df)
    create_feature_extractor_comparison(df)
    create_summary_dashboard(df)
    
    print("\n" + "=" * 80)
    print(f"✓ All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
