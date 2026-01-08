"""
Comparison Efficiency Analysis Script for 14 Models
Aggregates and visualizes efficiency results from all model scenarios
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Model configuration mapping based on the 14-model scenario
MODEL_CONFIG = {
    # Benchmark Scenario (Skenario 1)
    'A01': {
        'name': 'BERT Benchmark',
        'feature_extraction': 'None',
        'classifier': 'BERT',
        'dimensionality_reduction': 'None',
        'dataset_pattern': 'ag-news-normal',
        'experiment_pattern': 'bert_benchmark'
    },
    'A02': {
        'name': 'RoBERTa Benchmark',
        'feature_extraction': 'None',
        'classifier': 'RoBERTa',
        'dimensionality_reduction': 'None',
        'dataset_pattern': 'ag-news-normal',
        'experiment_pattern': 'roberta_benchmark'
    },
    'A03': {
        'name': 'DistilBERT Benchmark',
        'feature_extraction': 'None',
        'classifier': 'DistilBERT',
        'dimensionality_reduction': 'None',
        'dataset_pattern': 'ag-news-normal',
        'experiment_pattern': 'distilbert_benchmark'
    },
    
    # Modified Scenario 2 (Skenario 2)
    'B01': {
        'name': 'BERT + Bi-LSTM + J.Su Whitening',
        'feature_extraction': 'BERT Attention Layer',
        'classifier': 'Custom Bi-LSTM',
        'dimensionality_reduction': 'J.Su Whitening',
        'dataset_pattern': 'ag-news-bert-whitening-svd',
        'experiment_pattern': 'ag-news-bert-whitening-svd-bilstm'
    },
    'B02': {
        'name': 'BERT + Bi-LSTM + PCA Whitening',
        'feature_extraction': 'BERT Attention Layer',
        'classifier': 'Custom Bi-LSTM',
        'dimensionality_reduction': 'PCA Whitening',
        'dataset_pattern': 'ag-news-bert-whitening-pca',
        'experiment_pattern': 'ag-news-bert-whitening-pca-bilstm'
    },
    'B03': {
        'name': 'BERT + Bi-LSTM + ZCA Whitening',
        'feature_extraction': 'BERT Attention Layer',
        'classifier': 'Custom Bi-LSTM',
        'dimensionality_reduction': 'ZCA Whitening',
        'dataset_pattern': 'ag-news-bert-whitening-zca',
        'experiment_pattern': 'ag-news-bert-whitening-zca-bilstm'
    },
    'B04': {
        'name': 'RoBERTa + Bi-LSTM + J.Su Whitening',
        'feature_extraction': 'RoBERTa Attention Layer',
        'classifier': 'Custom Bi-LSTM',
        'dimensionality_reduction': 'J.Su Whitening',
        'dataset_pattern': 'ag-news-roberta-whitening-svd',
        'experiment_pattern': 'ag-news-roberta-whitening-svd-bilstm'
    },
    'B05': {
        'name': 'RoBERTa + Bi-LSTM + PCA Whitening',
        'feature_extraction': 'RoBERTa Attention Layer',
        'classifier': 'Custom Bi-LSTM',
        'dimensionality_reduction': 'PCA Whitening',
        'dataset_pattern': 'ag-news-roberta-whitening-pca',
        'experiment_pattern': 'ag-news-roberta-whitening-pca-bilstm'
    },
    'B06': {
        'name': 'RoBERTa + Bi-LSTM + ZCA Whitening',
        'feature_extraction': 'RoBERTa Attention Layer',
        'classifier': 'Custom Bi-LSTM',
        'dimensionality_reduction': 'ZCA Whitening',
        'dataset_pattern': 'ag-news-roberta-whitening-zca',
        'experiment_pattern': 'ag-news-roberta-whitening-zca-bilstm'
    },
    
    # Modified Scenario 3 (Skenario 3)
    'C01': {
        'name': 'BERT + MLP + J.Su Whitening',
        'feature_extraction': 'BERT Attention Layer',
        'classifier': 'BERT Classifier (MLP)',
        'dimensionality_reduction': 'J.Su Whitening',
        'dataset_pattern': 'ag-news-bert-whitening-svd',
        'experiment_pattern': 'ag-news-bert-whitening-svd-mlp'
    },
    'C02': {
        'name': 'BERT + MLP + PCA Whitening',
        'feature_extraction': 'BERT Attention Layer',
        'classifier': 'BERT Classifier (MLP)',
        'dimensionality_reduction': 'PCA Whitening',
        'dataset_pattern': 'ag-news-bert-whitening-pca',
        'experiment_pattern': 'ag-news-bert-whitening-pca-mlp'
    },
    'C03': {
        'name': 'BERT + MLP + ZCA Whitening',
        'feature_extraction': 'BERT Attention Layer',
        'classifier': 'BERT Classifier (MLP)',
        'dimensionality_reduction': 'ZCA Whitening',
        'dataset_pattern': 'ag-news-bert-whitening-zca',
        'experiment_pattern': 'ag-news-bert-whitening-zca-mlp'
    },
    'C04': {
        'name': 'RoBERTa + MLP + J.Su Whitening',
        'feature_extraction': 'RoBERTa Attention Layer',
        'classifier': 'RoBERTa Classifier (MLP)',
        'dimensionality_reduction': 'J.Su Whitening',
        'dataset_pattern': 'ag-news-roberta-whitening-svd',
        'experiment_pattern': 'ag-news-roberta-whitening-svd-mlp'
    },
    'C05': {
        'name': 'RoBERTa + MLP + PCA Whitening',
        'feature_extraction': 'RoBERTa Attention Layer',
        'classifier': 'RoBERTa Classifier (MLP)',
        'dimensionality_reduction': 'PCA Whitening',
        'dataset_pattern': 'ag-news-roberta-whitening-pca',
        'experiment_pattern': 'ag-news-roberta-whitening-pca-mlp'
    },
    'C06': {
        'name': 'RoBERTa + MLP + ZCA Whitening',
        'feature_extraction': 'RoBERTa Attention Layer',
        'classifier': 'RoBERTa Classifier (MLP)',
        'dimensionality_reduction': 'ZCA Whitening',
        'dataset_pattern': 'ag-news-roberta-whitening-zca',
        'experiment_pattern': 'ag-news-roberta-whitening-zca-mlp'
    }
}


class EfficiencyAnalyzer:
    """Analyze and compare efficiency metrics across all 14 models"""
    
    def __init__(self, efficiency_dir='efficiency_analysis'):
        self.efficiency_dir = Path(efficiency_dir)
        self.raw_results_dir = self.efficiency_dir / 'raw_results'
        self.output_dir = self.efficiency_dir / 'comparison_analysis'
        self.output_dir.mkdir(exist_ok=True)
        
        self.all_data = None
        self.summary_data = None
        
    def load_raw_results(self):
        """Load all raw efficiency results from CSV files"""
        print("Loading raw efficiency results...")
        
        all_results = []
        
        # Iterate through all CSV files in raw_results directory
        for csv_file in self.raw_results_dir.glob('*.csv'):
            try:
                df = pd.read_csv(csv_file)
                
                # Extract model information from filename
                filename = csv_file.stem
                
                # Parse filename to get dataset and experiment info
                # Format: {dataset}_{experiment}_percent{percentage}.csv
                parts = filename.split('_percent')
                if len(parts) == 2:
                    dataset_experiment = parts[0]
                    percentage = int(parts[1])
                    
                    # Split dataset and experiment
                    # Format: {dataset}_{experiment}
                    exp_parts = dataset_experiment.rsplit('_', 1)
                    if len(exp_parts) == 2:
                        dataset = exp_parts[0]
                        experiment = exp_parts[1]
                        
                        # Add metadata
                        df['model_id'] = self._identify_model(dataset, experiment)
                        df['dataset'] = dataset
                        df['experiment'] = experiment
                        df['percentage'] = percentage
                        df['source_file'] = csv_file.name
                        
                        all_results.append(df)
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                continue
        
        if all_results:
            self.all_data = pd.concat(all_results, ignore_index=True)
            print(f"Loaded {len(self.all_data)} records from {len(all_results)} files")
        else:
            print("No data loaded!")
            
        return self.all_data
    
    def _identify_model(self, dataset, experiment):
        """Identify model ID based on dataset and experiment patterns"""
        for model_id, config in MODEL_CONFIG.items():
            if (config['dataset_pattern'] in dataset and 
                config['experiment_pattern'] in experiment):
                return model_id
        return 'UNKNOWN'
    
    def create_summary_table(self):
        """Create summary statistics for each model"""
        if self.all_data is None:
            self.load_raw_results()
        
        print("Creating summary table...")
        
        summaries = []
        
        for model_id, config in MODEL_CONFIG.items():
            # Filter data for this model
            model_data = self.all_data[self.all_data['model_id'] == model_id]
            
            if model_data.empty:
                print(f"Warning: No data found for {model_id} ({config['name']})")
                continue
            
            # Calculate statistics for each percentage
            for pct in sorted(model_data['percentage'].unique()):
                pct_data = model_data[model_data['percentage'] == pct]
                
                # Get total time and GPU usage
                total_row = pct_data[pct_data['ket'] == 'Total']
                train_row = pct_data[pct_data['ket'] == 'Train']
                test_row = pct_data[pct_data['ket'] == 'Test']
                
                if not total_row.empty:
                    summary = {
                        'model_id': model_id,
                        'model_name': config['name'],
                        'feature_extraction': config['feature_extraction'],
                        'classifier': config['classifier'],
                        'dimensionality_reduction': config['dimensionality_reduction'],
                        'percentage': pct,
                        'total_time': total_row['elapsed_time'].values[0],
                        'train_time': train_row['elapsed_time'].values[0] if not train_row.empty else None,
                        'test_time': test_row['elapsed_time'].values[0] if not test_row.empty else None,
                        'gpu_used': total_row['gpu_used'].values[0],
                        'scenario': self._get_scenario(model_id)
                    }
                    summaries.append(summary)
        
        self.summary_data = pd.DataFrame(summaries)
        print(f"Created summary with {len(self.summary_data)} records")
        
        return self.summary_data
    
    def _get_scenario(self, model_id):
        """Get scenario type based on model ID"""
        if model_id.startswith('A'):
            return 'Benchmark (Skenario 1)'
        elif model_id.startswith('B'):
            return 'Modified - Bi-LSTM (Skenario 2)'
        elif model_id.startswith('C'):
            return 'Modified - MLP (Skenario 3)'
        return 'Unknown'
    
    def generate_comparison_tables(self):
        """Generate various comparison tables"""
        if self.summary_data is None:
            self.create_summary_table()
        
        print("Generating comparison tables...")
        
        # 1. Overall comparison at 100% data
        full_data = self.summary_data[self.summary_data['percentage'] == 100].copy()
        full_data = full_data.sort_values('total_time')
        
        # 2. Comparison by scenario
        scenario_comparison = self.summary_data.groupby(['scenario', 'model_id', 'model_name']).agg({
            'total_time': 'mean',
            'gpu_used': 'mean'
        }).reset_index()
        
        # 3. Comparison by dimensionality reduction technique
        dim_reduction_comparison = self.summary_data[
            self.summary_data['dimensionality_reduction'] != 'None'
        ].groupby(['dimensionality_reduction', 'model_id', 'model_name']).agg({
            'total_time': 'mean',
            'gpu_used': 'mean'
        }).reset_index()
        
        # 4. Comparison by classifier type
        classifier_comparison = self.summary_data.groupby(['classifier', 'model_id', 'model_name']).agg({
            'total_time': 'mean',
            'gpu_used': 'mean'
        }).reset_index()
        
        # 5. Efficiency ranking (time per percentage)
        efficiency_ranking = self.summary_data.copy()
        efficiency_ranking['time_per_pct'] = efficiency_ranking['total_time'] / efficiency_ranking['percentage']
        efficiency_ranking = efficiency_ranking.sort_values('time_per_pct')
        
        # Save tables
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        full_data.to_csv(self.output_dir / f'full_comparison_100pct_{timestamp}.csv', index=False)
        scenario_comparison.to_csv(self.output_dir / f'scenario_comparison_{timestamp}.csv', index=False)
        dim_reduction_comparison.to_csv(self.output_dir / f'dim_reduction_comparison_{timestamp}.csv', index=False)
        classifier_comparison.to_csv(self.output_dir / f'classifier_comparison_{timestamp}.csv', index=False)
        efficiency_ranking.to_csv(self.output_dir / f'efficiency_ranking_{timestamp}.csv', index=False)
        
        print(f"Comparison tables saved to {self.output_dir}")
        
        return {
            'full_data': full_data,
            'scenario_comparison': scenario_comparison,
            'dim_reduction_comparison': dim_reduction_comparison,
            'classifier_comparison': classifier_comparison,
            'efficiency_ranking': efficiency_ranking
        }
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        if self.summary_data is None:
            self.create_summary_table()
        
        print("Creating visualizations...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Total time comparison at 100% data
        fig, ax = plt.subplots(figsize=(14, 8))
        full_data = self.summary_data[self.summary_data['percentage'] == 100].copy()
        full_data = full_data.sort_values('total_time')
        
        colors = ['#FF6B6B' if 'Benchmark' in s else '#4ECDC4' if 'Bi-LSTM' in s else '#45B7D1' 
                 for s in full_data['scenario']]
        
        bars = ax.barh(full_data['model_name'], full_data['total_time'], color=colors)
        ax.set_xlabel('Total Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model', fontsize=12, fontweight='bold')
        ax.set_title('Total Training Time Comparison (100% Data)', fontsize=14, fontweight='bold', pad=20)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, full_data['total_time'])):
            ax.text(val + 50, bar.get_y() + bar.get_height()/2, 
                   f'{val:.1f}s', va='center', fontsize=9)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF6B6B', label='Benchmark'),
            Patch(facecolor='#4ECDC4', label='Modified - Bi-LSTM'),
            Patch(facecolor='#45B7D1', label='Modified - MLP')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'total_time_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. GPU memory usage comparison
        fig, ax = plt.subplots(figsize=(14, 8))
        full_data_sorted = full_data.sort_values('gpu_used', ascending=True)
        
        colors = ['#FF6B6B' if 'Benchmark' in s else '#4ECDC4' if 'Bi-LSTM' in s else '#45B7D1' 
                 for s in full_data_sorted['scenario']]
        
        bars = ax.barh(full_data_sorted['model_name'], full_data_sorted['gpu_used'], color=colors)
        ax.set_xlabel('GPU Memory Used (MB)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model', fontsize=12, fontweight='bold')
        ax.set_title('GPU Memory Usage Comparison (100% Data)', fontsize=14, fontweight='bold', pad=20)
        
        for i, (bar, val) in enumerate(zip(bars, full_data_sorted['gpu_used'])):
            ax.text(val + 10, bar.get_y() + bar.get_height()/2, 
                   f'{val:.0f} MB', va='center', fontsize=9)
        
        ax.legend(handles=legend_elements, loc='lower right')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'gpu_usage_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Time vs Data Percentage for all models
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for model_id, config in MODEL_CONFIG.items():
            model_data = self.summary_data[self.summary_data['model_id'] == model_id]
            if not model_data.empty:
                ax.plot(model_data['percentage'], model_data['total_time'], 
                       marker='o', label=config['name'], alpha=0.7)
        
        ax.set_xlabel('Data Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Training Time vs Data Percentage', fontsize=14, fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'time_vs_percentage_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Scenario comparison heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        scenario_pivot = self.summary_data.pivot_table(
            index='model_name', 
            columns='percentage', 
            values='total_time'
        )
        
        sns.heatmap(scenario_pivot, annot=True, fmt='.0f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Total Time (seconds)'}, ax=ax)
        ax.set_title('Training Time Heatmap by Model and Data Percentage', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Data Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'time_heatmap_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Dimensionality reduction technique comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        dim_data = self.summary_data[
            self.summary_data['dimensionality_reduction'] != 'None'
        ].copy()
        
        # Time comparison
        sns.boxplot(data=dim_data, x='dimensionality_reduction', y='total_time', ax=axes[0])
        axes[0].set_title('Training Time by Dimensionality Reduction Technique', 
                         fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Technique', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Total Time (seconds)', fontsize=11, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        
        # GPU comparison
        sns.boxplot(data=dim_data, x='dimensionality_reduction', y='gpu_used', ax=axes[1])
        axes[1].set_title('GPU Usage by Dimensionality Reduction Technique', 
                         fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Technique', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('GPU Memory (MB)', fontsize=11, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'dim_reduction_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Classifier type comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Time comparison
        sns.boxplot(data=self.summary_data, x='classifier', y='total_time', ax=axes[0])
        axes[0].set_title('Training Time by Classifier Type', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Classifier', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Total Time (seconds)', fontsize=11, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        
        # GPU comparison
        sns.boxplot(data=self.summary_data, x='classifier', y='gpu_used', ax=axes[1])
        axes[1].set_title('GPU Usage by Classifier Type', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Classifier', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('GPU Memory (MB)', fontsize=11, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'classifier_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 7. Efficiency scatter plot (Time vs GPU)
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for scenario in self.summary_data['scenario'].unique():
            scenario_data = self.summary_data[self.summary_data['scenario'] == scenario]
            ax.scatter(scenario_data['total_time'], scenario_data['gpu_used'], 
                      label=scenario, alpha=0.6, s=100)
        
        ax.set_xlabel('Total Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('GPU Memory (MB)', fontsize=12, fontweight='bold')
        ax.set_title('Efficiency Scatter Plot: Time vs GPU Usage', fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'efficiency_scatter_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {self.output_dir}")
    
    def generate_report(self):
        """Generate a comprehensive text report"""
        if self.summary_data is None:
            self.create_summary_table()
        
        print("Generating comprehensive report...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f'comparison_report_{timestamp}.txt'
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE EFFICIENCY ANALYSIS REPORT\n")
            f.write("14 Models Comparison\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 80 + "\n")
            
            full_data = self.summary_data[self.summary_data['percentage'] == 100].copy()
            full_data = full_data.sort_values('total_time')
            
            fastest = full_data.iloc[0]
            slowest = full_data.iloc[-1]
            avg_time = full_data['total_time'].mean()
            
            f.write(f"Total Models Analyzed: {len(full_data)}\n")
            f.write(f"Fastest Model: {fastest['model_name']} ({fastest['total_time']:.1f}s)\n")
            f.write(f"Slowest Model: {slowest['model_name']} ({slowest['total_time']:.1f}s)\n")
            f.write(f"Average Training Time: {avg_time:.1f}s\n")
            f.write(f"Time Difference: {slowest['total_time'] - fastest['total_time']:.1f}s ({((slowest['total_time']/fastest['total_time']-1)*100):.1f}%)\n\n")
            
            # GPU Summary
            full_data_gpu = full_data.sort_values('gpu_used')
            lowest_gpu = full_data_gpu.iloc[0]
            highest_gpu = full_data_gpu.iloc[-1]
            avg_gpu = full_data['gpu_used'].mean()
            
            f.write(f"Lowest GPU Usage: {lowest_gpu['model_name']} ({lowest_gpu['gpu_used']:.0f} MB)\n")
            f.write(f"Highest GPU Usage: {highest_gpu['model_name']} ({highest_gpu['gpu_used']:.0f} MB)\n")
            f.write(f"Average GPU Usage: {avg_gpu:.0f} MB\n\n")
            
            # Scenario Analysis
            f.write("\nSCENARIO ANALYSIS\n")
            f.write("-" * 80 + "\n")
            
            for scenario in sorted(self.summary_data['scenario'].unique()):
                scenario_data = self.summary_data[self.summary_data['scenario'] == scenario]
                scenario_full = scenario_data[scenario_data['percentage'] == 100]
                
                f.write(f"\n{scenario}\n")
                f.write(f"  Number of Models: {len(scenario_full)}\n")
                f.write(f"  Average Time: {scenario_full['total_time'].mean():.1f}s\n")
                f.write(f"  Average GPU: {scenario_full['gpu_used'].mean():.0f} MB\n")
                f.write(f"  Fastest: {scenario_full.loc[scenario_full['total_time'].idxmin(), 'model_name']} ({scenario_full['total_time'].min():.1f}s)\n")
                f.write(f"  Slowest: {scenario_full.loc[scenario_full['total_time'].idxmax(), 'model_name']} ({scenario_full['total_time'].max():.1f}s)\n")
            
            # Dimensionality Reduction Analysis
            f.write("\n\nDIMENSIONALITY REDUCTION TECHNIQUE ANALYSIS\n")
            f.write("-" * 80 + "\n")
            
            dim_data = self.summary_data[self.summary_data['dimensionality_reduction'] != 'None']
            for dim in sorted(dim_data['dimensionality_reduction'].unique()):
                dim_subset = dim_data[dim_data['dimensionality_reduction'] == dim]
                dim_full = dim_subset[dim_subset['percentage'] == 100]
                
                f.write(f"\n{dim}\n")
                f.write(f"  Number of Models: {len(dim_full)}\n")
                f.write(f"  Average Time: {dim_full['total_time'].mean():.1f}s\n")
                f.write(f"  Average GPU: {dim_full['gpu_used'].mean():.0f} MB\n")
            
            # Classifier Analysis
            f.write("\n\nCLASSIFIER TYPE ANALYSIS\n")
            f.write("-" * 80 + "\n")
            
            for classifier in sorted(self.summary_data['classifier'].unique()):
                classifier_data = self.summary_data[self.summary_data['classifier'] == classifier]
                classifier_full = classifier_data[classifier_data['percentage'] == 100]
                
                f.write(f"\n{classifier}\n")
                f.write(f"  Number of Models: {len(classifier_full)}\n")
                f.write(f"  Average Time: {classifier_full['total_time'].mean():.1f}s\n")
                f.write(f"  Average GPU: {classifier_full['gpu_used'].mean():.0f} MB\n")
            
            # Detailed Model Rankings
            f.write("\n\nDETAILED MODEL RANKINGS (100% Data)\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Rank':<6} {'Model ID':<8} {'Model Name':<40} {'Time (s)':<12} {'GPU (MB)':<10}\n")
            f.write("-" * 80 + "\n")
            
            for idx, (_, row) in enumerate(full_data.iterrows(), 1):
                f.write(f"{idx:<6} {row['model_id']:<8} {row['model_name']:<40} {row['total_time']:<12.1f} {row['gpu_used']:<10.0f}\n")
            
            # Data Efficiency Analysis
            f.write("\n\nDATA EFFICIENCY ANALYSIS\n")
            f.write("-" * 80 + "\n")
            f.write("Time per 1% of data (lower is better):\n\n")
            
            efficiency = self.summary_data.copy()
            efficiency['time_per_pct'] = efficiency['total_time'] / efficiency['percentage']
            efficiency_avg = efficiency.groupby('model_id').agg({
                'model_name': 'first',
                'time_per_pct': 'mean'
            }).sort_values('time_per_pct')
            
            f.write(f"{'Rank':<6} {'Model ID':<8} {'Model Name':<40} {'Time/% (s)':<12}\n")
            f.write("-" * 80 + "\n")
            
            for idx, (model_id, row) in enumerate(efficiency_avg.iterrows(), 1):
                f.write(f"{idx:<6} {model_id:<8} {row['model_name']:<40} {row['time_per_pct']:<12.2f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"Report saved to {report_file}")
        return report_file
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("=" * 80)
        print("STARTING COMPREHENSIVE EFFICIENCY ANALYSIS")
        print("=" * 80)
        
        # Load data
        self.load_raw_results()
        
        # Create summary
        self.create_summary_table()
        
        # Generate comparison tables
        tables = self.generate_comparison_tables()
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate report
        report_file = self.generate_report()
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"\nAll results saved to: {self.output_dir}")
        print(f"Report: {report_file}")
        
        return {
            'summary_data': self.summary_data,
            'comparison_tables': tables,
            'output_dir': self.output_dir,
            'report_file': report_file
        }


def main():
    """Main execution function"""
    # Create analyzer instance
    analyzer = EfficiencyAnalyzer(efficiency_dir='efficiency_analysis')
    
    # Run full analysis
    results = analyzer.run_full_analysis()
    
    print("\nAnalysis completed successfully!")
    print(f"Check {results['output_dir']} for all outputs")


if __name__ == "__main__":
    main()
