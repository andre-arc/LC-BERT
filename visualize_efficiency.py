"""
Visualize Efficiency Analysis Results

This script creates visualizations from aggregated efficiency data to help
compare performance across different experiments and subset percentages.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime


# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_latest_aggregated_results(base_dir='efficiency_analysis'):
    """
    Load the most recent aggregated results file.

    Args:
        base_dir: Directory containing aggregated results

    Returns:
        DataFrame with aggregated results
    """
    base_path = Path(base_dir)

    # Find all aggregated result files
    result_files = list(base_path.glob('all_efficiency_results_*.csv'))

    if not result_files:
        print(f"No aggregated results found in {base_dir}")
        print("Please run aggregate_efficiency_results.py first")
        return None

    # Get the most recent file
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading: {latest_file}")

    return pd.read_csv(latest_file)


def plot_time_by_percentage(df, output_dir='efficiency_analysis/plots'):
    """
    Plot elapsed time vs subset percentage for different experiments.

    Args:
        df: DataFrame with efficiency data
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    if 'percentage' not in df.columns or 'elapsed_time' not in df.columns:
        print("Required columns 'percentage' and 'elapsed_time' not found")
        return

    # Aggregate by percentage and experiment
    time_by_pct = df.groupby(['dataset', 'experiment', 'percentage'])['elapsed_time'].sum().reset_index()

    # Create plot for each dataset
    datasets = time_by_pct['dataset'].unique()

    for dataset in datasets:
        dataset_df = time_by_pct[time_by_pct['dataset'] == dataset]

        plt.figure(figsize=(12, 6))
        for experiment in dataset_df['experiment'].unique():
            exp_df = dataset_df[dataset_df['experiment'] == experiment]
            plt.plot(exp_df['percentage'], exp_df['elapsed_time'],
                    marker='o', label=experiment, linewidth=2)

        plt.xlabel('Subset Percentage (%)', fontsize=12)
        plt.ylabel('Total Elapsed Time (seconds)', fontsize=12)
        plt.title(f'Training Time vs Data Subset Size\n{dataset}', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = f"{output_dir}/time_vs_percentage_{dataset.replace('/', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()


def plot_gpu_usage(df, output_dir='efficiency_analysis/plots'):
    """
    Plot GPU memory usage for different experiments.

    Args:
        df: DataFrame with efficiency data
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    if 'gpu_used' not in df.columns:
        print("Column 'gpu_used' not found")
        return

    # Get max GPU usage by experiment
    gpu_by_exp = df.groupby(['dataset', 'experiment'])['gpu_used'].max().reset_index()

    # Create plot for each dataset
    datasets = gpu_by_exp['dataset'].unique()

    for dataset in datasets:
        dataset_df = gpu_by_exp[gpu_by_exp['dataset'] == dataset]

        plt.figure(figsize=(10, 6))
        plt.barh(dataset_df['experiment'], dataset_df['gpu_used'])
        plt.xlabel('Max GPU Memory Usage (MB)', fontsize=12)
        plt.ylabel('Experiment', fontsize=12)
        plt.title(f'GPU Memory Usage by Experiment\n{dataset}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        filename = f"{output_dir}/gpu_usage_{dataset.replace('/', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()


def plot_time_breakdown(df, output_dir='efficiency_analysis/plots'):
    """
    Plot time breakdown by phase (Dataloader, Train, Test) for experiments.

    Args:
        df: DataFrame with efficiency data
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    if 'ket' not in df.columns or 'elapsed_time' not in df.columns:
        print("Required columns 'ket' and 'elapsed_time' not found")
        return

    # Filter out 'Total' rows
    df_filtered = df[df['ket'] != 'Total'].copy()

    # Get average time by phase and experiment
    time_breakdown = df_filtered.groupby(['dataset', 'experiment', 'ket'])['elapsed_time'].mean().reset_index()

    # Create plot for each dataset
    datasets = time_breakdown['dataset'].unique()

    for dataset in datasets:
        dataset_df = time_breakdown[time_breakdown['dataset'] == dataset]

        # Pivot for stacked bar chart
        pivot_df = dataset_df.pivot(index='experiment', columns='ket', values='elapsed_time')

        plt.figure(figsize=(12, 6))
        pivot_df.plot(kind='barh', stacked=True, ax=plt.gca(), width=0.7)
        plt.xlabel('Average Elapsed Time (seconds)', fontsize=12)
        plt.ylabel('Experiment', fontsize=12)
        plt.title(f'Time Breakdown by Phase\n{dataset}', fontsize=14, fontweight='bold')
        plt.legend(title='Phase', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        filename = f"{output_dir}/time_breakdown_{dataset.replace('/', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()


def plot_efficiency_comparison(df, output_dir='efficiency_analysis/plots'):
    """
    Create a comprehensive comparison plot showing time and GPU usage.

    Args:
        df: DataFrame with efficiency data
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get total time and max GPU for each experiment
    summary = df.groupby(['dataset', 'experiment']).agg({
        'elapsed_time': 'sum',
        'gpu_used': 'max'
    }).reset_index()

    # Create scatter plot
    datasets = summary['dataset'].unique()

    for dataset in datasets:
        dataset_df = summary[summary['dataset'] == dataset]

        plt.figure(figsize=(10, 6))
        plt.scatter(dataset_df['elapsed_time'], dataset_df['gpu_used'],
                   s=200, alpha=0.6, c=range(len(dataset_df)), cmap='viridis')

        # Add labels for each point
        for idx, row in dataset_df.iterrows():
            plt.annotate(row['experiment'],
                        (row['elapsed_time'], row['gpu_used']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8)

        plt.xlabel('Total Time (seconds)', fontsize=12)
        plt.ylabel('Max GPU Usage (MB)', fontsize=12)
        plt.title(f'Efficiency Comparison: Time vs GPU\n{dataset}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = f"{output_dir}/efficiency_scatter_{dataset.replace('/', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()


def plot_scaling_efficiency(df, output_dir='efficiency_analysis/plots'):
    """
    Plot how efficiently experiments scale with data size.

    Args:
        df: DataFrame with efficiency data
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    if 'percentage' not in df.columns or 'elapsed_time' not in df.columns:
        print("Required columns not found for scaling analysis")
        return

    # Calculate time per percentage point
    time_by_pct = df.groupby(['dataset', 'experiment', 'percentage'])['elapsed_time'].sum().reset_index()
    time_by_pct['time_per_percent'] = time_by_pct['elapsed_time'] / time_by_pct['percentage']

    datasets = time_by_pct['dataset'].unique()

    for dataset in datasets:
        dataset_df = time_by_pct[time_by_pct['dataset'] == dataset]

        plt.figure(figsize=(12, 6))
        for experiment in dataset_df['experiment'].unique():
            exp_df = dataset_df[dataset_df['experiment'] == experiment]
            plt.plot(exp_df['percentage'], exp_df['time_per_percent'],
                    marker='o', label=experiment, linewidth=2)

        plt.xlabel('Subset Percentage (%)', fontsize=12)
        plt.ylabel('Time per Percentage Point (sec/%)', fontsize=12)
        plt.title(f'Scaling Efficiency\n{dataset}', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = f"{output_dir}/scaling_efficiency_{dataset.replace('/', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()


def generate_all_plots(df, output_dir='efficiency_analysis/plots'):
    """
    Generate all available plots.

    Args:
        df: DataFrame with efficiency data
        output_dir: Directory to save plots
    """
    print("\n=== Generating Visualizations ===\n")

    plot_time_by_percentage(df, output_dir)
    plot_gpu_usage(df, output_dir)
    plot_time_breakdown(df, output_dir)
    plot_efficiency_comparison(df, output_dir)
    plot_scaling_efficiency(df, output_dir)

    print(f"\n=== All plots saved to {output_dir} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize efficiency analysis results')
    parser.add_argument('--input-dir', type=str, default='efficiency_analysis',
                        help='Directory containing aggregated results (default: efficiency_analysis)')
    parser.add_argument('--output-dir', type=str, default='efficiency_analysis/plots',
                        help='Output directory for plots (default: efficiency_analysis/plots)')
    parser.add_argument('--input-file', type=str, default=None,
                        help='Specific input file to use (optional)')

    args = parser.parse_args()

    # Load data
    if args.input_file:
        print(f"Loading: {args.input_file}")
        df = pd.read_csv(args.input_file)
    else:
        df = load_latest_aggregated_results(args.input_dir)

    if df is None or df.empty:
        print("No data to visualize")
        exit(1)

    # Generate all plots
    generate_all_plots(df, args.output_dir)

    print("\nVisualization complete!")
