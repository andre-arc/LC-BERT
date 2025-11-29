"""
Aggregate Efficiency Analysis Results

This script collects all summary_efficiency.csv files from the save directory
and aggregates them into a single comprehensive report for easy comparison.
"""

import os
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime


def find_efficiency_files(base_dir='save', pattern='summary_efficiency*.csv'):
    """
    Recursively find all efficiency summary files in the save directory.

    Args:
        base_dir: Base directory to search (default: 'save')
        pattern: File pattern to match (default: 'summary_efficiency*.csv')

    Returns:
        List of Path objects for found files
    """
    base_path = Path(base_dir)
    return list(base_path.rglob(pattern))


def extract_metadata_from_path(file_path, base_dir='save'):
    """
    Extract dataset and experiment name from file path.

    Args:
        file_path: Path object for the efficiency file
        base_dir: Base directory to remove from path

    Returns:
        Dictionary with dataset and experiment metadata
    """
    parts = file_path.relative_to(base_dir).parts

    metadata = {
        'dataset': parts[0] if len(parts) > 0 else 'unknown',
        'experiment': parts[1] if len(parts) > 1 else 'unknown',
        'file_path': str(file_path)
    }

    return metadata


def load_and_augment_efficiency_data(file_path, base_dir='save'):
    """
    Load efficiency CSV and add metadata columns.

    Args:
        file_path: Path to the efficiency CSV file
        base_dir: Base directory for metadata extraction

    Returns:
        DataFrame with efficiency data and metadata
    """
    try:
        df = pd.read_csv(file_path)
        metadata = extract_metadata_from_path(Path(file_path), base_dir)

        # Add metadata columns
        for key, value in metadata.items():
            df[key] = value

        # Extract percentage from filename if present
        filename = Path(file_path).stem
        if '_' in filename:
            parts = filename.split('_')
            if len(parts) > 2 and parts[-1].isdigit():
                df['subset_percentage_file'] = int(parts[-1])

        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def aggregate_efficiency_results(base_dir='save', output_dir='efficiency_analysis'):
    """
    Aggregate all efficiency results into comprehensive reports.

    Args:
        base_dir: Base directory containing experiment results
        output_dir: Directory to save aggregated results

    Returns:
        Dictionary of aggregated DataFrames
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all efficiency files
    print(f"Searching for efficiency files in {base_dir}...")
    efficiency_files = find_efficiency_files(base_dir)
    print(f"Found {len(efficiency_files)} efficiency files")

    if not efficiency_files:
        print("No efficiency files found!")
        return None

    # Load and combine all data
    all_data = []
    for file_path in efficiency_files:
        df = load_and_augment_efficiency_data(file_path, base_dir)
        if df is not None:
            all_data.append(df)

    if not all_data:
        print("No data could be loaded!")
        return None

    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save comprehensive report
    output_path = os.path.join(output_dir, f'all_efficiency_results_{timestamp}.csv')
    combined_df.to_csv(output_path, index=False)
    print(f"\nSaved comprehensive report to: {output_path}")

    # Generate summary statistics by experiment
    if 'ket' in combined_df.columns and 'elapsed_time' in combined_df.columns:
        summary_by_experiment = combined_df.groupby(['dataset', 'experiment', 'ket']).agg({
            'elapsed_time': ['mean', 'std', 'min', 'max', 'count'],
            'gpu_used': ['mean', 'max']
        }).round(2)

        summary_path = os.path.join(output_dir, f'summary_by_experiment_{timestamp}.csv')
        summary_by_experiment.to_csv(summary_path)
        print(f"Saved experiment summary to: {summary_path}")

    # Generate summary by percentage (if available)
    if 'percentage' in combined_df.columns:
        summary_by_percentage = combined_df.groupby(['dataset', 'experiment', 'percentage']).agg({
            'elapsed_time': 'sum',
            'gpu_used': 'max'
        }).round(2)

        percentage_path = os.path.join(output_dir, f'summary_by_percentage_{timestamp}.csv')
        summary_by_percentage.to_csv(percentage_path)
        print(f"Saved percentage summary to: {percentage_path}")

    # Generate comparison table (total time by experiment)
    if 'ket' in combined_df.columns:
        total_time_df = combined_df[combined_df['ket'] == 'Total'].copy() if 'Total' in combined_df['ket'].values else combined_df.groupby(['dataset', 'experiment']).agg({'elapsed_time': 'sum'}).reset_index()

        if not total_time_df.empty:
            comparison_path = os.path.join(output_dir, f'experiment_comparison_{timestamp}.csv')
            total_time_df.to_csv(comparison_path, index=False)
            print(f"Saved experiment comparison to: {comparison_path}")

    print(f"\n=== Aggregation Complete ===")
    print(f"Total experiments analyzed: {combined_df['experiment'].nunique()}")
    print(f"Total datasets: {combined_df['dataset'].nunique()}")
    print(f"All results saved to: {output_dir}")

    return {
        'combined': combined_df,
        'output_dir': output_dir,
        'timestamp': timestamp
    }


def print_summary_stats(results):
    """
    Print summary statistics to console.

    Args:
        results: Dictionary returned by aggregate_efficiency_results
    """
    if results is None:
        return

    df = results['combined']

    print("\n=== Quick Summary ===")

    # Experiments by dataset
    print("\nExperiments by Dataset:")
    print(df.groupby('dataset')['experiment'].nunique().sort_values(ascending=False))

    # Average time by dataset
    if 'elapsed_time' in df.columns:
        print("\nAverage Total Time by Dataset (seconds):")
        avg_time = df.groupby('dataset')['elapsed_time'].sum().sort_values(ascending=False)
        print(avg_time.apply(lambda x: f"{x:.2f}"))

    # GPU usage by dataset
    if 'gpu_used' in df.columns:
        print("\nMax GPU Usage by Dataset (MB):")
        max_gpu = df.groupby('dataset')['gpu_used'].max().sort_values(ascending=False)
        print(max_gpu.apply(lambda x: f"{x:.2f}"))


def generate_comparison_report(base_dir='save', output_dir='efficiency_analysis',
                                experiments=None, metric='elapsed_time'):
    """
    Generate a focused comparison report for specific experiments.

    Args:
        base_dir: Base directory containing results
        output_dir: Output directory for reports
        experiments: List of experiment names to compare (None = all)
        metric: Metric to compare ('elapsed_time' or 'gpu_used')
    """
    results = aggregate_efficiency_results(base_dir, output_dir)

    if results is None:
        return

    df = results['combined']

    # Filter experiments if specified
    if experiments:
        df = df[df['experiment'].isin(experiments)]

    # Create pivot table for comparison
    if 'percentage' in df.columns and metric in df.columns:
        pivot = df.pivot_table(
            index='percentage',
            columns=['dataset', 'experiment'],
            values=metric,
            aggfunc='sum'
        )

        comparison_path = os.path.join(output_dir, f'comparison_{metric}_{results["timestamp"]}.csv')
        pivot.to_csv(comparison_path)
        print(f"\nSaved {metric} comparison to: {comparison_path}")

        return pivot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate efficiency analysis results')
    parser.add_argument('--base-dir', type=str, default='save',
                        help='Base directory containing experiment results (default: save)')
    parser.add_argument('--output-dir', type=str, default='efficiency_analysis',
                        help='Output directory for aggregated results (default: efficiency_analysis)')
    parser.add_argument('--experiments', type=str, nargs='+', default=None,
                        help='Specific experiments to compare (optional)')
    parser.add_argument('--metric', type=str, default='elapsed_time',
                        choices=['elapsed_time', 'gpu_used'],
                        help='Metric to compare (default: elapsed_time)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed summary statistics')

    args = parser.parse_args()

    # Run aggregation
    results = aggregate_efficiency_results(args.base_dir, args.output_dir)

    # Print summary if verbose
    if args.verbose and results:
        print_summary_stats(results)

    # Generate comparison report if experiments specified
    if args.experiments:
        generate_comparison_report(
            args.base_dir,
            args.output_dir,
            args.experiments,
            args.metric
        )
