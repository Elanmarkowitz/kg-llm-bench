# results_analyzer.py

import json
import os
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict, List
import seaborn as sns
import matplotlib.pyplot as plt

COLOR_MAP = {
    # Grouping by provider
    'gpt-4o-mini': '#4a4e69',  # Dark Gray/Blue
    
    # Amazon models
    'nova-lite': '#ffbb78',  # Light Orange
    'nova-pro': '#ff7f0e',  # Darker Orange
    
    # Anthropic models
    'claude-3.5-sonnet-v2': '#d62728',  # Red
    
    # Meta models
    'llama3.2-1b-instruct': '#b3a0d6',  # Light Purple
    'llama3.3-70b-instruct': '#9467bd',  # Purple

    'gemini-1.5-flash': '#4285F4',  # Google Blue
    
    # Add more models and colors as needed
}

MODEL_NAME_MAP = {
    'gemini-1.5-flash': 'gemini-1.5-flash',
    'gpt-4o-mini': 'gpt-4o-mini',
    'us.amazon.nova-lite-v1:0': 'nova-lite',
    'us.amazon.nova-pro-v1:0': 'nova-pro',
    'us.anthropic.claude-3-5-sonnet-20241022-v2:0': 'claude-3.5-sonnet-v2',
    'us.meta.llama3-2-1b-instruct-v1:0': 'llama3.2-1b-instruct',
    'us.meta.llama3-3-70b-instruct-v1:0': 'llama3.3-70b-instruct'
}

# Format name mapping
FORMAT_NAME_MAP = {
    'list_of_edges': 'List of Edges',
    'structured_json': 'Structured JSON',
    'structured_yaml': 'Structured YAML',
    'rdf_turtle3': 'RDF Turtle',
    'json_ld3': 'JSON-LD'
}

TASKS = [
    "ShortestPath",
    "TripleRetrieval",
    "HighestDegreeNode",
    "AggByRelation",
    "AggNeighborProperties"
]

class ResultsAnalyzer:
    def __init__(self, results_dir: str = "benchmark_data"):
        self.results_dir = Path(results_dir)
        self.results_data = []
        
    def load_results(self):
        """Load all results files from the results directory"""
        print(f"\nSearching for results in: {self.results_dir}")
        task_dirs = list(self.results_dir.glob("*"))
        task_dirs = [d for d in task_dirs if d.is_dir()]
        print(f"Found {len(task_dirs)} task directories")
        
        for task_dir in task_dirs:
            if task_dir.name not in TASKS:
                continue
            print(f"\nProcessing task: {task_dir.name}")
            format_dirs = list(task_dir.glob("*"))
            format_dirs = [d for d in format_dirs if d.is_dir()]
            print(f"Found {len(format_dirs)} format directories")
            
            for format_dir in format_dirs:
                if not format_dir.is_dir():
                    continue
                
                # Split format name and check for pseudonymization
                format_parts = format_dir.name.split('-')
                format_name = format_parts[0]
                is_pseudo = len(format_parts) > 1 and format_parts[1] == "pseudo"
                print(f"\n  Format: {format_name} (Pseudonymized: {is_pseudo})")
                    
                llm_dirs = list(format_dir.glob("*"))
                llm_dirs = [d for d in llm_dirs if d.is_dir()]
                print(f"  Found {len(llm_dirs)} model directories")
                
                for llm_dir in llm_dirs:
                    if not llm_dir.is_dir():
                        continue
                        
                    results_files = list(llm_dir.glob("small_results.json"))
                    if results_files:
                        print(f"    Processing results for model: {llm_dir.name}")
                        
                    for results_file in results_files:
                        print(f"      Loading: {results_file}")
                        with open(results_file) as f:
                            results = json.load(f)
                            
                            # Calculate metrics
                            scores = [r.get('score', 0) for r in results if r is not None]
                            avg_score = np.mean(scores) if scores else 0
                            std_score = np.std(scores) if scores else 0
                            
                            # Calculate token usage metrics
                            input_tokens = [r['usage_tokens']['prompt_tokens'] for r in results if r is not None]
                            avg_input_tokens = np.mean(input_tokens) if input_tokens else 0
                            std_input_tokens = np.std(input_tokens) if input_tokens else 0
                            
                            print(f"      Found {len(scores)} examples, avg score: {avg_score:.3f}")
                            
                            self.results_data.append({
                                'task': task_dir.stem,
                                'format': format_name,
                                'model': llm_dir.name,
                                'pseudonymized': is_pseudo,
                                'avg_score': avg_score,
                                'std_score': std_score,
                                'num_examples': len(scores),
                                'avg_input_tokens': avg_input_tokens,
                                'std_input_tokens': std_input_tokens
                            })

    def _get_ordered_tasks_and_formats(self, df: pd.DataFrame):
        """Helper method to get tasks and formats ordered by overall performance"""
        # Calculate average performance for each task
        task_performance = df.groupby('task')['avg_score'].mean().sort_values(ascending=False)
        ordered_tasks = task_performance.index.tolist()
        
        # Calculate average performance for each format
        format_performance = df.groupby('format')['avg_score'].mean().sort_values(ascending=False)
        ordered_formats = format_performance.index.tolist()
        
        return ordered_tasks, ordered_formats

    def plot_heatmap(self, output_file: str = "figs/results_heatmap.pdf", normalized_output_file: str = "figs/results_heatmap_normalized.pdf"):
        """Generate heatmap of results and a normalized version, both overall and per-model"""
        df = pd.DataFrame(self.results_data)
        
        # Map model names
        df['model'] = df['model'].map(MODEL_NAME_MAP)
        
        # Get ordered tasks and formats from raw data first
        ordered_tasks, ordered_formats = self._get_ordered_tasks_and_formats(df)
        
        # Average across pseudonymized and non-pseudonymized results
        df_avg = df.groupby(['task', 'format', 'model'])['avg_score'].mean().reset_index()
        
        # Generate overall heatmaps
        self._generate_single_heatmap(df_avg, output_file, normalized_output_file, 
                                    title_prefix="Overall",
                                    ordered_tasks=ordered_tasks,
                                    ordered_formats=ordered_formats)
        
        # Generate per-model heatmaps
        for model in df_avg['model'].unique():
            model_df = df_avg[df_avg['model'] == model]
            model_name = model  # Already mapped
            model_output = output_file.replace('.pdf', f'_{model_name}.pdf')
            model_normalized_output = normalized_output_file.replace('.pdf', f'_{model_name}.pdf')
            self._generate_single_heatmap(model_df, model_output, model_normalized_output, 
                                        title_prefix=f"Model: {model_name}",
                                        ordered_tasks=ordered_tasks,
                                        ordered_formats=ordered_formats)
    
    def _generate_single_heatmap(self, df: pd.DataFrame, output_file: str, normalized_output_file: str, 
                                title_prefix: str = "", ordered_tasks=None, ordered_formats=None):
        """Helper method to generate a pair of heatmaps (regular and normalized) for the given dataframe"""
        pivot = pd.pivot_table(
            df,
            values='avg_score',
            index='task',
            columns='format',
            aggfunc='mean'
        )
        
        # Reorder rows and columns if provided
        if ordered_tasks is not None:
            pivot = pivot.reindex(ordered_tasks)
        if ordered_formats is not None:
            pivot = pivot.reindex(columns=ordered_formats)
        
        # Regular heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pivot, 
            annot=True, 
            fmt='.3f',
            cmap='Greens',
            vmin=0,
            vmax=1,
            annot_kws={'size': 10},  # Increased font size
            cbar_kws={'label': 'Score'},
            linewidths=0.5,  # Adjusted to reduce horizontal margin
            linecolor='white'  # Optional: to make grid lines visible
        )
        
        plt.title(f"{title_prefix} Raw Heatmap", pad=20)  # Updated title
        plt.xticks(rotation=45, ha='right', fontsize=10)  # Diagonalize and increase font size
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Normalized heatmap (each row scaled independently)
        plt.figure(figsize=(12, 8))
        normalized_pivot = pivot.copy()
        for idx in normalized_pivot.index:
            row = normalized_pivot.loc[idx]
            min_val = row.min()
            max_val = row.max()
            if max_val > min_val:  # Avoid division by zero
                normalized_pivot.loc[idx] = (row - min_val) / (max_val - min_val)
        
        sns.heatmap(
            normalized_pivot,
            annot=pivot.values,  # Show original values but use normalized colors
            fmt='.3f',
            cmap='Greens',
            vmin=0,
            vmax=1,
            annot_kws={'size': 10},  # Increased font size
            cbar_kws={'label': 'Normalized Score'},
            linewidths=0.5,  # Adjusted to reduce horizontal margin
            linecolor='white'  # Optional: to make grid lines visible
        )
        
        plt.title(f"{title_prefix} Row Normalized Heatmap", pad=20)  # Updated title
        plt.xticks(rotation=45, ha='right', fontsize=10)  # Diagonalize and increase font size
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.savefig(normalized_output_file, bbox_inches='tight', dpi=300)
        plt.close()

    def generate_latex_table(self, output_file: str = "figs/results_summary.tex"):
        """Generate LaTeX table with format x task matrix, subdividing each task into normal/pseudo"""
        df = pd.DataFrame(self.results_data)
        
        # Generate overall table
        self._generate_single_latex_table(df, output_file)
        
        # Generate per-model tables
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            model_name = model.split('/')[-1]  # Get last part of model path
            model_output = output_file.replace('.tex', f'_{model_name}.tex')
            self._generate_single_latex_table(model_df, model_output, model_name)
    
    def _generate_single_latex_table(self, df: pd.DataFrame, output_file: str, model_name: str = None):
        """Helper method to generate a LaTeX table for the given dataframe"""
        # Create pivot table with tasks as columns and formats as rows
        pivot = pd.pivot_table(
            df,
            values='avg_score',
            index=['format'],
            columns=['task', 'pseudonymized'],
            aggfunc='mean'
        )
        
        # Calculate overall scores for each format and pseudonymization combination
        overall_scores = df.groupby(['format', 'pseudonymized'])['avg_score'].mean().unstack()
        
        # Add overall scores as new columns
        pivot['Overall', False] = overall_scores[False]
        pivot['Overall', True] = overall_scores[True]
        
        # Sort columns to ensure consistent ordering
        pivot = pivot.reindex(sorted(pivot.columns.levels[0]), axis=1, level=0)
        
        # Format the values to 3 decimal places
        pivot = pivot.round(3)
        
        caption = "Knowledge Graph Format Performance by Task"
        if model_name:
            caption += f" for Model {model_name}"
        
        latex_str = pivot.to_latex(
            float_format="%.3f",
            bold_rows=True,
            multicolumn=True,
            multicolumn_format='c',
            caption=caption,
            label=f"tab:results{'_' + model_name if model_name else ''}"
        )
        
        # Add header clarification
        latex_str = latex_str.replace('& \\multicolumn{2}', '& \\multicolumn{2}{c}')
        
        with open(output_file, 'w') as f:
            f.write(latex_str)

    def print_summary(self):
        """Print summary statistics"""
        df = pd.DataFrame(self.results_data)
        
        # Print overall summary
        print("\n=== Overall Results Summary ===")
        self._print_single_summary(df)
        
        # Print per-model summary
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            model_name = model.split('/')[-1]  # Get last part of model path
            print(f"\n=== Results Summary for Model: {model_name} ===")
            self._print_single_summary(model_df)
    
    def _print_single_summary(self, df: pd.DataFrame):
        """Helper method to print summary statistics for the given dataframe"""
        print("\nScores by task:")
        print(df.groupby('task')['avg_score'].agg(['mean', 'max', 'std', 'count']))
        
        print("\nScores by format:")
        print(df.groupby('format')['avg_score'].agg(['mean', 'max', 'std', 'count']))
        
        print("\nPseudonymization impact:")
        print("\nMean scores:")
        print(df.groupby(['task', 'pseudonymized'])['avg_score'].mean().unstack())
        print("\nMax scores:")
        print(df.groupby(['task', 'pseudonymized'])['avg_score'].max().unstack())

    def plot_radar(self, output_file: str = "figs/model_radar_plot.pdf"):
        """Generate a radar plot comparing models across different tasks.
        Uses the best performing format for each task."""
        df = pd.DataFrame(self.results_data)
        
        # Map model names using MODEL_NAME_MAP
        df['model'] = df['model'].map(MODEL_NAME_MAP)
        
        # Get the best score for each task-model combination across formats
        best_scores = df.groupby(['task', 'model'])['avg_score'].max().reset_index()
        
        # Get ordered tasks
        ordered_tasks, _ = self._get_ordered_tasks_and_formats(df)
        
        # Pivot the data for plotting
        pivot_data = best_scores.pivot(index='model', columns='task', values='avg_score')
        
        # Reorder columns (tasks)
        pivot_data = pivot_data.reindex(columns=ordered_tasks)
        
        # Set up the angles for the radar plot
        tasks = pivot_data.columns
        num_tasks = len(tasks)
        angles = [n / float(num_tasks) * 2 * np.pi for n in range(num_tasks)]
        angles += angles[:1]  # Complete the circle
        
        # Create the plot with adjusted size for single column
        fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(projection='polar'))
        
        # Plot data
        for idx, model in enumerate(pivot_data.index):
            values = pivot_data.loc[model].values.flatten().tolist()
            values += values[:1]  # Complete the circle
            
            color = COLOR_MAP.get(model, '#000000')  # Default to black if model not found
            
            # Plot the model line
            ax.plot(angles, values, 'o-', linewidth=1.5, label=model, alpha=0.7, color=color, markersize=3)
            ax.fill(angles, values, color=color, alpha=0.1)
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(tasks, fontsize=8)  # Smaller font size for task labels
        
        # Adjust y-axis labels
        ax.tick_params(axis='y', labelsize=8)  # Smaller font size for score labels
        
        # Add legend with adjusted position and font size
        plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.4), fontsize=7)  # Moved downward
        
        plt.title("Model Performance Across Tasks\n(Best Format per Task)", pad=20, fontsize=10)
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_format_radar(self, output_file: str = "figs/format_radar_plot.pdf"):
        """Generate radar plots comparing format performance relative to the mean.
        Shows how each format performs relative to the average across formats for each task."""
        df = pd.DataFrame(self.results_data)
        
        # Calculate mean score for each task-model combination across formats
        task_model_means = df.groupby(['task', 'model'])['avg_score'].mean()
        
        # Calculate the difference from mean for each format
        df['score_diff'] = df.apply(
            lambda row: row['avg_score'] - task_model_means[row['task'], row['model']], 
            axis=1
        )
        
        # Get the average difference for each task-format combination
        format_diffs = df.groupby(['task', 'format'])['score_diff'].mean().reset_index()
        
        # Pivot the data for plotting
        pivot_data = format_diffs.pivot(index='format', columns='task', values='score_diff')
        
        # Set up the angles for the radar plot
        tasks = pivot_data.columns
        num_tasks = len(tasks)
        angles = [n / float(num_tasks) * 2 * np.pi for n in range(num_tasks)]
        angles += angles[:1]  # Complete the circle
        
        # Create the plot with adjusted size for single column
        fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(projection='polar'))
        
        # Plot data
        colors = plt.cm.Set2(np.linspace(0, 1, len(pivot_data.index)))
        for idx, (format_name, color) in enumerate(zip(pivot_data.index, colors)):
            values = pivot_data.loc[format_name].values.flatten().tolist()
            values += values[:1]  # Complete the circle
            
            # Plot the format line
            ax.plot(angles, values, 'o-', linewidth=1.5, label=format_name, color=color, alpha=0.7, markersize=3)
            ax.fill(angles, values, color=color, alpha=0.1)
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(tasks, fontsize=8)  # Smaller font size for task labels
        
        # Add a grid
        ax.grid(True)
        
        # Add zero line for reference
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Adjust y-axis labels
        ax.tick_params(axis='y', labelsize=8)  # Smaller font size for score labels
        
        # Add legend with adjusted position and font size
        plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.4), fontsize=7)
        
        plt.title("Format Performance Relative to Mean\nAcross Tasks and Models", pad=20, fontsize=10)
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_model_radar_relative(self, output_file: str = "figs/model_radar_relative_plot.pdf"):
        """Generate a radar plot comparing model performance relative to the mean.
        Shows how each model performs relative to the average across models for each task."""
        df = pd.DataFrame(self.results_data)
        
        # Map model names using MODEL_NAME_MAP
        df['model'] = df['model'].map(MODEL_NAME_MAP)
        
        # Get the best score for each task-model combination across formats
        best_scores = df.groupby(['task', 'model'])['avg_score'].max().reset_index()
        
        # Get ordered tasks
        ordered_tasks, _ = self._get_ordered_tasks_and_formats(df)
        
        # Calculate mean score for each task across all models
        task_means = best_scores.groupby('task')['avg_score'].mean()
        
        # Calculate the difference from mean for each model
        best_scores['score_diff'] = best_scores.apply(
            lambda row: row['avg_score'] - task_means[row['task']], 
            axis=1
        )
        
        # Pivot the data for plotting
        pivot_data = best_scores.pivot(index='model', columns='task', values='score_diff')
        
        # Reorder columns (tasks)
        pivot_data = pivot_data.reindex(columns=ordered_tasks)
        
        # Set up the angles for the radar plot
        tasks = pivot_data.columns
        num_tasks = len(tasks)
        angles = [n / float(num_tasks) * 2 * np.pi for n in range(num_tasks)]
        angles += angles[:1]  # Complete the circle
        
        # Create the plot with adjusted size for single column
        fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(projection='polar'))
        
        # Plot data
        for idx, model in enumerate(pivot_data.index):
            values = pivot_data.loc[model].values.flatten().tolist()
            values += values[:1]  # Complete the circle
            
            color = COLOR_MAP.get(model, '#000000')  # Default to black if model not found
            
            # Plot the model line
            ax.plot(angles, values, 'o-', linewidth=1.5, label=model, alpha=0.7, color=color, markersize=3)
            ax.fill(angles, values, color=color, alpha=0.1)
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(tasks, fontsize=8)  # Smaller font size for task labels
        
        # Add a grid
        ax.grid(True)
        
        # Add zero line for reference
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Adjust y-axis labels
        ax.tick_params(axis='y', labelsize=8)  # Smaller font size for score labels
        
        # Add legend with adjusted position and font size
        plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.4), fontsize=7)
        
        plt.title("Model Performance Relative to Mean\nAcross Tasks (Best Format per Task)", pad=20, fontsize=10)
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_model_radar_relative_best_format(self, output_file: str = "figs/model_radar_relative_best_format_plot.pdf"):
        """Generate a radar plot comparing model performance relative to the mean.
        Shows how each model performs relative to the average across models for each task,
        using the best overall format for each model."""
        df = pd.DataFrame(self.results_data)
        
        # Map model names
        df['model'] = df['model'].map(MODEL_NAME_MAP)
        
        # Calculate average score for each model-format combination across all tasks
        format_means = df.groupby(['model', 'format'])['avg_score'].mean()
        
        # Find the best format for each model
        best_formats = format_means.groupby('model').idxmax().apply(lambda x: x[1])
        
        # Filter data to only include each model's best format
        best_format_data = []
        for model in best_formats.index:
            model_data = df[
                (df['model'] == model) & 
                (df['format'] == best_formats[model])
            ]
            best_format_data.append(model_data)
        
        best_format_df = pd.concat(best_format_data)
        
        # Calculate mean score for each task across all models
        task_means = best_format_df.groupby('task')['avg_score'].mean()
        
        # Calculate the difference from mean for each model
        best_format_df['score_diff'] = best_format_df.apply(
            lambda row: row['avg_score'] - task_means[row['task']], 
            axis=1
        )
        
        # Create pivot table for plotting
        pivot_data = best_format_df.pivot_table(
            values='score_diff',
            index='model',
            columns='task',
            aggfunc='mean'
        )
        
        # Set up the angles for the radar plot
        tasks = pivot_data.columns
        num_tasks = len(tasks)
        angles = [n / float(num_tasks) * 2 * np.pi for n in range(num_tasks)]
        angles += angles[:1]  # Complete the circle
        
        # Create the plot with adjusted size for single column
        fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(projection='polar'))
        
        # Plot data
        for idx, model in enumerate(pivot_data.index):
            values = pivot_data.loc[model].values.flatten().tolist()
            values += values[:1]  # Complete the circle
            
            color = COLOR_MAP.get(model, '#000000')  # Default to black if model not found
            
            # Plot the model line
            ax.plot(angles, values, 'o-', linewidth=1.5, label=model, alpha=0.7, color=color, markersize=3)
            ax.fill(angles, values, color=color, alpha=0.1)
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(tasks, fontsize=8)  # Smaller font size for task labels
        
        # Add a grid
        ax.grid(True)
        
        # Add zero line for reference
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Add legend with adjusted position and font size
        plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.4), fontsize=7)
        
        plt.title("Model Performance Relative to Mean\nAcross Tasks (Using Best Overall Format per Model)", pad=20, fontsize=10)
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_radar_best_overall_format(self, output_file: str = "figs/model_radar_best_format_plot.pdf"):
        """Generate a radar plot comparing models across different tasks.
        Uses the single best performing format overall for each model."""
        df = pd.DataFrame(self.results_data)
        
        # Map model names
        df['model'] = df['model'].map(MODEL_NAME_MAP)
        
        # Calculate average score for each model-format combination across all tasks
        format_means = df.groupby(['model', 'format'])['avg_score'].mean()
        
        # Find the best format for each model
        best_formats = format_means.groupby('model').idxmax().apply(lambda x: x[1])
        
        # Filter data to only include each model's best format
        best_format_data = []
        for model in best_formats.index:
            model_data = df[
                (df['model'] == model) & 
                (df['format'] == best_formats[model])
            ]
            best_format_data.append(model_data)
        
        best_format_df = pd.concat(best_format_data)
        
        # Create pivot table for plotting
        pivot_data = best_format_df.pivot_table(
            values='avg_score',
            index='model',
            columns='task',
            aggfunc='mean'
        )
        
        # Set up the angles for the radar plot
        tasks = pivot_data.columns
        num_tasks = len(tasks)
        angles = [n / float(num_tasks) * 2 * np.pi for n in range(num_tasks)]
        angles += angles[:1]  # Complete the circle
        
        # Create the plot with adjusted size for single column
        fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(projection='polar'))
        
        # Plot data
        for idx, model in enumerate(pivot_data.index):
            values = pivot_data.loc[model].values.flatten().tolist()
            values += values[:1]  # Complete the circle
            
            color = COLOR_MAP.get(model, '#000000')  # Default to black if model not found
            
            # Plot the model line
            ax.plot(angles, values, 'o-', linewidth=1.5, label=model, alpha=0.7, color=color, markersize=3)
            ax.fill(angles, values, color=color, alpha=0.1)
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(tasks, fontsize=8)  # Smaller font size for task labels
        
        # Add a grid
        ax.grid(True)
        
        # Add legend with adjusted position and font size
        plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.4), fontsize=7)
        
        plt.title("Model Performance Across Tasks\n(Using Best Overall Format per Model)", pad=20, fontsize=10)
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_format_radar_relative_pct(self, output_file: str = "figs/format_radar_relative_pct_plot.pdf"):
        """Generate radar plots comparing format performance relative to the mean as percentages.
        Shows how each format performs relative to the average across formats for each task."""
        df = pd.DataFrame(self.results_data)
        
        # Map model names
        df['model'] = df['model'].map(MODEL_NAME_MAP)

        # Calculate mean score for each task-model combination across formats
        task_model_means = df.groupby(['task', 'model'])['avg_score'].mean()
        
        # Calculate the percentage difference from mean for each format
        df['score_pct_diff'] = df.apply(
            lambda row: (row['avg_score'] - task_model_means[row['task'], row['model']]) / task_model_means[row['task'], row['model']] * 100 
            if task_model_means[row['task'], row['model']] > 0 else 0, 
            axis=1
        )
        
        # Get the average difference for each task-format combination
        format_diffs = df.groupby(['task', 'format'])['score_pct_diff'].mean().reset_index()
        
        # Pivot the data for plotting
        pivot_data = format_diffs.pivot(index='format', columns='task', values='score_pct_diff')
        
        # Set up the angles for the radar plot
        tasks = pivot_data.columns
        num_tasks = len(tasks)
        angles = [n / float(num_tasks) * 2 * np.pi for n in range(num_tasks)]
        angles += angles[:1]  # Complete the circle
        
        # Create the plot with adjusted size for single column
        fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(projection='polar'))
        
        # Plot data
        colors = plt.cm.Set2(np.linspace(0, 1, len(pivot_data.index)))
        for idx, (format_name, color) in enumerate(zip(pivot_data.index, colors)):
            values = pivot_data.loc[format_name].values.flatten().tolist()
            values += values[:1]  # Complete the circle
            
            # Plot the format line
            ax.plot(angles, values, 'o-', linewidth=1.5, label=format_name, color=color, alpha=0.7, markersize=3)
            ax.fill(angles, values, color=color, alpha=0.1)
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(tasks, fontsize=8)  # Smaller font size for task labels
        
        # Add a grid
        ax.grid(True)
        
        # Add zero line for reference
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Add legend with adjusted position and font size
        plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.4), fontsize=7)
        
        plt.title("Format Performance Relative to Mean (%)\nAcross Tasks and Models", pad=20, fontsize=10)
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_model_radar_relative_pct(self, output_file: str = "figs/model_radar_relative_pct_plot.pdf"):
        """Generate a radar plot comparing model performance relative to the mean as percentages.
        Shows how each model performs relative to the average across models for each task."""
        df = pd.DataFrame(self.results_data)
        

        # Map model names
        df['model'] = df['model'].map(MODEL_NAME_MAP)

        # Get the best score for each task-model combination across formats
        best_scores = df.groupby(['task', 'model'])['avg_score'].max().reset_index()
        
        # Calculate mean score for each task across all models
        task_means = best_scores.groupby('task')['avg_score'].mean()
        
        # Calculate the percentage difference from mean for each model
        best_scores['score_pct_diff'] = best_scores.apply(
            lambda row: (row['avg_score'] - task_means[row['task']]) / task_means[row['task']] * 100
            if task_means[row['task']] > 0 else 0,
            axis=1
        )
        
        # Pivot the data for plotting
        pivot_data = best_scores.pivot(index='model', columns='task', values='score_pct_diff')
        
        # Set up the angles for the radar plot
        tasks = pivot_data.columns
        num_tasks = len(tasks)
        angles = [n / float(num_tasks) * 2 * np.pi for n in range(num_tasks)]
        angles += angles[:1]  # Complete the circle
        
        # Create the plot with adjusted size for single column
        fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(projection='polar'))
        
        # Plot data
        colors = plt.cm.Set2(np.linspace(0, 1, len(pivot_data.index)))
        for idx, (model, color) in enumerate(zip(pivot_data.index, colors)):
            values = pivot_data.loc[model].values.flatten().tolist()
            values += values[:1]  # Complete the circle
            
            color = COLOR_MAP.get(model, '#000000')  # Default to black if model not found
            
            # Plot the model line
            ax.plot(angles, values, 'o-', linewidth=1.5, label=model, alpha=0.7, color=color, markersize=3)
            ax.fill(angles, values, color=color, alpha=0.1)
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(tasks, fontsize=8)  # Smaller font size for task labels
        
        # Add a grid
        ax.grid(True)
        
        # Add zero line for reference
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Add legend with adjusted position and font size
        plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.4), fontsize=7)
        
        plt.title("Model Performance Relative to Mean (%)\nAcross Tasks (Using Best Format per Task)", pad=20, fontsize=10)
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_model_radar_relative_best_format_pct(self, output_file: str = "figs/model_radar_relative_best_format_pct_plot.pdf"):
        """Generate a radar plot comparing model performance relative to the mean as percentages.
        Shows how each model performs relative to the average across models for each task,
        using the best overall format for each model."""
        df = pd.DataFrame(self.results_data)
        
        # Map model names
        df['model'] = df['model'].map(MODEL_NAME_MAP)

        # Calculate average score for each model-format combination across all tasks
        format_means = df.groupby(['model', 'format'])['avg_score'].mean()
        
        # Find the best format for each model
        best_formats = format_means.groupby('model').idxmax().apply(lambda x: x[1])
        
        # Filter data to only include each model's best format
        best_format_data = []
        for model in best_formats.index:
            model_data = df[
                (df['model'] == model) & 
                (df['format'] == best_formats[model])
            ]
            best_format_data.append(model_data)
        
        best_format_df = pd.concat(best_format_data)
        
        # Calculate mean score for each task across all models
        task_means = best_format_df.groupby('task')['avg_score'].mean()
        
        # Calculate the percentage difference from mean for each model
        best_format_df['score_pct_diff'] = best_format_df.apply(
            lambda row: (row['avg_score'] - task_means[row['task']]) / task_means[row['task']] * 100
            if task_means[row['task']] > 0 else 0,
            axis=1
        )
        
        # Create pivot table for plotting
        pivot_data = best_format_df.pivot_table(
            values='score_pct_diff',
            index='model',
            columns='task',
            aggfunc='mean'
        )
        
        # Set up the angles for the radar plot
        tasks = pivot_data.columns
        num_tasks = len(tasks)
        angles = [n / float(num_tasks) * 2 * np.pi for n in range(num_tasks)]
        angles += angles[:1]  # Complete the circle
        
        # Create the plot with adjusted size for single column
        fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(projection='polar'))
        
        # Plot data
        for idx, model in enumerate(pivot_data.index):
            values = pivot_data.loc[model].values.flatten().tolist()
            values += values[:1]  # Complete the circle
            
            color = COLOR_MAP.get(model, '#000000')  # Default to black if model not found
            
            # Plot the model line
            ax.plot(angles, values, 'o-', linewidth=1.5, label=model, alpha=0.7, color=color, markersize=3)
            ax.fill(angles, values, color=color, alpha=0.1)
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(tasks, fontsize=8)  # Smaller font size for task labels
        
        # Add a grid
        ax.grid(True)
        
        # Add zero line for reference
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Add legend with adjusted position and font size
        plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.4), fontsize=7)
        
        plt.title("Model Performance Relative to Mean (%)\nAcross Tasks (Using Best Overall Format per Model)", pad=20, fontsize=10)
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_pseudo_comparison(self, output_file: str = "figs/pseudo_comparison.pdf"):
        """Generate bar graph comparing pseudonymized vs non-pseudonymized results for each task"""
        df = pd.DataFrame(self.results_data)
        
        # Map model names
        df['model'] = df['model'].map(MODEL_NAME_MAP)
        
        # Get ordered tasks
        ordered_tasks, _ = self._get_ordered_tasks_and_formats(df)
        
        # Calculate mean and std dev for each task and pseudonymization combination
        stats = df.groupby(['task', 'pseudonymized']).agg({
            'avg_score': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        stats.columns = ['task', 'pseudonymized', 'mean', 'std']
        
        # Set up the plot
        plt.figure(figsize=(15, 8))
        
        # Calculate bar positions
        x = np.arange(len(ordered_tasks))
        width = 0.35
        
        # Plot bars
        non_pseudo = stats[~stats['pseudonymized']].set_index('task').reindex(ordered_tasks)
        pseudo = stats[stats['pseudonymized']].set_index('task').reindex(ordered_tasks)
        
        plt.bar(x - width/2, non_pseudo['mean'], width, label='Original',
                color='skyblue', yerr=non_pseudo['std'], capsize=5)
        plt.bar(x + width/2, pseudo['mean'], width, label='Pseudonymized',
                color='lightcoral', yerr=pseudo['std'], capsize=5)
        
        # Customize plot
        plt.xlabel('Task')
        plt.ylabel('Score')
        plt.title('Comparison of Original vs Pseudonymized Performance by Task')
        plt.xticks(x, ordered_tasks, rotation=45, ha='right')
        plt.legend()
        
        # Add grid for better readability
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_model_pseudo_impact(self, output_file: str = "figs/model_pseudo_impact.pdf"):
        """Generate bar graph showing the impact of pseudonymization on each model's overall performance"""
        df = pd.DataFrame(self.results_data)
        
        # Map model names
        df['model'] = df['model'].map(MODEL_NAME_MAP)
        
        # Calculate mean and std dev for each model and pseudonymization combination
        stats = df.groupby(['model', 'pseudonymized']).agg({
            'avg_score': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        stats.columns = ['model', 'pseudonymized', 'mean', 'std']
        
        # Calculate the difference (pseudo - non_pseudo) for each model
        model_impacts = []
        for model in stats['model'].unique():
            model_stats = stats[stats['model'] == model]
            non_pseudo = model_stats[~model_stats['pseudonymized']].iloc[0]
            pseudo = model_stats[model_stats['pseudonymized']].iloc[0]
            
            # Calculate combined standard error
            combined_std = np.sqrt(non_pseudo['std']**2 + pseudo['std']**2)
            
            model_impacts.append({
                'model': model,
                'impact': pseudo['mean'] - non_pseudo['mean'],
                'std': combined_std
            })
        
        impact_df = pd.DataFrame(model_impacts)
        
        # Sort by impact
        impact_df = impact_df.sort_values('impact', ascending=True)
        
        # Set up the plot
        plt.figure(figsize=(12, 6))
        
        # Create bars
        bars = plt.bar(
            range(len(impact_df)), 
            impact_df['impact'],
            yerr=impact_df['std'],
            capsize=5
        )
        
        # Color bars based on positive/negative impact
        for i, bar in enumerate(bars):
            if impact_df['impact'].iloc[i] >= 0:
                bar.set_color('lightgreen')
            else:
                bar.set_color('lightcoral')
        
        # Customize plot
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.xlabel('Model')
        plt.ylabel('Performance Impact (Pseudo - Original)')
        plt.title('Impact of Pseudonymization on Model Performance')
        
        # Format x-axis labels
        plt.xticks(
            range(len(impact_df)),
            impact_df['model'],
            rotation=45,
            ha='right'
        )
        
        # Add value labels on the bars
        for i, v in enumerate(impact_df['impact']):
            plt.text(
                i, 
                v + (0.01 if v >= 0 else -0.01), 
                f'{v:.3f}',
                ha='center',
                va='bottom' if v >= 0 else 'top',
                fontsize=8
            )
        
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_combined_heatmaps(self, output_file: str = "figs/combined_heatmaps.pdf"):
        """Generate a combined heatmap plot for all models in a grid layout."""
        df = pd.DataFrame(self.results_data)
        
        # Map model names
        df['model'] = df['model'].map(MODEL_NAME_MAP)
        
        # Get ordered tasks and formats from raw data
        ordered_tasks, ordered_formats = self._get_ordered_tasks_and_formats(df)
        
        # Average across pseudonymized and non-pseudonymized results
        df_avg = df.groupby(['task', 'format', 'model'])['avg_score'].mean().reset_index()
        
        # Filter the DataFrame to include only the specified models using their exact names
        models_to_include = ['gemini-1.5-flash', 'claude-3.5-sonnet-v2', 'gpt-4o-mini', 'nova-pro', 'llama3.3-70b-instruct']
        df_avg = df_avg[df_avg['model'].isin(models_to_include)]

        # tasks_to_include = ['ShortestPath', 'ShortestPathFlexible']
        # df_avg = df_avg[df_avg['task'].isin(tasks_to_include)]
        # output_file = "figs/shortest_path/combined_heatmaps_shortest_path_flexible.pdf"
        # Create a figure with subplots in a 2x7 grid
        fig = plt.figure(figsize=(8, 4)) 
        gs = plt.GridSpec(2, len(df_avg['model'].unique()), figure=fig, wspace=0.1)  # Reduced horizontal margin
        
        # Create axes for each subplot
        axes = [[plt.subplot(gs[i, j]) for j in range(len(df_avg['model'].unique()))] 
                for i in range(2)]
        
        # Create a single colorbar axis
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        
        # First, create all pivot tables and find global min/max per task
        all_pivots = {}
        task_min_max = {}
        
        for model in sorted(df_avg['model'].unique()):
            model_df = df_avg[df_avg['model'] == model]
            pivot = model_df.pivot(index='task', columns='format', values='avg_score')
            pivot = pivot.reindex(ordered_tasks).reindex(columns=ordered_formats)
            all_pivots[model] = pivot
            
            # Update min/max for each task
            for task in pivot.index:
                if task not in task_min_max:
                    task_min_max[task] = {'min': float('inf'), 'max': float('-inf')}
                task_min_max[task]['min'] = min(task_min_max[task]['min'], pivot.loc[task].min())
                task_min_max[task]['max'] = max(task_min_max[task]['max'], pivot.loc[task].max())
        
        for col_idx, model in enumerate(sorted(df_avg['model'].unique())):
            pivot = all_pivots[model]
            
            # Unnormalized heatmap (top row)
            sns.heatmap(
                pivot,
                annot=True,
                fmt='.2f',
                cmap='Greens',
                vmin=0,
                vmax=1,
                ax=axes[0][col_idx],
                cbar=True if col_idx == len(df_avg['model'].unique())-1 else False,
                cbar_ax=cbar_ax if col_idx == len(df_avg['model'].unique())-1 else None,
                cbar_kws={'label': 'Score'},
                xticklabels=False,
                yticklabels=True if col_idx == 0 else False,
                annot_kws={"size": 6}
            )
            
            # Create normalized pivot using global task min/max
            normalized_pivot = pivot.copy()
            for task in pivot.index:
                min_val = task_min_max[task]['min']
                max_val = task_min_max[task]['max']
                if max_val > min_val:  # Avoid division by zero
                    normalized_pivot.loc[task] = (pivot.loc[task] - min_val) / (max_val - min_val)
            
            # Normalized heatmap (bottom row)
            sns.heatmap(
                normalized_pivot,
                annot=pivot.values,
                fmt='.2f',
                cmap='Greens',
                vmin=0,
                vmax=1,
                ax=axes[1][col_idx],
                cbar=False,
                xticklabels=True,
                yticklabels=True if col_idx == 0 else False,
                annot_kws={"size": 6}
            )
            
            # Only show y-labels on leftmost plots
            if col_idx == 0:
                axes[0][col_idx].set_ylabel('Raw Heatmap', fontsize=10)  # Increased font size
                axes[1][col_idx].set_ylabel('Row Normalized Heatmap', fontsize=10)  # Increased font size
            else:
                axes[0][col_idx].set_ylabel('')
                axes[1][col_idx].set_ylabel('')
            
            # Rotate x-labels on all bottom plots
            axes[1][col_idx].set_xticklabels(
                axes[1][col_idx].get_xticklabels(),
                rotation=45,
                ha='right',
                fontsize=8  # Increased font size
            )
            
            # Rotate y-labels on all left plots
            if col_idx == 0:
                axes[0][col_idx].set_yticklabels(
                    axes[0][col_idx].get_yticklabels(),
                    rotation=30,
                    va='top',  
                    ha='right',   # Horizontal alignment to the right
                    fontsize=8  # Increased font size
                )
                axes[1][col_idx].set_yticklabels(
                    axes[1][col_idx].get_yticklabels(),
                    rotation=30,
                    va='top',  
                    ha='right',   # Horizontal alignment to the right
                    fontsize=8  # Increased font size
                )

            # Add model name only at the top
            axes[0][col_idx].set_title(model, fontsize=10)  # Increased font size
            
            # Set x-label only on bottom row
            axes[0][col_idx].set_xlabel('')
            axes[1][col_idx].set_xlabel('')
            # axes[1][col_idx].set_xlabel('Format' if col_idx == len(df_avg['model'].unique())//2 else '')
            
        # Add row labels on the left
        # fig.text(0.02, 0.75, 'Raw Heatmap', rotation=90, va='center', fontsize=16)  # Increased font size
        # fig.text(0.02, 0.25, 'Row Normalized Heatmap', rotation=90, va='center', fontsize=16)  # Increased font size
        
        plt.tight_layout()
        # Adjust layout to make room for the colorbar
        plt.subplots_adjust(right=0.9)
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

    def generate_full_results_latex_table(self, output_file: str = "figs/full_results_summary.tex"):
        """Generate a LaTeX table with format -> model structure, with pseudonymized as sub-rows."""
        df = pd.DataFrame(self.results_data)

        # Map model names using MODEL_NAME_MAP
        df['model'] = df['model'].map(MODEL_NAME_MAP)

        df['format'] = df['format'].map(FORMAT_NAME_MAP)

        # Task name mapping
        TASK_NAME_MAP = {
            'AggByRelation': '\makecell{Agg by\\\\ Relation}',
            'AggNeighborProperties': '\makecell{Agg Neighbor\\\\ Properties}',
            'HighestDegreeNode': '\makecell{Highest\\\\ Degree}',
            'ShortestPath': '\makecell{Shortest\\\\ Path}',
            'TripleRetrieval': '\makecell{Triple\\\\ Retrieval}'
        }

        # Create pivot table for scores
        pivot = df.pivot_table(
            values='avg_score',
            index=['format', 'model', 'pseudonymized'],
            columns='task',
            aggfunc='mean'
        ).round(3)

        # Rename columns using task mapping
        pivot = pivot.rename(columns=TASK_NAME_MAP)

        # Calculate overall scores
        pivot['Overall'] = pivot.mean(axis=1)

        # Sort formats in desired order
        format_order = ['List of Edges', 'Structured JSON', 'Structured YAML', 'RDF Turtle', 'JSON-LD']
        pivot = pivot.reindex(format_order, level=0)

        # Calculate format-level averages
        format_avgs = {}
        for format_name in format_order:
            format_data = pivot.xs(format_name, level=0)
            # Calculate averages for non-pseudo and pseudo separately
            non_pseudo = format_data[~format_data.index.get_level_values('pseudonymized')].mean()
            pseudo = format_data[format_data.index.get_level_values('pseudonymized')].mean()
            format_avgs[format_name] = {'non_pseudo': non_pseudo, 'pseudo': pseudo}

        # Start building LaTeX string
        latex_lines = []
        
        # Add required packages
        latex_lines.extend([
            "% Required packages for rotated text and multirow",
            "%\\usepackage{rotating}",
            "%\\usepackage{multirow}",
            "%\\usepackage{makecell}",
            ""
        ])
        
        latex_lines.append("\\begin{longtable}{p{1.5cm}lcccccc}")  # Adjusted first column width for rotated text
        
        # Add caption and label
        latex_lines.append("\\caption{Full Results Summary by Format and Model} \\\\")
        latex_lines.append("\\label{tab:full_results_summary} \\\\")
        
        # Add headers
        latex_lines.append("\\toprule")
        headers = ['Format', 'Model', '\makecell{Agg by\\\\ Relation}', '\makecell{Agg Neighbor\\\\ Properties}', 
                  '\makecell{Highest\\\\ Degree}', '\makecell{Shortest\\\\ Path}', '\makecell{Triple\\\\ Retrieval}', 'Overall']
        latex_lines.append(" & ".join(headers) + " \\\\")
        latex_lines.append("\\midrule")
        latex_lines.append("\\endfirsthead")
        
        # Add continued headers for subsequent pages
        latex_lines.append("\\multicolumn{8}{c}{\\tablename\\ \\thetable\\ -- Continued from previous page} \\\\")
        latex_lines.append("\\toprule")
        latex_lines.append(" & ".join(headers) + " \\\\")
        latex_lines.append("\\midrule")
        latex_lines.append("\\endhead")
        
        # Add footer for all but last page
        latex_lines.append("\\midrule")
        latex_lines.append("\\multicolumn{8}{r}{Continued on next page} \\\\")
        latex_lines.append("\\endfoot")
        
        # Add footer for last page
        latex_lines.append("\\bottomrule")
        latex_lines.append("\\endlastfoot")

        # Process each format
        current_format = None
        first_model = True
        for idx in pivot.index:
            format_name, model, is_pseudo = idx
            
            # Add format separator if needed
            if current_format != format_name:
                if current_format is not None:
                    # Add format overall before moving to next format
                    avgs = format_avgs[current_format]
                    latex_lines.append("\\midrule")
                    row = [
                        "",
                        "\\textbf{Format Overall}",
                    ] + [f"{v:.3f}" for v in avgs['non_pseudo']]
                    latex_lines.append(" & ".join(row) + " \\\\")
                    row = [
                        "",
                        "\\quad +pseudo",
                    ] + [f"{v:.3f}" for v in avgs['pseudo']]
                    latex_lines.append(" & ".join(row) + " \\\\")
                    latex_lines.append("\\midrule")
                current_format = format_name
                first_model = True
            
            # Format the row
            values = pivot.loc[idx]
            if not is_pseudo:
                # Main model row with rotated format name
                row = [
                    f"\\multirow{{2}}{{=}}{{\\rotatebox[origin=c]{{90}}{{{format_name}}}}}" if first_model else "",  # Rotated format name
                    f"\\textbf{{{model}}}",
                ] + [f"{v:.3f}" for v in values]
                latex_lines.append(" & ".join(row) + " \\\\")
                first_model = False
            else:
                # Pseudonymized sub-row
                row = [
                    "",  # Empty for format
                    "\\quad +pseudo",  # Shortened pseudonymized indicator
                ] + [f"{v:.3f}" for v in values]
                latex_lines.append(" & ".join(row) + " \\\\")

        # Add the last format's overall after all models
        avgs = format_avgs[current_format]
        latex_lines.append("\\midrule")
        row = [
            "",
            "\\textbf{Format Overall}",
        ] + [f"{v:.3f}" for v in avgs['non_pseudo']]
        latex_lines.append(" & ".join(row) + " \\\\")
        row = [
            "",
            "\\quad +pseudo",
        ] + [f"{v:.3f}" for v in avgs['pseudo']]
        latex_lines.append(" & ".join(row) + " \\\\")

        # Add Format Overall section showing model averages across all formats
        latex_lines.append("\\midrule")
        
        # Process each model for Format Overall section
        first_model = True
        for model in pivot.index.get_level_values('model').unique():
            # Get scores for non-pseudonymized version
            model_scores = pivot.xs((model, False), level=('model', 'pseudonymized'), drop_level=False)
            avg_scores = model_scores.mean()  # Average across all formats
            
            # Main model row with rotated Format Overall text
            row = [
                f"\\multirow{{2}}{{=}}{{\\rotatebox[origin=c]{{90}}{{All Formats}}}}" if first_model else "",
                f"\\textbf{{{model}}}",
            ] + [f"{avg_scores[col]:.3f}" for col in pivot.columns]
            latex_lines.append(" & ".join(row) + " \\\\")
            
            # Get scores for pseudonymized version
            model_scores_pseudo = pivot.xs((model, True), level=('model', 'pseudonymized'), drop_level=False)
            avg_scores_pseudo = model_scores_pseudo.mean()  # Average across all formats
            
            # Pseudonymized sub-row
            row = [
                "",  # Empty for format
                "\\quad +pseudo",  # Shortened pseudonymized indicator
            ] + [f"{avg_scores_pseudo[col]:.3f}" for col in pivot.columns]
            latex_lines.append(" & ".join(row) + " \\\\")
            first_model = False

        latex_lines.append("\\midrule")
        # Calculate overall means for each task (column)
        # For non-pseudonymized
        overall_row = [
            "",
            "\\textbf{Overall Score}",
        ] + [f"{pivot[col][~pivot.index.get_level_values('pseudonymized')].mean():.3f}" for col in pivot.columns]
        latex_lines.append(" & ".join(overall_row) + " \\\\")

        # For pseudonymized
        overall_row_pseudo = [
            "",
            "\\quad +pseudo",
        ] + [f"{pivot[col][pivot.index.get_level_values('pseudonymized')].mean():.3f}" for col in pivot.columns]
        latex_lines.append(" & ".join(overall_row_pseudo) + " \\\\")

        # Finish table
        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{longtable}")

        # Write to file
        with open(output_file, 'w') as f:
            f.write("\n".join(latex_lines))

    def generate_best_format_table(self, output_file: str = "figs/best_format_summary.tex"):
        """Generate a LaTeX table with the best overall format for each model."""
        df = pd.DataFrame(self.results_data)

        # Map model names using MODEL_NAME_MAP
        df['model'] = df['model'].map(MODEL_NAME_MAP)

        # Calculate average score for each model-format combination
        format_means = df.groupby(['model', 'format'])['avg_score'].mean().reset_index()

        # Find the best format for each model
        best_formats = format_means.loc[format_means.groupby('model')['avg_score'].idxmax()]

        # Map format names to their display names
        best_formats['format'] = best_formats['format'].map(FORMAT_NAME_MAP)

        # Start building LaTeX string
        latex_lines = []
        
        # Add required packages
        latex_lines.append("\\begin{table}[ht]")
        latex_lines.append("\\centering")
        latex_lines.append("\\caption{Best Textualization Strategy for Each Model}")
        latex_lines.append("\\begin{tabular}{ll}")
        latex_lines.append("\\toprule")
        latex_lines.append("Model & Best $f$ \\\\")
        latex_lines.append("\\midrule")

        # Add rows for each model and its best format
        for _, row in best_formats.iterrows():
            latex_lines.append(f"{row['model']} & {row['format']} \\\\")
        
        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")

        # Write to file
        with open(output_file, 'w') as f:
            f.write("\n".join(latex_lines))

    def plot_pseudo_scatter_by_format(self, output_file: str = "figs/pseudo_scatter_by_format.pdf"):
        """Generate a scatter plot showing pseudonymization impact for each format-model combination."""
        df = pd.DataFrame(self.results_data)
        
        # Map model names
        df['model'] = df['model'].map(MODEL_NAME_MAP)
        df['format'] = df['format'].map(FORMAT_NAME_MAP)

        # Calculate means for each model-task-format combination
        means = df.groupby(['model', 'task', 'format', 'pseudonymized'])['avg_score'].agg(['mean', 'std', 'count']).reset_index()

        # Pivot to get pseudo and non-pseudo scores side by side
        pivot = means.pivot(index=['model', 'task', 'format'], columns='pseudonymized', values=['mean', 'std', 'count']).reset_index()
        pivot.columns = ['model', 'task', 'format', 'mean_non_pseudo', 'mean_pseudo', 'std_non_pseudo', 'std_pseudo', 'n_non_pseudo', 'n_pseudo']
        
        # Calculate difference and standard error
        pivot['diff'] = pivot['mean_pseudo'] - pivot['mean_non_pseudo']
        # Standard error of the difference using pooled standard error formula
        pivot['se'] = np.sqrt(
            (pivot['std_pseudo']**2 / pivot['n_pseudo']) + 
            (pivot['std_non_pseudo']**2 / pivot['n_non_pseudo'])
        )
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Calculate x positions with jitter
        formats = pivot['format'].unique()
        x_positions = {format_name: i for i, format_name in enumerate(formats)}
        jitter = np.random.normal(0, 0.1, size=len(pivot))
        x_coords = [x_positions[format_name] + j for format_name, j in zip(pivot['format'], jitter)]
        
        # Create color map for models
        models = pivot['model'].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        model_color_map = dict(zip(models, colors))
        
        # Plot error bars and points
        for model in models:
            mask = pivot['model'] == model
            plt.scatter(
                [x_coords[i] for i in range(len(pivot)) if mask.iloc[i]],
                pivot[mask]['diff'],
                label=model,
                color=model_color_map[model],
                alpha=0.6
            )
        
        # Add zero line
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # Customize plot
        plt.xlabel('Format')
        plt.ylabel('Pseudonymization Impact (Pseudo - Original)')
        plt.title('Impact of Pseudonymization by Format and Model')
        plt.xticks(range(len(formats)), formats, rotation=45, ha='right')
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_pseudo_scatter_by_model(self, output_file: str = "figs/pseudo_scatter_by_model.pdf"):
        """Generate a scatter plot showing pseudonymization impact for each model-task-format combination."""
        df = pd.DataFrame(self.results_data)
        
        # Map model names
        df['model'] = df['model'].map(MODEL_NAME_MAP)
        df['format'] = df['format'].map(FORMAT_NAME_MAP)

        # Calculate means for each model-task-format combination
        means = df.groupby(['model', 'task', 'format', 'pseudonymized'])['avg_score'].agg(['mean', 'std', 'count']).reset_index()
        
        # Pivot to get pseudo and non-pseudo scores side by side
        pivot = means.pivot(index=['model', 'task', 'format'], columns='pseudonymized', values=['mean', 'std', 'count']).reset_index()
        pivot.columns = ['model', 'task', 'format', 'mean_non_pseudo', 'mean_pseudo', 'std_non_pseudo', 'std_pseudo', 'n_non_pseudo', 'n_pseudo']
        
        # Calculate difference and standard error
        pivot['diff'] = pivot['mean_pseudo'] - pivot['mean_non_pseudo']
        # Standard error of the difference using pooled standard error formula
        pivot['se'] = np.sqrt(
            (pivot['std_pseudo']**2 / pivot['n_pseudo']) + 
            (pivot['std_non_pseudo']**2 / pivot['n_non_pseudo'])
        )
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Calculate x positions with jitter
        models = pivot['model'].unique()
        x_positions = {model: i for i, model in enumerate(models)}
        jitter = np.random.normal(0, 0.1, size=len(pivot))
        x_coords = [x_positions[model] + j for model, j in zip(pivot['model'], jitter)]
        
        # Create color map for tasks
        tasks = pivot['task'].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(tasks)))
        task_color_map = dict(zip(tasks, colors))
        
        # Plot error bars and points
        for task in tasks:
            mask = pivot['task'] == task
            plt.scatter(
                [x_coords[i] for i in range(len(pivot)) if mask.iloc[i]],
                pivot[mask]['diff'],
                label=task,
                color=task_color_map[task],
                alpha=0.6
            )
        
        # Add zero line
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # Customize plot
        plt.xlabel('Model')
        plt.ylabel('Pseudonymization Impact (Pseudo - Original)')
        plt.title('Impact of Pseudonymization by Model and Task')
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        plt.legend(title='Task', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_pseudo_scatter_by_task(self, output_file: str = "figs/pseudo_scatter_by_task.pdf"):
        """Generate a scatter plot showing pseudonymization impact for each task-model-format combination."""
        df = pd.DataFrame(self.results_data)
        
        # Map model names
        df['model'] = df['model'].map(MODEL_NAME_MAP)
        df['format'] = df['format'].map(FORMAT_NAME_MAP)

        # Calculate means for each model-task-format combination
        means = df.groupby(['model', 'task', 'format', 'pseudonymized'])['avg_score'].agg(['mean', 'std', 'count']).reset_index()
        
        # Pivot to get pseudo and non-pseudo scores side by side
        pivot_full = means.pivot(index=['model', 'task', 'format'], columns='pseudonymized', values=['mean', 'std', 'count']).reset_index()
        pivot_full.columns = ['model', 'task', 'format', 'mean_non_pseudo', 'mean_pseudo', 'std_non_pseudo', 'std_pseudo', 'n_non_pseudo', 'n_pseudo']

        means_stats = df.groupby(['task', 'model', 'pseudonymized'])['avg_score'].agg(['mean', 'std', 'count']).reset_index()
        pivot_stats = means_stats.pivot(index=['task', 'model'], columns='pseudonymized', values=['mean', 'std', 'count']).reset_index()
        pivot_stats.columns = ['task', 'model', 'mean_non_pseudo', 'mean_pseudo', 'std_non_pseudo', 'std_pseudo', 'n_non_pseudo', 'n_pseudo']

        # Calculate difference and standard error
        pivot_full['diff'] = pivot_full['mean_pseudo'] - pivot_full['mean_non_pseudo']
        pivot_stats['diff'] = pivot_stats['mean_pseudo'] - pivot_stats['mean_non_pseudo']

        # Standard error of the difference using pooled standard error formula
        pivot_stats['se'] = np.sqrt(
            (pivot_stats['std_pseudo']**2 / pivot_stats['n_pseudo']) + 
            (pivot_stats['std_non_pseudo']**2 / pivot_stats['n_non_pseudo'])
        )
        pivot = pivot_full
        # Create figure
        plt.figure(figsize=(6, 3.5))  # Adjusted for a single column figure
        
        # Create color map for models
        models = pivot['model'].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        model_color_map = dict(zip(models, colors))
        model_color_map = COLOR_MAP

        # Calculate x positions with jitter
        tasks = pivot['task'].unique()
        x_positions = {task: i for i, task in enumerate(tasks)}
        jitter = np.random.normal(0, 0.1, size=len(pivot))
        x_coords = [x_positions[task] + j for task, j in zip(pivot['task'], jitter)]
        jitter_stats = np.tile(np.arange(-0.3, 0.3, 0.6/len(models)), len(pivot_stats) // len(models))  
        x_coords_stats = [x_positions[task] + j for task, j in zip(pivot_stats['task'], jitter_stats)]
        
        # Plot error bars and points
        for model in models:
            mask = pivot['model'] == model
            plt.scatter(
                [x_coords[i] for i in range(len(pivot)) if mask.iloc[i]],
                pivot[mask]['diff'],
                label=model,
                color=model_color_map[model],
                alpha=0.6
            )
            mask = pivot_stats['model'] == model
            plt.errorbar(
                [x_coords_stats[i] for i in range(len(pivot_stats)) if mask.iloc[i]],
                pivot_stats[mask]['diff'],
                yerr=pivot_stats[mask]['se'],  # Using standard error as the error bars
                color=model_color_map[model],
                alpha=0.6,
                fmt='o',  # Changed to a valid format string
                capsize=5  # Size of the error bar caps 
            )
        
        # Add zero line
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # Customize plot
        plt.xlabel('')
        plt.ylabel('Difference (Pseudo - Original)')
        plt.title('Impact of Pseudonymization by Task')
        plt.xticks(range(len(tasks)), tasks, rotation=25, ha='center')
        # plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

    def generate_token_usage_table(self, output_file: str = "figs/token_usage_summary.tex"):
        """Generate a LaTeX table showing average token usage by format."""
        df = pd.DataFrame(self.results_data)
        
        # Map format names using FORMAT_NAME_MAP
        df['format'] = df['format'].map(FORMAT_NAME_MAP)
        
        # Calculate average token usage for each format
        token_stats = df.groupby(['format'])['avg_input_tokens'].agg(['mean', 'std']).round(1)
        
        # Start building LaTeX string
        latex_lines = []
        
        latex_lines.append("\\begin{table}[ht]")
        latex_lines.append("\\centering")
        latex_lines.append("\\caption{Average Input Token Usage by Format}")
        latex_lines.append("\\begin{tabular}{lr}")
        latex_lines.append("\\toprule")
        latex_lines.append("Textualizer & Mean Input Tokens \\\\")
        latex_lines.append("\\midrule")
        
        # Process each format
        for format_name in FORMAT_NAME_MAP.values():
            if format_name in token_stats.index.get_level_values('format'):
                orig_tokens = token_stats.loc[(format_name), 'mean']
                orig_std = token_stats.loc[(format_name), 'std']
                
                latex_lines.append(f"{format_name} & {orig_tokens:.1f} $\\pm$ {orig_std:.1f} \\\\")
        
        # Add overall averages
        latex_lines.append("\\midrule")
        overall_orig = token_stats['mean'].mean()
        overall_orig_std = token_stats['std'].mean()
        latex_lines.append(f"Overall & {overall_orig:.1f} $\\pm$ {overall_orig_std:.1f} \\\\")
        
        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write("\n".join(latex_lines))

def main():
    print("Starting results analysis...")
    analyzer = ResultsAnalyzer()
    
    print("\nLoading results files...")
    analyzer.load_results()
    
    if not analyzer.results_data:
        print("\nNo results data was found! Please check if:")
        print("1. The benchmark_data directory exists")
        print("2. It contains task directories ending in 'Task'")
        print("3. The task directories contain format directories")
        print("4. The format directories contain model directories with 'small_results.json' files")
        return
    
    # Make figs directory
    os.makedirs("figs", exist_ok=True)

    print("\nGenerating token usage table...")
    analyzer.generate_token_usage_table()
    
    print("\nGenerating LaTeX table...")
    analyzer.generate_latex_table()
    
    print("\nGenerating best format summary table...")
    analyzer.generate_best_format_table()
    
    print("\nGenerating full results LaTeX table...")
    analyzer.generate_full_results_latex_table()
    
    print("\nGenerating combined heatmap plot...")
    analyzer.plot_combined_heatmaps()
    
    print("\nGenerating radar plots...")
    analyzer.plot_radar()  # Best format per task
    analyzer.plot_radar_best_overall_format()  # Best overall format
    analyzer.plot_format_radar()  # Format comparison (absolute)
    analyzer.plot_format_radar_relative_pct()  # Format comparison (percentage)
    analyzer.plot_model_radar_relative()  # Model performance relative to mean (absolute, best format per task)
    analyzer.plot_model_radar_relative_pct()  # Model performance relative to mean (percentage, best format per task)
    analyzer.plot_model_radar_relative_best_format()  # Model performance relative to mean (absolute, best overall format)
    analyzer.plot_model_radar_relative_best_format_pct()  # Model performance relative to mean (percentage, best overall format)
    
    print("\nGenerating pseudo comparison plots...")
    analyzer.plot_pseudo_comparison()
    analyzer.plot_model_pseudo_impact()
    analyzer.plot_pseudo_scatter_by_format()  # New scatter plot by format
    analyzer.plot_pseudo_scatter_by_model()  # New scatter plot by model
    analyzer.plot_pseudo_scatter_by_task()
    
    print("\nGenerating summary statistics...")
    analyzer.print_summary()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()