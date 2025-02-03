# results_analyzer.py

import json
import os
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict, List
import seaborn as sns
import matplotlib.pyplot as plt

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
                            
                            print(f"      Found {len(scores)} examples, avg score: {avg_score:.3f}")
                            
                            self.results_data.append({
                                'task': task_dir.stem,
                                'format': format_name,
                                'model': llm_dir.name,
                                'pseudonymized': is_pseudo,
                                'avg_score': avg_score,
                                'std_score': std_score,
                                'num_examples': len(scores)
                            })

    def plot_heatmap(self, output_file: str = "results_heatmap.pdf", normalized_output_file: str = "results_heatmap_normalized.pdf"):
        """Generate heatmap of results and a normalized version, both overall and per-model"""
        df = pd.DataFrame(self.results_data)
        
        # Generate overall heatmaps
        self._generate_single_heatmap(df, output_file, normalized_output_file, title_prefix="Overall")
        
        # Generate per-model heatmaps
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            model_name = model.split('/')[-1]  # Get last part of model path
            model_output = output_file.replace('.pdf', f'_{model_name}.pdf')
            model_normalized_output = normalized_output_file.replace('.pdf', f'_{model_name}.pdf')
            self._generate_single_heatmap(model_df, model_output, model_normalized_output, title_prefix=f"Model: {model_name}")
    
    def _generate_single_heatmap(self, df: pd.DataFrame, output_file: str, normalized_output_file: str, title_prefix: str = ""):
        """Helper method to generate a pair of heatmaps (regular and normalized) for the given dataframe"""
        pivot = pd.pivot_table(
            df,
            values='avg_score',
            index='task',
            columns=['format', 'pseudonymized'],
            aggfunc='mean'
        )
        
        # Regular heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pivot, 
            annot=True, 
            fmt='.3f',
            cmap='Greens',
            vmin=0,
            vmax=1,
            annot_kws={'size': 8},
            cbar_kws={'label': 'Score'}
        )
        
        plt.title(f"{title_prefix} Task Performance by Format and Pseudonymization", pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
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
            annot_kws={'size': 8},
            cbar_kws={'label': 'Normalized Score'}
        )
        
        plt.title(f"{title_prefix} Task Performance by Format and Pseudonymization (Row Normalized)", pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(normalized_output_file, bbox_inches='tight', dpi=300)
        plt.close()

    def generate_latex_table(self, output_file: str = "results_summary.tex"):
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

    def plot_radar(self, output_file: str = "model_radar_plot.pdf"):
        """Generate a radar plot comparing models across different tasks.
        Uses the best performing format for each task."""
        df = pd.DataFrame(self.results_data)
        
        # Get the best score for each task-model combination across formats
        best_scores = df.groupby(['task', 'model'])['avg_score'].max().reset_index()
        
        # Pivot the data for plotting
        pivot_data = best_scores.pivot(index='model', columns='task', values='avg_score')
        
        # Set up the angles for the radar plot
        tasks = pivot_data.columns
        num_tasks = len(tasks)
        angles = [n / float(num_tasks) * 2 * np.pi for n in range(num_tasks)]
        angles += angles[:1]  # Complete the circle
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot data
        for idx, model in enumerate(pivot_data.index):
            values = pivot_data.loc[model].values.flatten().tolist()
            values += values[:1]  # Complete the circle
            
            # Plot the model line
            ax.plot(angles, values, 'o-', linewidth=2, label=model.split('/')[-1], alpha=0.7)
            ax.fill(angles, values, alpha=0.1)
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(tasks)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title("Model Performance Across Tasks\n(Best Format per Task)", pad=20)
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_format_radar(self, output_file: str = "format_radar_plot.pdf"):
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
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        # Plot data
        colors = plt.cm.Set2(np.linspace(0, 1, len(pivot_data.index)))
        for idx, (format_name, color) in enumerate(zip(pivot_data.index, colors)):
            values = pivot_data.loc[format_name].values.flatten().tolist()
            values += values[:1]  # Complete the circle
            
            # Plot the format line
            ax.plot(angles, values, 'o-', linewidth=2, label=format_name, color=color, alpha=0.7)
            ax.fill(angles, values, color=color, alpha=0.1)
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(tasks)
        
        # Add a grid
        ax.grid(True)
        
        # Add zero line for reference
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title("Format Performance Relative to Mean\nAcross Tasks and Models", pad=20)
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_radar_best_overall_format(self, output_file: str = "model_radar_best_format_plot.pdf"):
        """Generate a radar plot comparing models across different tasks.
        Uses the single best performing format overall for each model."""
        df = pd.DataFrame(self.results_data)
        
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
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot data
        colors = plt.cm.Set2(np.linspace(0, 1, len(pivot_data.index)))
        for idx, (model, color) in enumerate(zip(pivot_data.index, colors)):
            values = pivot_data.loc[model].values.flatten().tolist()
            values += values[:1]  # Complete the circle
            
            # Get the best format for this model for the legend
            best_format = best_formats[model]
            label = f"{model.split('/')[-1]}\n(using {best_format})"
            
            # Plot the model line
            ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color, alpha=0.7)
            ax.fill(angles, values, color=color, alpha=0.1)
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(tasks)
        
        # Add a grid
        ax.grid(True)
        
        # Add legend with adjusted position and font size
        plt.legend(loc='center left', bbox_to_anchor=(1.2, 0.5), fontsize=8)
        
        plt.title("Model Performance Across Tasks\n(Using Best Overall Format per Model)", pad=20)
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

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
    
    print("\nGenerating LaTeX table...")
    analyzer.generate_latex_table()
    
    print("\nGenerating heatmap plot...")
    analyzer.plot_heatmap()
    
    print("\nGenerating radar plots...")
    analyzer.plot_radar()  # Best format per task
    analyzer.plot_radar_best_overall_format()  # Best overall format
    analyzer.plot_format_radar()  # Format comparison
    
    print("\nGenerating summary statistics...")
    analyzer.print_summary()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()