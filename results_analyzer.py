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

    def generate_latex_table(self, output_file: str = "results_summary.tex"):
        """Generate LaTeX table with format x task matrix, subdividing each task into normal/pseudo"""
        df = pd.DataFrame(self.results_data)
        
        # Create pivot table with tasks as columns and formats as rows
        pivot = pd.pivot_table(
            df,
            values='avg_score',
            index=['model', 'format'],
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
        
        latex_str = pivot.to_latex(
            float_format="%.3f",
            bold_rows=True,
            multicolumn=True,
            multicolumn_format='c',
            caption="Knowledge Graph Format Performance by Task",
            label="tab:results"
        )
        
        # Add header clarification
        latex_str = latex_str.replace('& \\multicolumn{2}', '& \\multicolumn{2}{c}')
        
        with open(output_file, 'w') as f:
            f.write(latex_str)

    def plot_heatmap(self, output_file: str = "results_heatmap.pdf", normalized_output_file: str = "results_heatmap_normalized.pdf"):
        """Generate heatmap of results and a normalized version"""
        df = pd.DataFrame(self.results_data)
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
        
        plt.title("Task Performance by Format and Pseudonymization", pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Normalized heatmap (each row scaled independently)
        plt.figure(figsize=(12, 8))
        # Normalize each row to [0,1] range
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
        
        plt.title("Task Performance by Format and Pseudonymization (Row Normalized)", pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(normalized_output_file, bbox_inches='tight', dpi=300)
        plt.close()
        
    def print_summary(self):
        """Print summary statistics"""
        df = pd.DataFrame(self.results_data)
        
        print("\n=== Results Summary ===")
        
        print("\nScores by task:")
        print(df.groupby('task')['avg_score'].agg(['mean', 'max', 'std', 'count']))
        
        print("\nScores by format:")
        print(df.groupby('format')['avg_score'].agg(['mean', 'max', 'std', 'count']))
        
        print("\nPseudonymization impact:")
        print("\nMean scores:")
        print(df.groupby(['task', 'pseudonymized'])['avg_score'].mean().unstack())
        print("\nMax scores:")
        print(df.groupby(['task', 'pseudonymized'])['avg_score'].max().unstack())

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
    
    print("\nGenerating summary statistics...")
    analyzer.print_summary()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()