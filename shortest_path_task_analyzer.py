import json
import os
import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from collections import defaultdict

# Reuse the same model name mapping from results_analyzer
MODEL_NAME_MAP = {
    'gemini-1.5-flash': 'gemini-1.5-flash',
    'gpt-4o-mini': 'gpt-4o-mini',
    'us.amazon.nova-lite-v1:0': 'nova-lite',
    'us.amazon.nova-pro-v1:0': 'nova-pro',
    'us.anthropic.claude-3-5-sonnet-20241022-v2:0': 'claude-3.5-sonnet-v2',
    'us.meta.llama3-2-1b-instruct-v1:0': 'llama3.2-1b-instruct',
    'us.meta.llama3-3-70b-instruct-v1:0': 'llama3.3-70b-instruct'
}

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
}

FORMAT_NAME_MAP = {
    'list_of_edges': 'List of Edges',
    'structured_json': 'Structured JSON',
    'structured_yaml': 'Structured YAML',
    'rdf_turtle3': 'RDF Turtle',
    'json_ld3': 'JSON-LD'
}

class ShortestPathAnalyzer:
    def __init__(self, results_dir: str = "benchmark_data"):
        self.results_dir = Path(results_dir) / "ShortestPath"
        self.results_data = []
        self.full_results_data = []
        self.path_length_data = []
            
    def extract_path_from_answer(self, answer_text):
        """Extract the path from an answer string"""
        # Match the list format in the answer: SHORTEST PATH: ['Entity1', 'Entity2', ...]
        pattern = r"SHORTEST PATH: \[(.*?)\]"
        match = re.search(pattern, answer_text)
        if match:
            # Parse entities from list string
            entities_str = match.group(1)
            
            entities = [entity.strip().strip("'\"") for entity in entities_str.split(',')]
            return entities
        return None
        
    def compute_path_length(self, path):
        """Compute the length of a path (nodes - 1)"""
        return len(path) - 1 if path else -1
        
    def get_ground_truth_path(self, example_id):
        """Get the ground truth path for an example"""
        if not self.ground_truth or example_id >= len(self.ground_truth):
            return None
            
        example = self.ground_truth[example_id]
        paths = []
        for path in example.get("shortest_paths", []):
            paths.append([entity["label"] for entity in path])
        return paths
        
    def get_ground_truth_length(self, example_id):
        """Get the ground truth path length for an example"""
        paths = self.get_ground_truth_path(example_id)
        if not paths:
            return 0
        # All paths should be the same length for shortest path
        return len(paths[0]) - 1 if paths[0] else 0
    
    def load_results(self):
        """Load all ShortestPath task results files"""
        print(f"\nSearching for ShortestPath results in: {self.results_dir}")
        format_dirs = list(self.results_dir.glob("*"))
        format_dirs = [d for d in format_dirs if d.is_dir() and not d.name in ["kg", "pseudo_kg"]]
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
                        
                        # Process each example
                        correct_count = 0
                        total_count = 0
                        path_lengths = defaultdict(list)  # For tracking scores by path length
                        
                        for i, example in enumerate(results):
                            if example is None:
                                continue
                                
                            total_count += 1
                            
                            # Extract predicted path from answer
                            predicted_answer = example.get("response", None)
                            predicted_path = self.extract_path_from_answer(predicted_answer)
                            predicted_length = self.compute_path_length(predicted_path)
                            
                            # Get ground truth paths
                            gt_paths = example.get("shortest_paths", [])
                            gt_length = len(gt_paths[0]) - 1 if gt_paths else -1

                            score = example.get("score", 0.0)
                            correct_count += score
                            
                            # Add to results
                            path_lengths[gt_length].append(score)
                            
                            self.full_results_data.append({
                                'format': format_name,
                                'model': llm_dir.name,
                                'pseudonymized': is_pseudo,
                                'example_id': i,
                                'gt_path_length': gt_length,
                                'predicted_path_length': predicted_length,
                                'score': score
                            })
                        
                        # Compute overall accuracy
                        avg_score = correct_count / total_count if total_count > 0 else 0
                        self.results_data.append({
                            'format': format_name,
                            'model': llm_dir.name,
                            'pseudonymized': is_pseudo,
                            'avg_score': avg_score,
                            'correct_count': correct_count,
                            'num_examples': total_count
                        })
                        
                        # Compute accuracy by path length
                        for length, scores in path_lengths.items():
                            avg_length_score = sum(scores) / len(scores) if scores else 0
                            self.path_length_data.append({
                                'format': format_name,
                                'model': llm_dir.name,
                                'pseudonymized': is_pseudo,
                                'path_length': length,
                                'avg_score': avg_length_score,
                                'num_examples': len(scores)
                            })
                    

    def plot_model_comparison(self, output_file: str = "figs/shortest_path/shortest_path_model_comparison.pdf"):
        """Generate a bar plot comparing model performance, averaged over formats"""
        df = pd.DataFrame(self.results_data)
        breakpoint()
        
        if df.empty:
            print("No data available for model comparison plot")
            return
            
        # Map names
        df['model'] = df['model'].map(lambda x: MODEL_NAME_MAP.get(x, x))
        df['format'] = df['format'].map(lambda x: FORMAT_NAME_MAP.get(x, x))
        
        # Calculate average scores by model, averaging over formats
        model_scores = df.groupby(['model'])['avg_score'].mean().reset_index()
        model_scores = model_scores.sort_values('avg_score', ascending=False)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_scores['model'], model_scores['avg_score'], color=[COLOR_MAP.get(model, 'gray') for model in model_scores['model']])
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Customize plot
        plt.xlabel('Model')
        plt.ylabel('Average Accuracy')
        plt.title('ShortestPath Task Performance by Model\n(Averaged over Formats)')
        plt.ylim(0, 1.05)  # Set y-axis limit
        plt.grid(True, axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        
    def plot_format_comparison(self, output_file: str = "figs/shortest_path/shortest_path_format_comparison.pdf"):
        """Generate a bar plot comparing format performance, averaged over models"""
        df = pd.DataFrame(self.results_data)
        
        if df.empty:
            print("No data available for format comparison plot")
            return
            
        # Map names
        df['model'] = df['model'].map(lambda x: MODEL_NAME_MAP.get(x, x))
        df['format'] = df['format'].map(lambda x: FORMAT_NAME_MAP.get(x, x))
        
        # Calculate average scores by format, averaging over models
        format_scores = df.groupby(['format', 'pseudonymized'])['avg_score'].mean().reset_index()
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Set up the plot
        bar_width = 0.35
        formats = sorted(format_scores['format'].unique())
        x = np.arange(len(formats))
        
        # Separate pseudonymized and non-pseudonymized
        non_pseudo = format_scores[format_scores['pseudonymized'] == False]
        pseudo = format_scores[format_scores['pseudonymized'] == True]
        
        # Create a mapping from format to index
        format_to_idx = {format: i for i, format in enumerate(formats)}
        
        # Plot bars
        non_pseudo_x = [format_to_idx.get(f, -1) for f in non_pseudo['format']]
        pseudo_x = [format_to_idx.get(f, -1) for f in pseudo['format']]
        
        plt.bar(x - bar_width/2, non_pseudo['avg_score'], bar_width, label='Original', color='skyblue')
        plt.bar(x + bar_width/2, pseudo['avg_score'], bar_width, label='Pseudonymized', color='salmon')
        
        # Customize plot
        plt.xlabel('Format')
        plt.ylabel('Average Accuracy')
        plt.title('ShortestPath Task Performance by Format\n(Averaged over Models)')
        plt.xticks(x, formats)
        plt.ylim(0, 1.05)  # Set y-axis limit
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        
    def plot_path_length_performance(self, output_file: str = "figs/shortest_path/path_length_performance.pdf"):
        """Generate a line plot showing performance by path length"""
        df = pd.DataFrame(self.path_length_data)
        
        if df.empty:
            print("No data available for path length performance plot")
            return
            
        # Map names
        df['model'] = df['model'].map(lambda x: MODEL_NAME_MAP.get(x, x))
        df['format'] = df['format'].map(lambda x: FORMAT_NAME_MAP.get(x, x))
        
        # Average over models and formats
        length_scores = df.groupby(['path_length'])['avg_score'].agg(['mean', 'count']).reset_index()
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Primary axis for scores
        plt.subplot(111)
        plt.plot(length_scores['path_length'], length_scores['mean'], marker='o', linewidth=2, color='blue')
        
        # Add count above each point
        for i, count in enumerate(length_scores['count']):
            plt.annotate(f"n={count}", 
                         (length_scores['path_length'][i], length_scores['mean'][i]), 
                         textcoords="offset points", 
                         xytext=(0,10), 
                         ha='center')
        
        # Customize plot
        plt.xlabel('Path Length')
        plt.ylabel('Average Accuracy')
        plt.title('ShortestPath Task Performance by Path Length\n(Averaged over Models and Formats)')
        plt.ylim(0, 1.05)  # Set y-axis limit
        plt.grid(True, alpha=0.3)
        plt.xticks(length_scores['path_length'])
        
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        
    def plot_path_length_by_model(self, output_file: str = "figs/shortest_path/path_length_by_model.pdf"):
        """Generate a line plot showing performance by path length for each model"""
        df = pd.DataFrame(self.path_length_data)
        
        if df.empty:
            print("No data available for path length by model plot")
            return
            
        # Map names
        df['model'] = df['model'].map(lambda x: MODEL_NAME_MAP.get(x, x))
        df['format'] = df['format'].map(lambda x: FORMAT_NAME_MAP.get(x, x))
        
        # Average over formats for each model
        model_length_scores = df.groupby(['model', 'path_length']).agg({
            'avg_score': 'mean',
            'num_examples': 'sum'
        }).reset_index()
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Get unique models and lengths for the plot
        models = sorted(model_length_scores['model'].unique())
        lengths = sorted(model_length_scores['path_length'].unique())
        
        # Plot line for each model
        for model in models:
            model_data = model_length_scores[model_length_scores['model'] == model]
            plt.plot(model_data['path_length'], model_data['avg_score'], 
                     marker='o', linewidth=2, label=model, 
                     color=COLOR_MAP.get(model, None))
                     
        # Find the best format for each model and add dashed line
        # Group by model and format, calculate average score across all path lengths
        format_model_scores = df.groupby(['model', 'format']).agg({
            'avg_score': 'mean'
        }).reset_index()
        
        # Get the best format for each model
        best_formats = format_model_scores.loc[format_model_scores.groupby('model')['avg_score'].idxmax()]
        
        # Add dashed line for best format for each model
        for _, row in best_formats.iterrows():
            model = row['model']
            best_format = row['format']
            
            # Get data for this model and format
            best_format_data = df[(df['model'] == model) & (df['format'] == best_format)]
            
            # Group by path length and calculate average (in case there are duplicates with different pseudonymized values)
            best_format_data = best_format_data.groupby('path_length')['avg_score'].mean().reset_index()
            
            # Sort by path length for proper line plotting
            best_format_data = best_format_data.sort_values('path_length')
            
            # Plot dashed line
            plt.plot(best_format_data['path_length'], best_format_data['avg_score'], 
                     linestyle='--', marker='s', linewidth=1.5, 
                     label=f"{model} (Best: {best_format})",
                     color=COLOR_MAP.get(model, None))
        
        # Customize plot
        plt.xlabel('Path Length')
        plt.ylabel('Average Accuracy')
        plt.title('ShortestPath Task Performance by Path Length and Model\n(Solid: Avg over Formats, Dashed: Best Format)')
        plt.ylim(0, 1.05)  # Set y-axis limit
        plt.grid(True, alpha=0.3)
        plt.xticks(lengths)
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        
    def plot_predicted_vs_actual_length(self, output_file: str = "predicted_vs_actual_length.pdf"):
        """Generate a heatmap comparing predicted path length vs actual path length"""
        df = pd.DataFrame(self.full_results_data)
        
        if df.empty:
            print("No data available for predicted vs actual length plot")
            return
        breakpoint()
        # Create a contingency table of actual vs predicted path lengths
        # Bin predicted path lengths, grouping all values above 5 into a single bin
        df['predicted_path_length_bin'] = pd.cut(df['predicted_path_length'], 
                                              bins=[-1, 0, 1, 2, 3, 4, float('inf')], 
                                              labels=['None', '1', '2', '3', '4', '5+'], 
                                              right=True)
        
        contingency = pd.crosstab(
            df['gt_path_length'], 
            df['predicted_path_length_bin'],
            normalize='index',  # Normalize by row (actual path length)
            rownames=['Actual Length'],
            colnames=['Predicted Length']
        )
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(contingency, annot=True, cmap='YlGnBu', fmt='.2f', cbar_kws={'label': 'Proportion'})
        
        plt.title('Actual vs Predicted Path Length\n(Normalized by Actual Length)')
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        
    def plot_error_analysis(self, output_file: str = "error_analysis.pdf"):
        """Generate a bar plot showing error types (off by 1, off by 2, etc.)"""
        df = pd.DataFrame(self.full_results_data)
        
        if df.empty:
            print("No data available for error analysis plot")
            return
            
        # Calculate difference between predicted and actual path length
        df['length_diff'] = df['predicted_path_length'] - df['gt_path_length']
        
        # Count occurrences of each difference
        diff_counts = df['length_diff'].value_counts().reset_index()
        diff_counts.columns = ['Length Difference', 'Count']
        diff_counts = diff_counts.sort_values('Length Difference')
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(diff_counts['Length Difference'], diff_counts['Count'], color='skyblue')
        
        # Add counts above bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(diff_counts['Count']),
                    f'{int(height)}', ha='center', va='bottom')
        
        # Customize plot
        plt.xlabel('Predicted Path Length - Actual Path Length')
        plt.ylabel('Number of Examples')
        plt.title('Error Analysis: Difference Between Predicted and Actual Path Length')
        plt.grid(True, axis='y', alpha=0.3)
        plt.xticks(diff_counts['Length Difference'])
        
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        
    def generate_latex_table(self, output_file: str = "shortest_path_results.tex"):
        """Generate a LaTeX table showing performance by model and format"""
        df = pd.DataFrame(self.results_data)
        
        if df.empty:
            print("No data available for LaTeX table")
            return
            
        # Map names
        df['model'] = df['model'].map(lambda x: MODEL_NAME_MAP.get(x, x))
        df['format'] = df['format'].map(lambda x: FORMAT_NAME_MAP.get(x, x))
        
        # Create a pivot table for the LaTeX table
        pivot = pd.pivot_table(
            df, 
            values='avg_score',
            index=['model'],
            columns=['format', 'pseudonymized'],
            aggfunc='mean'
        )
        
        # Round values
        pivot = pivot.round(3)
        
        # Start building LaTeX table
        latex_lines = []
        latex_lines.append("\\begin{table}[ht]")
        latex_lines.append("\\centering")
        latex_lines.append("\\caption{ShortestPath Task Performance by Model and Format}")
        
        # Column definitions
        num_formats = len(pivot.columns) // 2  # Divide by 2 for original/pseudo
        latex_lines.append("\\begin{tabular}{l" + "cc" * num_formats + "}")
        latex_lines.append("\\toprule")
        
        # Column headers (formats)
        header1 = "\\multirow{2}{*}{Model}"
        for format_name in sorted(df['format'].unique()):
            header1 += f" & \\multicolumn{{2}}{{c}}{{{format_name}}}"
        latex_lines.append(header1 + " \\\\")
        
        # Subheaders (original/pseudo)
        header2 = ""
        for _ in range(num_formats):
            header2 += " & Orig. & Pseudo"
        latex_lines.append(header2 + " \\\\")
        latex_lines.append("\\midrule")
        
        # Data rows
        for model, row in pivot.iterrows():
            model_row = f"{model}"
            for format_name in sorted(df['format'].unique()):
                orig_value = row.get((format_name, False), np.nan)
                pseudo_value = row.get((format_name, True), np.nan)
                
                model_row += f" & {orig_value:.3f}" if not np.isnan(orig_value) else " & -"
                model_row += f" & {pseudo_value:.3f}" if not np.isnan(pseudo_value) else " & -"
            
            latex_lines.append(model_row + " \\\\")
        
        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write("\n".join(latex_lines))
            
    def generate_path_length_latex_table(self, output_file: str = "path_length_results.tex"):
        """Generate a LaTeX table showing performance by path length for each model"""
        df = pd.DataFrame(self.path_length_data)
        
        if df.empty:
            print("No data available for path length LaTeX table")
            return
            
        # Map names
        df['model'] = df['model'].map(lambda x: MODEL_NAME_MAP.get(x, x))
        
        # Average over formats
        model_length_scores = df.groupby(['model', 'path_length']).agg({
            'avg_score': 'mean',
            'num_examples': 'sum'
        }).reset_index()
        
        # Create a pivot table for the LaTeX table
        pivot = pd.pivot_table(
            model_length_scores, 
            values='avg_score',
            index=['model'],
            columns=['path_length'],
            aggfunc='mean'
        )
        
        # Round values
        pivot = pivot.round(3)
        
        # Add average column
        pivot['Avg'] = pivot.mean(axis=1)
        
        # Start building LaTeX table
        latex_lines = []
        latex_lines.append("\\begin{table}[ht]")
        latex_lines.append("\\centering")
        latex_lines.append("\\caption{ShortestPath Task Performance by Model and Path Length}")
        
        # Column definitions
        num_lengths = len(pivot.columns)
        latex_lines.append("\\begin{tabular}{l" + "c" * num_lengths + "}")
        latex_lines.append("\\toprule")
        
        # Column headers (path lengths)
        header = "Model"
        for length in pivot.columns:
            if length == 'Avg':
                header += f" & \\textbf{{Avg.}}"
            else:
                header += f" & Length {length}"
        latex_lines.append(header + " \\\\")
        latex_lines.append("\\midrule")
        
        # Data rows
        for model, row in pivot.iterrows():
            model_row = f"{model}"
            for length in pivot.columns:
                value = row[length]
                if length == 'Avg':
                    model_row += f" & \\textbf{{{value:.3f}}}"
                else:
                    model_row += f" & {value:.3f}" if not np.isnan(value) else " & -"
            
            latex_lines.append(model_row + " \\\\")
        
        # Add average row
        avg_row = "\\textbf{Average}"
        for length in pivot.columns:
            value = pivot[length].mean()
            if length == 'Avg':
                avg_row += f" & \\textbf{{{value:.3f}}}"
            else:
                avg_row += f" & {value:.3f}" if not np.isnan(value) else " & -"
        latex_lines.append("\\midrule")
        latex_lines.append(avg_row + " \\\\")
        
        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write("\n".join(latex_lines))

def main():
    print("Starting ShortestPath analysis...")
    analyzer = ShortestPathAnalyzer()
    analyzer.load_results()
    
    if not analyzer.results_data:
        print("\nNo ShortestPath results data was found!")
        return
    
    # Make shortest path figs directory
    os.makedirs("figs/shortest_path", exist_ok=True)
    
    print("\nGenerating model comparison plot...")
    analyzer.plot_model_comparison()
    
    print("\nGenerating format comparison plot...")
    analyzer.plot_format_comparison()
    
    print("\nGenerating path length performance plot...")
    analyzer.plot_path_length_performance()
    
    print("\nGenerating path length by model plot...")
    analyzer.plot_path_length_by_model()
    
    print("\nGenerating predicted vs actual length plot...")
    analyzer.plot_predicted_vs_actual_length()
    
    print("\nGenerating error analysis plot...")
    analyzer.plot_error_analysis()
    
    print("\nGenerating LaTeX tables...")
    analyzer.generate_latex_table()
    analyzer.generate_path_length_latex_table()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 