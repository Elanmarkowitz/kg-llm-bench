import json
import os
import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

class HighestDegreeAnalyzer:
    def __init__(self, results_dir: str = "benchmark_data"):
        self.results_dir = Path(results_dir) / "HighestDegreeNode"
        self.results_data = []
        self.full_results_data = []
    def load_results(self):
        """Load all HighestDegree task results files"""
        print(f"\nSearching for HighestDegree results in: {self.results_dir}")
        format_dirs = list(self.results_dir.glob("*"))
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
                        
                        # Extract edge direction information
                        edge_directions = [r.get('edge_direction', 'unknown') for r in results if r is not None]
                        scores = [r.get('score', 0) for r in results if r is not None]
                        max_degrees = [r.get('max_degree', 0) for r in results if r is not None]
                        
                        # Group by edge direction
                        direction_data = {}
                        for direction, score, max_degree in zip(edge_directions, scores, max_degrees):
                            if direction not in direction_data:
                                direction_data[direction] = {'scores': [], 'max_degrees': []}
                            direction_data[direction]['scores'].append(score)
                            direction_data[direction]['max_degrees'].append(max_degree)
                            self.full_results_data.append({
                                'format': format_name,
                                'model': llm_dir.name,
                                'pseudonymized': is_pseudo,
                                'edge_direction': direction,
                                'score': score,
                                'max_degree': max_degree
                            })
                        # Calculate metrics for each direction
                        for direction, data in direction_data.items():
                            avg_score = np.mean(data['scores'])
                            std_score = np.std(data['scores'])
                            avg_max_degree = np.mean(data['max_degrees'])
                            std_max_degree = np.std(data['max_degrees'])
                            
                            self.results_data.append({
                                'format': format_name,
                                'model': llm_dir.name,
                                'pseudonymized': is_pseudo,
                                'edge_direction': direction,
                                'avg_score': avg_score,
                                'std_score': std_score,
                                'avg_max_degree': avg_max_degree,
                                'std_max_degree': std_max_degree,
                                'num_examples': len(data['scores'])
                            })

    def plot_edge_direction_comparison(self, output_file: str = "figs/highest_degree/highest_degree_edge_direction.pdf"):
        """Generate a bar plot comparing format performance by edge direction, averaged over models"""
        df = pd.DataFrame(self.results_data)
        
        # Map names
        df['model'] = df['model'].map(MODEL_NAME_MAP)
        df['format'] = df['format'].map(FORMAT_NAME_MAP)
        
        # Calculate average scores by edge direction and format, averaging over models
        avg_scores = df.groupby(['edge_direction', 'format', 'model'])['avg_score'].agg(['mean']).reset_index()
        avg_scores = avg_scores.groupby(['edge_direction', 'format'])['mean'].agg(['mean', 'min', 'max']).reset_index()

        # Create figure
        plt.figure(figsize=(6, 3))
        
        # Set up the plot
        bar_width = 0.15
        formats = sorted(avg_scores['format'].unique())
        edge_directions = sorted(avg_scores['edge_direction'].unique())
        
        x = np.arange(len(edge_directions))
        
        # Plot bars for each format
        colors = plt.cm.Set2(np.linspace(0, 1, len(formats)))
        for i, format_name in enumerate(formats):
            format_data = avg_scores[avg_scores['format'] == format_name]
            
            # Calculate error bars using min and max
            yerr = [format_data['mean'] - format_data['min'], format_data['max'] - format_data['mean']]

            plt.bar(x + i * bar_width, 
                   format_data['mean'],
                   bar_width,
                   label=format_name,
                   color=colors[i],
                   yerr=None, # No error bars
                   capsize=5)
        
        # Customize plot
        plt.xlabel('Edge Direction')
        plt.ylabel('Average Score')
        plt.title('Format Performance by Edge Direction\n(Averaged over Models)')
        plt.xticks(x + bar_width * (len(formats) - 1) / 2, edge_directions)
        plt.legend(title='Format', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, axis='y', alpha=0.3)
        
        # Ensure the plot fits with the legend
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_model_edge_direction_comparison(self, output_file: str = "figs/highest_degree/highest_degree_edge_direction.pdf"):
        """Generate a bar plot comparing format performance by edge direction, averaged over models"""
        df = pd.DataFrame(self.results_data)
        
        # Map names
        df['model'] = df['model'].map(MODEL_NAME_MAP)
        df['format'] = df['format'].map(FORMAT_NAME_MAP)
        
        # Calculate average scores by edge direction and format, averaging over models
        avg_scores = df.groupby(['edge_direction', 'format', 'model'])['avg_score'].agg(['mean']).reset_index()
        avg_scores = avg_scores.groupby(['edge_direction', 'model'])['mean'].agg(['mean', 'min', 'max']).reset_index()

        # Create figure
        plt.figure(figsize=(6, 3))
        
        # Set up the plot
        bar_width = 0.15
        formats = sorted(avg_scores['format'].unique())
        edge_directions = sorted(avg_scores['edge_direction'].unique())
        
        x = np.arange(len(edge_directions))
        
        # Plot bars for each format
        colors = plt.cm.Set2(np.linspace(0, 1, len(formats)))
        for i, format_name in enumerate(formats):
            format_data = avg_scores[avg_scores['format'] == format_name]
            
            # Calculate error bars using min and max
            yerr = [format_data['mean'] - format_data['min'], format_data['max'] - format_data['mean']]

            plt.bar(x + i * bar_width, 
                   format_data['mean'],
                   bar_width,
                   label=format_name,
                   color=colors[i],
                   yerr=None, # No error bars
                   capsize=5)
        
        # Customize plot
        plt.xlabel('Edge Direction')
        plt.ylabel('Average Score')
        plt.title('Format Performance by Edge Direction\n(Averaged over Models)')
        plt.xticks(x + bar_width * (len(formats) - 1) / 2, edge_directions)
        plt.legend(title='Format', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, axis='y', alpha=0.3)
        
        # Ensure the plot fits with the legend
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

    def generate_edge_direction_latex_table(self, output_file: str = "figs/highest_degree/highest_degree_edge_direction.tex"):
        """Generate a LaTeX table showing performance by edge direction"""
        df = pd.DataFrame(self.results_data)
        
        # Map names
        df['model'] = df['model'].map(MODEL_NAME_MAP)
        df['format'] = df['format'].map(FORMAT_NAME_MAP)
        
        # Calculate statistics
        stats = df.groupby(['format', 'model', 'edge_direction']).agg({
            'avg_score': ['mean', 'std'],
            'avg_max_degree': ['mean', 'std']
        }).round(3)
        
        # Flatten column names
        stats.columns = ['score_mean', 'score_std', 'degree_mean', 'degree_std']
        stats = stats.reset_index()
        
        # Start building LaTeX table
        latex_lines = []
        latex_lines.append("\\begin{table}[ht]")
        latex_lines.append("\\centering")
        latex_lines.append("\\caption{HighestDegree Task Performance by Edge Direction}")
        latex_lines.append("\\begin{tabular}{llllrr}")
        latex_lines.append("\\toprule")
        latex_lines.append("Format & Model & Direction & Score & Avg Max Degree \\\\")
        latex_lines.append("\\midrule")
        
        # Add rows
        current_format = None
        for _, row in stats.iterrows():
            if current_format != row['format']:
                if current_format is not None:
                    latex_lines.append("\\midrule")
                current_format = row['format']
                format_str = row['format']
            else:
                format_str = ""
            
            latex_lines.append(
                f"{format_str} & {row['model']} & {row['edge_direction']} & "
                f"{row['score_mean']:.3f} $\\pm$ {row['score_std']:.3f} & "
                f"{row['degree_mean']:.1f} $\\pm$ {row['degree_std']:.1f} \\\\"
            )
        
        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write("\n".join(latex_lines))

    def plot_binned_degree_performance(self, output_file: str = "figs/highest_degree/degree_binned_performance.pdf", n_bins: int = 10):
        """Generate a scatter plot showing binned max degree vs average accuracy"""
        df = pd.DataFrame(self.full_results_data)  # Changed to use full_results_data
        
        # Map names
        df['model'] = df['model'].map(MODEL_NAME_MAP)
        df['format'] = df['format'].map(FORMAT_NAME_MAP)
        
        # Create degree bins
        # df['degree_bin'] = pd.qcut(df['max_degree'], q=n_bins, labels=False, duplicates='drop')
        
        # Calculate mean degree and score for each bin and edge direction
        binned_data = df.groupby(['max_degree', 'edge_direction']).agg({
            'score': ['mean', 'count']
        }).reset_index()
        # Create the plot
        plt.figure(figsize=(10, 6))
        # Plot points for each edge direction
        for direction in sorted(binned_data['edge_direction'].unique()):
            direction_data = binned_data[binned_data['edge_direction'] == direction]
            plt.scatter(direction_data['max_degree'], 
                        direction_data['score']['mean'], 
                        s=direction_data['score']['count'],  # Size based on count
                        label=direction,
                        alpha=0.8)
        
        plt.xlabel('Average Max Degree')
        plt.ylabel('Average Score')
        plt.title('Performance vs Node Degree\n(Averaged across all models and formats)')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Edge Direction')
        
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

def main():
    print("Starting HighestDegree analysis...")
    analyzer = HighestDegreeAnalyzer()
    analyzer.load_results()
    
    if not analyzer.results_data:
        print("\nNo HighestDegree results data was found!")
        return
    
    # Make figs/highest_degree directory
    os.makedirs("figs/highest_degree", exist_ok=True)

    print("\nGenerating edge direction comparison plot...")
    analyzer.plot_edge_direction_comparison()
    
    print("\nGenerating edge direction LaTeX table...")
    analyzer.generate_edge_direction_latex_table()
    
    print("\nGenerating binned degree performance plot...")
    analyzer.plot_binned_degree_performance()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 