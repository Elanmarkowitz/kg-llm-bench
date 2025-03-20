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

class AggByRelationAnalyzer:
    def __init__(self, results_dir: str = "benchmark_data"):
        self.results_dir = Path(results_dir) / "AggByRelation"
        self.results_data = []
        self.full_results_data = []
    def load_results(self):
        """Load all AggByRelation task results files"""
        print(f"\nSearching for AggByRelation results in: {self.results_dir}")
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
                        
                        # Extract relation information
                        relations = [r.get('relation', 'unknown') for r in results if r is not None]
                        scores = [r.get('score', 0) for r in results if r is not None]
                        answers = [r.get('answer', [0])[0] for r in results if r is not None]
                        
                        # Group by relation
                        relation_data = {}
                        for relation, score, answer in zip(relations, scores, answers):
                            if relation not in relation_data:
                                relation_data[relation] = {'scores': [], 'answers': []}
                            relation_data[relation]['scores'].append(score)
                            relation_data[relation]['answers'].append(answer)
                            self.full_results_data.append({
                                'format': format_name,
                                'model': llm_dir.name,
                                'pseudonymized': is_pseudo,
                                'relation': relation,
                                'score': score,
                                'answer': int(answer)
                            })
                        
                        self.results_data.append({
                            'format': format_name,
                            'model': llm_dir.name,
                            'pseudonymized': is_pseudo,
                            'avg_score': np.mean(scores),
                            'std_score': np.std(scores),
                            'num_examples': len(scores)
                        })
                        

    def plot_relation_comparison(self, output_file: str = "agg_by_relation_comparison.pdf"):
        """Generate a bar plot comparing format performance by relation, averaged over models"""
        df = pd.DataFrame(self.results_data)
        
        # Map names
        df['model'] = df['model'].map(MODEL_NAME_MAP)
        df['format'] = df['format'].map(FORMAT_NAME_MAP)
        
        # Calculate average scores by relation and format, averaging over models
        avg_scores = df.groupby(['relation', 'format', 'model'])['avg_score'].agg(['mean']).reset_index()
        avg_scores = avg_scores.groupby(['relation', 'format'])['mean'].agg(['mean', 'min', 'max']).reset_index()

        # Create figure
        plt.figure(figsize=(6, 3))
        
        # Set up the plot
        bar_width = 0.15
        formats = sorted(avg_scores['format'].unique())
        relations = sorted(avg_scores['relation'].unique())
        
        x = np.arange(len(relations))
        
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
        plt.xlabel('Relation')
        plt.ylabel('Average Score')
        plt.title('Format Performance by Relation\n(Averaged over Models)')
        plt.xticks(x + bar_width * (len(formats) - 1) / 2, relations)
        plt.legend(title='Format', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, axis='y', alpha=0.3)
        
        # Ensure the plot fits with the legend
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

    def generate_relation_latex_table(self, output_file: str = "agg_by_relation.tex"):
        """Generate a LaTeX table showing performance by relation"""
        df = pd.DataFrame(self.results_data)
        
        # Map names
        df['model'] = df['model'].map(MODEL_NAME_MAP)
        df['format'] = df['format'].map(FORMAT_NAME_MAP)
        
        # Calculate statistics
        stats = df.groupby(['format', 'model', 'relation']).agg({
            'avg_score': ['mean', 'std'],
            'avg_answer': ['mean', 'std']
        }).round(3)
        
        # Flatten column names
        stats.columns = ['score_mean', 'score_std', 'answer_mean', 'answer_std']
        stats = stats.reset_index()
        
        # Start building LaTeX table
        latex_lines = []
        latex_lines.append("\\begin{table}[ht]")
        latex_lines.append("\\centering")
        latex_lines.append("\\caption{AggByRelation Task Performance by Relation}")
        latex_lines.append("\\begin{tabular}{llllrr}")
        latex_lines.append("\\toprule")
        latex_lines.append("Format & Model & Relation & Score & Avg Answer \\\\")
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
                f"{format_str} & {row['model']} & {row['relation']} & "
                f"{row['score_mean']:.3f} $\\pm$ {row['score_std']:.3f} & "
                f"{row['answer_mean']:.1f} $\\pm$ {row['answer_std']:.1f} \\\\"
            )
        
        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write("\n".join(latex_lines))

    def plot_binned_answer_performance(self, output_file: str = "answer_binned_performance.pdf", n_bins: int = 10):
        """Generate a scatter plot showing binned answer vs average accuracy"""
        df = pd.DataFrame(self.full_results_data)  # Changed to use full_results_data
        
        # Map names
        df['model'] = df['model'].map(MODEL_NAME_MAP)
        df['format'] = df['format'].map(FORMAT_NAME_MAP)
        
        # Create answer bins
        bins = np.concatenate([np.arange(0, 10), np.arange(10, df['answer'].max() + 10, 10)])
        bin_size = np.concatenate([np.repeat(1,10), np.repeat(10, len(bins) - 10)])
        df['answer_bin'] = pd.cut(df['answer'], bins=bins, right=False, labels=False)
        

        # Calculate mean answer and score for each bin and relation
        binned_data = df.groupby(['answer_bin']).agg({
            'score': ['mean', 'count'],
            'answer': ['mean', 'count']
        }).reset_index()
        binned_data['bin_start'] = bins[binned_data['answer_bin']]
        binned_data['bin_end'] = bins[binned_data['answer_bin']] + bin_size[binned_data['answer_bin']]
        # Create the plot
        plt.figure(figsize=(5, 3))
        # Plot points for each relation
        # Create a bar chart showing the mean performance for different answer values
        plt.bar((binned_data['bin_start'] + binned_data['bin_end'])/2, binned_data['score']['mean'], 
                width=binned_data['bin_end'] - binned_data['bin_start'] - 0.2, 
                color='skyblue', alpha=0.7)
        
        # Add labels and title
        plt.xlabel('Answer Value')
        plt.ylabel('Mean Score')
        plt.title('AggByRelation Task Performance by Answer Value')
        
        # Add counts labels on top of the bars
        for index, value in enumerate(binned_data['score']['count']):
            plt.text((binned_data['bin_start'][index] + binned_data['bin_end'][index])/2, binned_data['score']['mean'][index], f"{value}", ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

def main():
    print("Starting AggByRelation analysis...")
    analyzer = AggByRelationAnalyzer()
    analyzer.load_results()
    
    if not analyzer.results_data:
        print("\nNo AggByRelation results data was found!")
        return
    
    # print("\nGenerating relation comparison plot...")
    # analyzer.plot_relation_comparison()
    
    # print("\nGenerating relation LaTeX table...")
    # analyzer.generate_relation_latex_table()
    
    # print("\nGenerating binned answer performance plot...")
    analyzer.plot_binned_answer_performance()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 