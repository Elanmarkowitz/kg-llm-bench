import yaml
import random
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from kg_builder import KnowledgeGraph
from tasks import TripleRetrievalTask, ShortestPathTask, HighestDegreeNodeTask, AggByRelationTask, AggNeighborPropertiesTask
from tasks.base_task import BaseTask

# Load the configuration file
with open('configs/run_small_datasets.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load the knowledge graph
kg = KnowledgeGraph()
kg.load_entities('data/countries/entities.tsv')
kg.load_core_nodes('data/countries/nodes.tsv')
kg.load_relations('data/countries/relations.tsv')
kg.load_edges('data/countries/edges.tsv')
kg.load_attributes('data/countries/attributes.tsv')

# Define a mapping from task type to task class
task_classes = {
    'TripleRetrievalTask': TripleRetrievalTask,
    'ShortestPathTask': ShortestPathTask,
    'HighestDegreeNodeTask': HighestDegreeNodeTask,
    'AggByRelationTask': AggByRelationTask,
    'AggNeighborPropertiesTask': AggNeighborPropertiesTask
}

# Iterate over the task configurations and run the experiments
for task_config in config['task_configs']:
    task_type = task_config['type']
    task_class = task_classes.get(task_type)
    
    if not task_class:
        print(f"Unknown task type: {task_type}")
        continue
    
    for llm_config in config['llm_configs']:
        for conversion_config in config['conversion_configs']:
            for pseudonomizer_config in config['pseudonomizer_configs']:
                task: BaseTask = task_class(conversion_config, llm_config, pseudonomizer_config,
                                            base_dataset_file=task_config['base_dataset_file'],
                                            dataset_file=task_config['dataset_file'],
                                            results_file=task_config['results_file'])
                
                task.load_base_dataset()
                
                task.load_formatted_dataset()
                
                try:
                    task.load_results()
                    print(f"Skipping: Results already exist for {task.results_file}")
                    continue
                except ValueError:
                    print("No results found, running task")
                print(f"Running task: {task_type} with LLM: {llm_config['model']}")
                task.run()
                
                print(f"Finished running task: {task_type} with LLM: {llm_config['model']}")

print("All experiments completed.")