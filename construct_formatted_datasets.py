import yaml
import random
from pathlib import Path
from kg_builder import KnowledgeGraph
from tasks import TripleRetrievalTask, ShortestPathTask, HighestDegreeNodeTask, AggByRelationTask, AggNeighborPropertiesTask
from tasks.base_task import BaseTask
import argparse


# Argument parser for configuration file
parser = argparse.ArgumentParser(description='Construct formatted datasets.')
parser.add_argument('--config', type=str, default='configs/construct_formatted_datasets_small.yaml',
                    help='Path to the configuration file')
args = parser.parse_args()

# Load the configuration file
with open(args.config, 'r') as file:
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

# Iterate over the task configurations and construct the formatted datasets
for task_config in config['task_configs']:
    task_type = task_config['type']
    task_class = task_classes.get(task_type)
    
    if not task_class:
        print(f"Unknown task type: {task_type}")
        continue
    
    llm_config = {'model': None, 'provider': None}

    for conversion_config in config['conversion_configs']:
        for pseudonomizer_config in config['pseudonomizer_configs']:
            dataset_file = task_config['base_dataset_file']  # use same filename as base data
            task: BaseTask = task_class(conversion_config, llm_config, pseudonomizer_config,
                                        base_dataset_file=task_config['base_dataset_file'],
                                        dataset_file=dataset_file)
            
            try:
                task.load_base_dataset()
            except ValueError:
                print(f"Skipping: Base dataset not found {task_config['base_dataset_file']}")
                continue
            
            try:
                task.load_formatted_dataset()
                if not args.reformat:
                    print(f"Skipping: Formatted dataset already exists {task.dataset_file}")
                    continue
                else:
                    print(f"Reformatting existing dataset {task.dataset_file}")
            except ValueError:
                print("Dataset does not yet exist, creating")
            task.construct_formatted_instances()
            task.save_formatted_dataset()