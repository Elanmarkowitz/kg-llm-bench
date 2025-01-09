import yaml
import random
from pathlib import Path
from kg_builder import KnowledgeGraph
from tasks import TripleRetrievalTask, ShortestPathTask, HighestDegreeNodeTask, AggByRelationTask, AggNeighborPropertiesTask

# Load the configuration file
with open('configs/construct_base_datasets_small.yaml', 'r') as file:
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

# Iterate over the task configurations and construct the base datasets
for task_config in config['task_configs']:
    task_type = task_config['type']
    task_class = task_classes.get(task_type)
    
    if not task_class:
        print(f"Unknown task type: {task_type}")
        continue
    
    conversion_config = {'type': "list_of_edges"}
    llm_config = {'model': 'gpt-4o-mini', 'provider': 'openai'}
    pseudonomizer_config = config['pseudonomizer_configs'][0]  # Assuming a single pseudonomizer config for simplicity

    task = task_class(conversion_config, llm_config, pseudonomizer_config,
                      base_dataset_file=task_config['base_dataset_file'])
    
    try:
        task.load_base_dataset()
        print(f"Skipping: Base dataset already exists for {task.base_data_file}")
        continue
    except ValueError:
        print("Dataset does not yet exist")
    task.construct_base_instances(kg, 
                                    num_instances=task_config['num_instances'], 
                                    num_seed_entities=task_config['num_seed_entities'], 
                                    max_edges=task_config['max_edges'])
    task.save_base_dataset()