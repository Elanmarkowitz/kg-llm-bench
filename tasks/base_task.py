from copy import deepcopy
import json
import os
import random
import numpy as np
from kg_builder import KnowledgeGraph, KnowledgeGraphTextPresenter, Triple
from llm.llm import LLM
from pseudonymizer import Pseudonymizer
from samplers import graph_samplers
from pathlib import PosixPath

class BaseTask:
    """Tasks are the main function that runs things. They handle sampling the kg, pseudonimizing, creating task question, 
    creating the question, making the llm request, and evaluating the response."""

    def __init__(self, 
                 task_name, 
                 conversion_config, 
                 llm_config, 
                 pseudonomizer_config, 
                 dataset_file=None, 
                 results_file=None):
        self.task_name = task_name
        self.llm_config = llm_config
        self.conversion_config = conversion_config
        self.text_presenter_type = conversion_config['type']
        self.llm_type = llm_config['model']
        self.pseudonomizer_config = pseudonomizer_config
        
        self.model = LLM(**llm_config)
        self.text_presenter = KnowledgeGraphTextPresenter(**conversion_config)
        if pseudonomizer_config is not None:
            self.pseudonomizer = Pseudonymizer(**pseudonomizer_config)
            self.text_presenter_type += "-pseudo"
        else: 
            self.pseudonomizer = None
        
        self.results = []
        self.data = []
        self.task_dir = PosixPath('benchmark_data') / task_name
        os.makedirs(self.task_dir, exist_ok=True)

        self.dataset_instance_dir = self.task_dir / self.text_presenter_type

        os.makedirs(os.path.join(self.dataset_instance_dir, 'results'), exist_ok=True)
        os.makedirs(os.path.join(self.task_dir, 'kg'), exist_ok=True)
        os.makedirs(os.path.join(self.dataset_instance_dir, 'kg'), exist_ok=True)

        self.dataset_file = dataset_file
        if not self.dataset_file:
            self.dataset_file = os.path.join(self.dataset_instance_dir, f'{task_name}_dataset.json')

        self.results_file = results_file
        if not self.results_file:
            self.results_file = self.dataset_instance_dir / 'results' / f'{self.llm_type}_results.json'
       
        self.base_data_file = self.task_dir / f'{self.task_name}_base_dataset.json'

    def run(self):
        raise NotImplementedError('You must implement the run method in your task class')
    
    def construct_instances(self, kg: KnowledgeGraph, num_instances=10):
        raise NotImplementedError('You must implement the construct_instances method in your task class')

    def reformat_instances(self):
        """Should reformat self.data using self.text_presenter and self.pseudonomizer"""
        raise NotImplementedError('You must implement the reformat_instances method in your task class')

    def save_results(self):
        self.save_data(file_path=self.results_file, save_data=self.results)

    def save_dataset(self):
        self.save_data(file_path=self.dataset_file, save_data=self.data)

    def save_base_data(self):
        file_path = self.base_data_file
        save_data = deepcopy(self.data)

        for instance in save_data:
            if instance is None: # To Do: check edge cases.
                continue
            if 'kg' in instance:
                kg: KnowledgeGraph = instance.pop('kg')
                id = instance['id']
                kg_path = os.path.join(self.task_dir, 'kg', f"kg_{id:04d}.pkl") #TODO: We need a way to have different kg folders if we create new versions of a task
                instance['kg_path'] = kg.save_kg(kg_path)
            
            if 'text_kg' in instance:
                instance.pop('text_kg')
            
            if 'prompt' in instance:
                instance.pop('prompt')

        if os.path.exists(file_path):
            filename = os.path.split(file_path)[-1]
            base, ext = os.path.splitext(filename)
            counter = 1
            new_dataset_file = f"{base}_{counter}{ext}"
            while os.path.exists(new_dataset_file):
                counter += 1
                new_dataset_file = f"{base}_{counter}{ext}"
            self.base_data_file = self.task_dir / new_dataset_file
            file_path = self.base_data_file
        with open(file_path, 'w') as f:
            json.dump(save_data, f, cls=CustomJSONEncoder, indent=4)

    def save_data(self, file_path=None, save_data=None):
        file_path = file_path or self.dataset_file
        save_data = deepcopy(save_data) or deepcopy(self.data)

        if not file_path:
            raise ValueError('Dataset file path not set. Please set the dataset file path before saving the dataset.')
        
        for instance in save_data:
            if instance is None: # To Do: check for edge cases
                continue
            if 'kg' in instance:
                kg: KnowledgeGraph = instance.pop('kg')
                id = instance['id']
                kg_path = os.path.join(self.dataset_instance_dir, 'kg', f"kg_{id:04d}.pkl")
                instance['kg_path'] = kg.save_kg(kg_path)

        if os.path.exists(file_path):
            base, ext = os.path.splitext(file_path)
            counter = 1
            new_dataset_file = f"{base}_{counter}{ext}"
            while os.path.exists(new_dataset_file):
                counter += 1
                new_dataset_file = f"{base}_{counter}{ext}"
            self.dataset_file = new_dataset_file
        with open(file_path, 'w') as f:
            json.dump(save_data, f, cls=CustomJSONEncoder)
    
    def load_dataset(self):
        if self.dataset_file and os.path.exists(self.dataset_file):
            with open(self.dataset_file, 'r') as f:
                self.data = json.load(f)
                for instance in self.data:
                    if instance is None: # To Do:
                        continue
                    if 'kg_path' in instance:
                        kg_path = instance['kg_path']
                        instance['kg'] = KnowledgeGraph().load_kg(kg_path)
                
class TripleRetrievalTask(BaseTask):

    def __init__(self, conversion_config, llm_config, pseudonomizer_config, dataset_file=None, results_file=None):
        super().__init__("TripleRetrieval", 
                         conversion_config, 
                         llm_config, 
                         pseudonomizer_config, 
                         dataset_file, 
                         results_file)

    def run(self):
        self.results = deepcopy(self.data)
        for instance in self.results:
            if instance is None:  # To Do:
                continue
            prompt = instance['prompt']
            response = self.model(prompt)
            instance['response'] = response
            instance['score'] = self.evaluate_response(response, instance['answer'])
        self.save_results()
    
    def evaluate_response(self, response, answer):
        return 1.0 if response == answer else 0.0

    def construct_instances(self, kg: KnowledgeGraph, 
                            num_instances=10, 
                            num_seed_entities=10, 
                            max_edges=100):
        """Constructs instances for the task."""
        for instance in range(num_instances):
            seed_entities = random.sample(list(kg.core_nodes.keys()), num_seed_entities)
            self.data.append(self.construct_instance(kg, seed_entities, max_edges, instance))

    def construct_instance(self, kg: KnowledgeGraph, seed_entities, max_edges=100, instance_id=0):
        # Retrieve triples based on seed entities
        sampled_kg = graph_samplers.sample_ego_graph_from_kg(kg, seed_entities, radius=1)
        sampled_kg = graph_samplers.prune_kg(sampled_kg, max_edges=max_edges, max_degree=20)

        text_kg = self.text_presenter.to_list_of_edges(sampled_kg)

        triple_sample = random.choice([triple for triple in sampled_kg.graph.edges(data='relation_id')])
        triple_sample = Triple(head=kg.entities[triple_sample[0]], 
                               relation=kg.relations[triple_sample[2]], 
                               tail=kg.entities[triple_sample[1]])

        if random.randint(0, 1) == 0:
            triple, corruption_type = triple_sample, "None"
        else:
            triple, corruption_type = self.corrupt_triplet(triple_sample, sampled_kg)
        
        question = f"Is the following triplet fact present in the knowledge graph (Yes/No)? ({triple.head.label}, {triple.relation.label}, {triple.tail.label})"

        prompt = self.structure_prompt(question, text_kg)

        answer = "Yes" if corruption_type == "None" else "No"

        return {
            'id': instance_id,
            'prompt': prompt,
            'question': question,
            'text_kg': text_kg,
            'triple': triple,
            'answer': answer,
            'corruption_type': corruption_type,
            'seed_entities': seed_entities,
            'kg': kg # TODO: Find way to save kg
        }
    
    def reformat_instances(self):
        """Reformats self.data using self.text_presenter."""
        for instance in self.data:
            kg = instance['kg']
            text_kg = self.text_presenter.to_list_of_edges(kg)
            instance['text_kg'] = text_kg

    def corrupt_triplet(self, triple: Triple, kg: KnowledgeGraph):
        corrupted_triplet = deepcopy(triple)
        # Get all possible entities that don't create a true triple when substituted
        corruption_type = random.choice(['head', 'relation', 'tail'])
        if corruption_type == 'head':  # If corrupting subject or object
            candidate_entities = [
                head for head in kg.entities.values()
                if (not kg.graph.has_edge(head.entity_id, triple.tail.entity_id)) 
                or kg.graph.get_edge_data(head.entity_id, triple.tail.entity_id)['relation'] != triple.relation.label
            ]
            corrupted_triplet.head = random.choice(candidate_entities)
        
        elif corruption_type == 'tail':  # If corrupting tail/object
            candidate_entities = [
                tail for tail in kg.entities.values()
                if (not kg.graph.has_edge(triple.head.entity_id, tail)) or 
                kg.graph.get_edge_data(triple.head.entity_id, tail)['relation'] != triple.relation.label
            ]
            corrupted_triplet.tail = random.choice(candidate_entities)

        else: # If corrupting relation
            candidate_relations = list(r for r in kg.relations.values() if r.label != triple.relation.label)
            corrupted_triplet.relation = random.choice(candidate_relations)

        return corrupted_triplet, corruption_type

    def structure_prompt(self, question, text_kg):
        intro = f"Your job is to answer questions using the following knowledge graph. {self.text_presenter.get_description()}. You must rely exclusively on the information presented in the Knowledge Graph to answer questions."
        prompt = f"{intro}\n\nKnowledge Graph:\n{text_kg}\n\n{question}"
        return prompt
    

class CustomJSONEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)  # Convert np.int64 to Python int
        elif isinstance(obj, np.floating):
            return float(obj)  # Convert np.float64 to Python float
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy arrays to lists
        elif hasattr(obj, 'to_dict'): # Handle objects with a `to_dict` method
            return obj.to_dict()
        return super(CustomJSONEncoder, self).default(obj)


# sample_kg: (KnowledgeGraph -> KnowledgeGraph)
# text_presenter: (KnowledgeGraph -> str representing the KG)

if __name__ == "__main__":
    kg = KnowledgeGraph()

    # Load entities and nodes
    kg.load_entities('../data/countries/entities.tsv')
    kg.load_core_nodes('../data/countries/nodes.tsv')

    # Load relations
    kg.load_relations('../data/countries/relations.tsv')

    # Load edges and attributes
    kg.load_edges('../data/countries/edges.tsv')
    kg.load_attributes('../data/countries/attributes.tsv')

    # Print graph information
    # kg.print_graph_info()

    # Visualize the graph
    # kg.visualize_graph()
    # kg.get_ego_graph(3393, radius=1)

    # Create an instance of KnowledgeGraphTextPresenter and extract triplets
    conversion_config = {'type': "list_of_edges"}
    llm_config = {'model': 'gpt-4o-mini', 'provider': 'openai'}
    pseudonomizer_config = {'pseudonym_file': '../data/countries/pseudonym_data/country_pseudonyms.tsv'}


    task = TripleRetrievalTask(conversion_config, llm_config, pseudonomizer_config)

    seed_entities = random.sample(list(kg.core_nodes.keys()), 10)

    task.load_dataset()

    # breakpoint()

    task.construct_instances(kg, num_instances=10, num_seed_entities=10, max_edges=100)

    task.save_base_data()
    task.save_dataset()

    task.run()
