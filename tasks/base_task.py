from copy import deepcopy
import json
import os
from pathlib import Path
import random
import re
import uuid
from tqdm import tqdm
import numpy as np
from kg_builder import KnowledgeGraph, KnowledgeGraphTextPresenter, Triple
from llm.llm import LLM
from pseudonymizer import Pseudonymizer
from samplers import graph_samplers
from pathlib import Path

class InstanceConstructionError(Exception):
    """Custom exception to be raised when instance construction fails."""
    pass

class BaseTask:
    """Tasks are the main function that runs things. They handle sampling the kg, pseudonimizing, creating task question, 
    creating the question, making the llm request, and evaluating the response."""

    def __init__(self, 
                 task_name, 
                 conversion_config, 
                 llm_config, 
                 pseudonomizer_config, 
                 base_dataset_file=None,
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
        if pseudonomizer_config:
            self.pseudonomizer = Pseudonymizer(**pseudonomizer_config)
            self.text_presenter_type += "-pseudo"
        else: 
            self.pseudonomizer = None
        
        self.results = []
        self.base_data = []
        self.formatted_data = []

        self.task_dir = Path('benchmark_data') / task_name
        os.makedirs(self.task_dir, exist_ok=True)
        os.makedirs(self.task_dir/'kg', exist_ok=True)
        os.makedirs(self.task_dir/'pseudo_kg', exist_ok=True)

        self.dataset_instance_dir = self.task_dir / self.text_presenter_type
        os.makedirs(self.dataset_instance_dir/'results', exist_ok=True)

        if dataset_file:
            self.dataset_file = self.dataset_instance_dir / dataset_file
        else:
            self.dataset_file = self.dataset_instance_dir / f'{task_name}_dataset.json'
        
        if results_file:
            self.results_file = self.dataset_instance_dir / 'results' / results_file
        else:
            self.results_file = self.dataset_instance_dir / 'results' / f'{self.llm_type}_results.json'
       
        if base_dataset_file:
            self.base_data_file = self.task_dir / base_dataset_file
        else:
            self.base_data_file = self.task_dir / f'{self.task_name}_base_dataset.json'

    def pseudonymize_kg(self, kg: KnowledgeGraph):
        if self.pseudonomizer:
            self.pseudonomizer.clear_mapping()
            self.pseudonomizer.create_mapping(kg)
            return self.pseudonomizer.pseudonymize(kg)
        else:
            raise ValueError("No pseudonomizer configured")
    
    def construct_base_instances(self, kg: KnowledgeGraph,
                                 num_instances=10, 
                                 num_seed_entities=10, 
                                 max_edges=100):
        """Constructs instances for the task."""
        print("Constructing base data")
        instances = 0
        pbar = tqdm(total=num_instances)
        while instances < num_instances:
            seed_entities = random.sample(list(kg.core_nodes.keys()), num_seed_entities)
            try:
                instance = self.construct_instance(kg, seed_entities, instance_id=instances, max_edges=max_edges)
                self.base_data.append(instance)
                instances += 1
                pbar.update()
            except InstanceConstructionError:
                print("Failed to construct instance, retrying")
    
    def construct_instance(self, kg: KnowledgeGraph, seed_entities, max_edges=100, instance_id=0):
        raise NotImplementedError('You must implement the construct_instance method in your task class')

    def construct_formatted_instances(self):
        self.formatted_data = deepcopy(self.base_data)
        for instance in self.formatted_data:
            assert 'kg_path' in instance
            kg = instance.pop('kg')
            if self.pseudonomizer:
                if not 'pseudonomizer_mapping' in instance:
                        raise ValueError("Pseudonomizer config set but no pseudonomizer mapping in the base data")
                self.pseudonomizer.load_mapping(instance['pseudonomizer_mapping'])
                if 'pseudo_kg' in instance:
                    kg = instance.pop('pseudo_kg')
                else:
                    kg = self.pseudonomizer.pseudonymize(kg)
                # answer is Yes/No so no change needed
            text_kg = self.text_presenter.convert(kg)
            self.format_instance(instance, text_kg)

    def format_instance(self, instance, text_kg):
        """Should reformat self.data using self.text_presenter and self.pseudonomizer"""
        raise NotImplementedError('You must implement the reformat_instances method in your task class')

    def run(self):
        if not self.llm_config:
            raise ValueError("No LLM configured")
        self.results = deepcopy(self.formatted_data)
        for instance in self.results:
            if instance is None:
                continue
            prompt = instance['prompt']
            response = self.model(prompt)
            instance['response'] = response
            instance['score'] = self.evaluate_response(response, instance['answer'])
        self.save_results()

    def save_results(self):
        print(f"Saving results to {self.results_file}")
        save_path = self._save_data(file_path=self.results_file, save_data=self.results)
        self.results_file = save_path

    def save_formatted_dataset(self):
        print(f"Saving formatted dataset to {self.dataset_file}")
        save_path = self._save_data(file_path=self.dataset_file, save_data=self.formatted_data)
        self.dataset_file = save_path

    def save_base_dataset(self):
        print(f"Saving base dataset to {self.base_data_file}")
        save_path = self._save_data(file_path=self.base_data_file, save_data=self.base_data)
        self.base_data_file = save_path
        self.load_base_dataset()  # load so that we have kg_path and pseudo_kg_path

    def load_base_dataset(self):
        print("Loading base data")
        self.base_data = self._load_data(self.base_data_file)

    def load_formatted_dataset(self):
        print("Loading formatted data")
        self.formatted_data = self._load_data(self.dataset_file)

    def load_results(self):
        print("Loading results data")
        self.results = self._load_data(self.results_file)

    def _save_data(self, file_path, save_data):
        save_id = str(uuid.uuid4())[:6]
        save_data = deepcopy(save_data)

        if not file_path:
            raise ValueError('No filepath given')
        
        for instance in save_data:
            if instance is None:
                continue
            if 'kg' in instance:
                kg: KnowledgeGraph = instance.pop('kg')
                if 'kg_path' not in instance:
                    id = instance['id']
                    kg_path = self.task_dir/'kg'/f"kg_{save_id}_{id:04d}.pkl"
                    instance['kg_path'] = kg.save_kg(kg_path)

            if 'pseudo_kg' in instance:
                pseudo_kg: KnowledgeGraph = instance.pop('pseudo_kg')
                if 'pseudo_kg_path' not in instance:
                    id = instance['id']
                    pseudo_kg_path = self.task_dir/'pseudo_kg'/f"kg_{save_id}_{id:04d}.pkl"
                    instance['pseudo_kg_path'] = pseudo_kg.save_kg(pseudo_kg_path)

        if os.path.exists(file_path):
            file_path = self.get_next_filepath(file_path)
        with open(file_path, 'w') as f:
            json.dump(save_data, f, cls=CustomJSONEncoder, indent=4)
        return file_path
    
    @staticmethod
    def get_next_filepath(path: Path) -> Path:
        """
        Get the next available filename by incrementing a number suffix.
        If the file doesn't exist, returns the original path.
        If the file exists, finds the next available number.
        """
        if not path.exists():
            return path
        
        parent = path.parent
        stem = path.stem
        suffix = path.suffix
        
        # Check if stem already ends with _## (any number of digits)
        pattern = r"(.+?)_(\d+)$"
        match = re.match(pattern, stem)
        
        if match:
            # If filename already has a number, start from that base
            base = match.group(1)
            # Find all existing numbered files with same base
            existing_files = sorted([
                int(re.match(pattern, p.stem).group(2))
                for p in parent.glob(f"{base}_[0-9]*{suffix}")
                if re.match(pattern, p.stem)
            ])
        else:
            base = stem
            # Find all existing numbered files
            existing_files = sorted([
                int(re.match(pattern, p.stem).group(2))
                for p in parent.glob(f"{base}_[0-9]*{suffix}")
                if re.match(pattern, p.stem)
            ])
        
        # Find first available number
        if not existing_files:
            return parent / f"{base}_01{suffix}"
        
        # Find first gap in sequence, or use max + 1
        for i, num in enumerate(existing_files, start=1):
            if i != num:
                return parent / f"{base}_{i:02d}{suffix}"
        return parent / f"{base}_{(max(existing_files) + 1):02d}{suffix}"

    def _load_data(self, filepath):
        if not filepath or not os.path.exists(filepath):
            raise ValueError(f"{filepath} does not exist, cannot load")
        with open(filepath, 'r') as f:
            data = json.load(f)
            for instance in data:
                if instance is None:
                    continue
                if 'kg_path' in instance:
                    kg_path = instance['kg_path']
                    instance['kg'] = KnowledgeGraph().load_kg(kg_path)
                if 'pseudo_kg_path' in instance:
                    pseudo_kg_path = instance['pseudo_kg_path']
                    instance['pseudo_kg'] = KnowledgeGraph().load_kg(pseudo_kg_path)
        return data
    
    def reevaluate(self):
        self.load_results()
        for instance in self.results:
            if instance is None:
                continue
            prompt = instance['prompt']
            response = self.model(prompt)
            instance['response'] = response
            instance['score'] = self.evaluate_response(response, instance['answer'])
        self.save_results()
        
            
                
class TripleRetrievalTask(BaseTask):

    def __init__(self, conversion_config, llm_config, pseudonomizer_config, 
                 base_dataset_file=None, dataset_file=None, results_file=None):
        super().__init__("TripleRetrieval", 
                         conversion_config, 
                         llm_config, 
                         pseudonomizer_config, 
                         base_dataset_file,
                         dataset_file, 
                         results_file)
    
    def evaluate_response(self, response, answer):
        return 1.0 if response == answer else 0.0

    def construct_instance(self, kg: KnowledgeGraph, seed_entities, instance_id=0, max_edges=100):
        # Retrieve triples based on seed entities
        sampled_kg = graph_samplers.sample_ego_graph_from_kg(kg, seed_entities, radius=1)
        sampled_kg = graph_samplers.prune_kg(sampled_kg, max_edges=max_edges, max_degree=20)

        pseudo_kg = self.pseudonymize_kg(sampled_kg)
        
        triple_sample = random.choice([triple for triple in sampled_kg.graph.edges(data='relation_id')])
        triple_sample = Triple(head=sampled_kg.entities[triple_sample[0]], 
                               relation=sampled_kg.relations[triple_sample[2]], 
                               tail=sampled_kg.entities[triple_sample[1]])

        if random.randint(0, 1) == 0:
            triple, corruption_type = triple_sample, "None"
        else:
            triple, corruption_type = self.corrupt_triplet(triple_sample, sampled_kg)
        
        question = f"Is the following triplet fact present in the knowledge graph (Yes/No)? ({triple.head.label}, {triple.relation.label}, {triple.tail.label})"

        answer = "Yes" if corruption_type == "None" else "No"

        return {
            'id': instance_id,
            'question': question,
            'triple': triple,
            'answer': answer,
            'corruption_type': corruption_type,
            'seed_entities': seed_entities,
            'kg': sampled_kg,
            'pseudo_kg': pseudo_kg,
            'pseudonomizer_mapping': self.pseudonomizer.copy_mapping()
        }
    
    def question(self, triple):
        return f"Is the following triplet fact present in the knowledge graph (Yes/No)? ({triple.head.label}, {triple.relation.label}, {triple.tail.label})"

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

    def format_instance(self, instance, text_kg):
        triple: Triple = Triple.from_dict(instance['triple'])
        if self.pseudonomizer: # assumes mapping already in place
            triple.head = self.pseudonomizer.map_entity(triple.head)
            triple.tail = self.pseudonomizer.map_entity(triple.tail)
        instance['prompt'] = self.structure_prompt(self.question(triple), text_kg)
        instance['question'] = self.question(triple)

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
    kg.load_entities('data/countries/entities.tsv')
    kg.load_core_nodes('data/countries/nodes.tsv')

    # Load relations
    kg.load_relations('data/countries/relations.tsv')

    # Load edges and attributes
    kg.load_edges('data/countries/edges.tsv')
    kg.load_attributes('data/countries/attributes.tsv')

    # Print graph information
    # kg.print_graph_info()

    # Visualize the graph
    # kg.visualize_graph()
    # kg.get_ego_graph(3393, radius=1)

    # Create an instance of KnowledgeGraphTextPresenter and extract triplets
    conversion_config = {'type': "list_of_edges"}
    llm_config = {'model': 'gpt-4o-mini', 'provider': 'openai'}
    pseudonomizer_config = {'pseudonym_file': 'data/countries/pseudonym_data/country_pseudonyms.tsv'}


    task = TripleRetrievalTask(conversion_config, llm_config, pseudonomizer_config)

    seed_entities = random.sample(list(kg.core_nodes.keys()), 10)
    
    
    try:
        task.load_base_dataset()
    except ValueError:
        task.construct_base_instances(kg, num_instances=10, num_seed_entities=10, max_edges=100)
        task.save_base_dataset()
    
    try:
        task.load_formatted_dataset()
    except ValueError:
        task.construct_formatted_instances()
        task.save_formatted_dataset()

    # try:
    #     task.run()
    # except ValueError:
    #     print("No LLM configured, skipping run")

