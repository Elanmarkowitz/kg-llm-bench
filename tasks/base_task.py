from copy import deepcopy
import json
import os
import random
from kg_builder import KnowledgeGraph, KnowledgeGraphTextPresenter, Triple
from llm.llm import LLM
from pseudonymizer import Pseudonymizer
from samplers import graph_samplers

class BaseTask:
    """Tasks are the main function that runs things. They handle sampling the kg, pseudonimizing, creating task question, 
    creating the question, making the llm request, and evaluating the response."""

    def __init__(self, task_name):
        self.task_name = task_name

    def run(self, kg, seed_entities, config):
        raise NotImplementedError('You must implement the run method in your task class')
    

class TripleRetrievalTask(BaseTask):

    def __init__(self, conversion_config, llm_config, pseudonomizer_config, dataset_file=None, results_file=None):
        super().__init__("Triple Retrieval")
        self.text_presenter = KnowledgeGraphTextPresenter(**conversion_config)
        self.model = LLM(**llm_config)
        self.pseudonomizer = Pseudonymizer(**pseudonomizer_config)
        self.dataset_file = dataset_file
        self.results_file = results_file
        self.data = []
        self.results = []

    def run(self):
        self.results = deepcopy(self.data)
        breakpoint()
        for instance in self.results:
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
            self.data.append(self.construct_instance(kg, seed_entities, max_edges))

    def construct_instance(self, kg: KnowledgeGraph, seed_entities, max_edges=100):
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
            'prompt': prompt,
            'question': question,
            'text_kg': text_kg,
            'triple': triple,
            'answer': answer,
            'corruption_type': corruption_type,
            'seed_entities': seed_entities,
            # 'kg': kg # TODO: Find way to save kg
        }

    def save_results(self):
        if self.results_file:
            with open(self.results_file, 'w') as f:
                json.dump(self.results, f)

    def save_data(self):
        if self.dataset_file:
            with open(self.dataset_file, 'w') as f:
                json.dump(self.data, f)
    
    def load_data(self):
        if self.dataset_file and os.path.exists(self.dataset_file):
            with open(self.dataset_file, 'r') as f:
                self.data = json.load(f)

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

    triples = task.construct_instances(kg, num_instances=10, num_seed_entities=10, max_edges=100)
    breakpoint()

    task.run()