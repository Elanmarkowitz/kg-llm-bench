# pseudonymizer.py
from copy import deepcopy
import pandas as pd
import random

from kg_builder import KnowledgeGraph

class Pseudonymizer:
    def __init__(self, pseudonym_file, seed=1234):
        self.seed = 1234
        random.seed(self.seed)
        self.pseudonym_file = pseudonym_file
        self.pseudonyms = self.load_pseudonyms(pseudonym_file)
        self.mapping = {}

    def load_pseudonyms(self, pseudonym_file):
        pseudonyms = pd.read_csv(pseudonym_file, sep='\t')
        pseudonyms = pseudonyms.names.to_list()
        random.shuffle(pseudonyms)
        return pseudonyms
    
    def clear_mapping(self):
        self.mapping.clear()

    def create_mapping(self, knowledge_graph: KnowledgeGraph, append=False):
        if self.mapping and not append:
            raise ValueError('Mapping already exists. Please clear the mapping before creating a new one or set the append kwarg.')
        if len(self.pseudonyms) < len(knowledge_graph.core_nodes):
            raise ValueError(f'Number of pseudonyms ({len(self.pseudonyms)}) must be greater than or equal to the number of nodes ({len(knowledge_graph.core_nodes)})')
        core_nodes = list(knowledge_graph.core_nodes.values())
        pseudos = random.sample(self.pseudonyms, len(core_nodes))
        for entity, pseudo in zip(core_nodes, pseudos):
            if entity.entity_id not in self.mapping:
                self.mapping[entity.entity_id] = pseudo
            else:
                breakpoint()
                raise ValueError(f'Duplicate entity label found: {entity.entity_id}')

    def pseudonymize(self, knowledge_graph: KnowledgeGraph):
        pseudo_kg = deepcopy(knowledge_graph)
        for entity in knowledge_graph.entities.values():
            if entity.entity_id in self.mapping:
                entity.entity_id = self.mapping[entity.entity_id]
        for entity in knowledge_graph.core_nodes.values():
            if entity.entity_id in self.mapping:
                entity.entity_id = self.mapping[entity.entity_id]
        return pseudo_kg

# Example usage:
# pseudonymizer = Pseudonymizer('country_pseudonyms.tsv')
# knowledge_graph = ...  # Load or create your knowledge graph
# pseudonymized_graph = pseudonymizer.pseudonymize(knowledge_graph)


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

    from samplers import graph_samplers
    import random
    sampled_nodes = random.sample(list(kg.core_nodes.keys()), 10)
    sampled_kg = graph_samplers.sample_ego_graph_from_kg(kg, sampled_nodes, radius=1)

    pseudonymizer = Pseudonymizer('data/countries/pseudonym_data/country_pseudonyms.tsv')
    pseudonymizer.create_mapping(sampled_kg)
    pseudonymized_graph = pseudonymizer.pseudonymize(sampled_kg)
    print(pseudonymized_graph)