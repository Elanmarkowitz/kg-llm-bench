import pickle
import random
from typing import Dict
import networkx as nx
import pandas as pd
import os
import matplotlib.pyplot as plt
import json
from collections import deque

# Class to represent each entity (node in the graph)
class Entity:
    def __init__(self, entity_id, wikidata_id, label):
        self.entity_id = entity_id  # Entity ID from entities.tsv or nodes.tsv
        self.wikidata_id = wikidata_id  # WikiData ID
        self.label = label  # The name of the entity

    def __repr__(self):
        return f"Entity(entity_id={self.entity_id}, wikidata_id={self.wikidata_id}, label={self.label})"

    def to_dict(self):
        return {
            'entity_id': self.entity_id,
            'wikidata_id': self.wikidata_id,
            'label': self.label
        }

    @classmethod
    def from_dict(cls, data):
        return cls(data['entity_id'], data['wikidata_id'], data['label'])

# Class to represent relations between entities (edges in the graph)
class Relation:
    def __init__(self, relation_id, wikidata_id, label):
        self.relation_id = relation_id  # Relation ID from relations.tsv
        self.wikidata_id = wikidata_id  # WikiData relation ID
        self.label = label  # The name of the relation

    def __repr__(self):
        return f"Relation(id={self.relation_id}, label={self.label})"

    def to_dict(self):
        return {
            'relation_id': self.relation_id,
            'wikidata_id': self.wikidata_id,
            'label': self.label
        }

    @classmethod
    def from_dict(cls, data):
        return cls(data['relation_id'], data['wikidata_id'], data['label'])

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        return super().default(obj)

class Triple:
    def __init__(self, head: Entity, relation: Relation, tail: Entity):
        self.head = head  # Relation ID from relations.tsv
        self.relation = relation  # WikiData relation ID
        self.tail = tail  # The name of the relation

    def __repr__(self):
        return f"Triple(head={self.head}, relation={self.relation}, tail={self.tail})"

    def to_dict(self):
        return {
            'head': self.head.to_dict(),
            'relation': self.relation.to_dict(),
            'tail': self.tail.to_dict()
        }

    @classmethod
    def from_dict(cls, data):
        head = Entity.from_dict(data['head'])
        relation = Relation.from_dict(data['relation'])
        tail = Entity.from_dict(data['tail'])
        return cls(head, relation, tail)

class KnowledgeGraph:
    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()  # Directed graph # TODO: Switch to MultiDiGraph
        self.entities: Dict[int, Entity] = {}  # Stores Entity objects by entity_id
        self.relations: Dict[int, Relation] = {}  # Stores Relation objects by relation_id
        self.core_nodes: Dict[int, Entity] = {} # Stores core nodes in the graph

    # Load entities (nodes) from entities.tsv or nodes.tsv
    def load_entities(self, file_path):
        df = pd.read_csv(file_path, sep='\t')
        for _, row in df.iterrows():
            entity = Entity(row['entityID'], row['wikidataID'], row['label'])
            self.entities[row['entityID']] = entity
            self.graph.add_node(row['entityID'], label=row['label'], wikidata_id=row['wikidataID']) # TODO: Fix to enable pseudonomization on this, or remove labels

    # Load relations from relations.tsv
    def load_relations(self, file_path):
        df = pd.read_csv(file_path, sep='\t')
        for _, row in df.iterrows():
            relation = Relation(row['relationID'], row['wikidataID'], row['label'])
            self.relations[row['relationID']] = relation

    # load core nodes from nodes.tsv
    def load_core_nodes(self, file_path):
        df = pd.read_csv(file_path, sep='\t')
        for _, row in df.iterrows():
            entity = Entity(row['entityID'], row['wikidataID'], row['label'])
            self.core_nodes[row['entityID']] = entity

    # Load edges from edges.tsv and add them to the graph
    def load_edges(self, file_path):
        df = pd.read_csv(file_path, sep='\t')
        for _, row in df.iterrows():
            head = row['headEntity']
            tail = row['tailEntity']
            relation_id = row['relation']
            if relation_id in self.relations:
                relation_label = self.relations[relation_id].label
                self.graph.add_edge(head, tail, relation=relation_label, relation_id=relation_id)
            else:
                raise ValueError(f"Relation ID {relation_id} not found in the relations.")

    # Load attributes from attributes.tsv and add them as edges
    def load_attributes(self, file_path):
        df = pd.read_csv(file_path, sep='\t')
        for _, row in df.iterrows():
            head = row['headEntity']
            tail = row['tailEntity']
            relation_id = row['relation']
            if relation_id in self.relations:
                relation_label = self.relations[relation_id].label
                self.graph.add_edge(head, tail, relation=relation_label, relation_id=relation_id)

    def save_kg(self, file_path):
        data = {
            'entities': self.entities,
            'relations': self.relations,
            'core_nodes': self.core_nodes,
            'graph': self.graph
        }
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
        return str(file_path)

    def load_kg(self, file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            self.entities = data['entities']
            self.relations = data['relations']
            self.core_nodes = data['core_nodes']
            self.graph = data['graph']
        return self

    # Print graph information
    def print_graph_info(self):
        print("Graph Nodes (Entities):", self.graph.nodes(data=True))
        print("Graph Edges (Relations):", self.graph.edges(data=True))

    # Visualize the graph (optional)
    def visualize_graph(self):
        nx.draw(self.graph, with_labels=True, node_color='lightblue', font_weight='bold')

    # Method to get and visualize ego graph with radius 1
    def get_ego_graph(self, entity_id, radius=1):
        if entity_id in self.graph.nodes:
            ego_g = nx.ego_graph(self.graph, entity_id, radius=radius)
            print(f"Ego Graph centered on entity {entity_id} with radius {radius}:")
            print("Nodes in the subgraph:", ego_g.nodes(data=True))
            print("Edges in the subgraph:", ego_g.edges(data=True))

            # Prepare node labels using human-readable labels
            node_labels = {node: self.entities[node].label for node in ego_g.nodes}
            # Visualize the ego graph
            plt.figure(figsize=(8, 6))
            pos = nx.spring_layout(ego_g)  # Layout for visualization
            nx.draw(ego_g, pos, with_labels=True, node_size=300, font_size=10, labels=node_labels, node_color='lightgreen', font_weight='bold', edge_color="gray")
            # nx.draw(G, pos, with_labels=True, node_size=300, node_color="skyblue", font_size=10, font_color="black", font_weight="bold", edge_color="gray")
            edge_labels = nx.get_edge_attributes(ego_g, 'relation')
            nx.draw_networkx_edge_labels(ego_g, pos, edge_labels=edge_labels, font_color='red')
            plt.show()

        else:
            print(f"Entity ID {entity_id} not found in the graph.")

    def add_edge(self, e1, e2, relation, relation_id):
        self.graph.add_edge(e1, e2, relation=relation, relation_id=relation_id)

    def has_edge(self, e1, e2):
        return self.graph.has_edge(e1, e2)
    
        
    def get_neighbors(self, ent, fwd=True, bkw=True):
            assert fwd or bkw
            results = set()
            if bkw: 
                predecessors = set(self.graph.predecessors(ent))
                results = results.union(predecessors)
    
            if fwd:
                successors = set(self.graph.successors(ent))
                results = results.union(successors)
        
            return results

    def get_shortest_paths(self, ent1, ent2, depth=None):
        """Finds all shortest paths between two entities using BFS."""
        if ent1 not in self.graph or ent2 not in self.graph:
            return []

        queue = deque([(ent1, [ent1], 0)])
        visited = set()
        shortest_paths = []
        shortest_length = None
        
        while queue:
            current_node, path, current_depth = queue.popleft()
            if current_node in visited:
                continue

            visited.add(current_node)

            if current_node == ent2:
                if shortest_length is None:
                    shortest_length = len(path)
                if len(path) == shortest_length:
                    shortest_paths.append(path)
                elif len(path) < shortest_length:
                    shortest_paths = [path]
                    shortest_length = len(path)
                continue

            if depth is not None and current_depth >= depth:
                continue
            
            for neighbor in self.get_neighbors(current_node, fwd=True, bkw=True):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor], current_depth + 1))

        return shortest_paths
    
    def __repr__(self):
        num_edges = len(self.graph.edges)
        num_entities = len(self.entities)
        num_relations = len(self.relations)
        return f"KnowledgeGraph(num_edges={num_edges}, num_entities={num_entities}, num_relations={num_relations})"
        


class KnowledgeGraphTextPresenter:
    """
    TextPresenter handles turning a kg into in-context text version
    """
    FORMAT_DESCRIPTION = {
        "list_of_edges": "The knowledge graph is presented as a list of directed edges of the form (subject, relation, object)."
    }

    def __init__(self, type="list_of_edges"):
        self.type = type

    def get_description(self):
        return self.FORMAT_DESCRIPTION[self.type]

    def convert(self, kg: KnowledgeGraph):
        """Converts a knowledge graph into a textual representation"""
        if self.type == "list_of_edges":
            text = self.to_list_of_edges(kg)
        return text

    # Method to generate triplets (head, relation, tail) from the ego graph
    def get_triplets(self, kg):
        """gets the triplets in the kg"""
        triplets = []
        for head, tail, relation in kg.graph.edges(data='relation'):
            triplets.append((kg.entities[head].label, relation, kg.entities[tail].label))
        return triplets
        
        # if entity_id in kg.graph.nodes:
        #     ego_g = nx.ego_graph(self.kg.graph, entity_id, radius=radius)
        #     triplets = []
        # else:
        #     print(f"Entity ID {entity_id} not found in the graph.")
        #     return []

    # Method to convert triplets into human-readable sentences
    def get_triplet_sentences(self, triplets):
        sentences = []
        for head, relation, tail in triplets:
            sentence = f"({head}, {relation}, {tail})"
            sentences.append(sentence)
        return sentences

    # Method to generate a textual summary from a subset of the knowledge graph
    def get_summary(self, entity_id, radius=1):
        sentences = self.get_triplet_sentences(entity_id, radius)
        summary = " ".join(sentences)
        return summary

    def to_list_of_edges(self, kg: KnowledgeGraph):
        """Takes a knowledge graph (or knowledge graph subgraph)"""
        # get kg triplets
        triplets = self.get_triplets(kg)

        text = "Edges: [\n" + ",\n".join(self.get_triplet_sentences(triplets)) + "\n]\n"
        
        return text


if __name__ == "__main__":
    kg = KnowledgeGraph()

    # Load entities and nodes
    kg.load_entities('data/countries/entities.tsv')
    kg.load_entities('data/countries/nodes.tsv')

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
    presenter = KnowledgeGraphTextPresenter(kg)

    # Get triplets
    triplets = presenter.get_triplets(entity_id=3393, radius=1)
    print("Triplets (head, relation, tail):")
    for triplet in triplets:
        print(triplet)

    # Get sentences from triplets
    sentences = presenter.get_triplet_sentences(entity_id=3393, radius=1)
    print("\nTriplet Sentences:")
    for sentence in sentences:
        print(sentence)

    # Get textual summary of the knowledge graph subset
    summary = presenter.get_summary(entity_id=3393, radius=1)
    print("\nTextual Summary of Knowledge Graph Subset:")
    print(summary)
