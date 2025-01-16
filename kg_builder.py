import pickle
import random
from typing import Dict
import networkx as nx
import pandas as pd
import os
import matplotlib.pyplot as plt
import json
from collections import deque

from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS

import urllib
import yaml

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
    def viz_ego_graph(self, entity_id, radius=1):
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
        "list_of_edges": "The knowledge graph is presented as a list of directed edges of the form (subject, relation, object).",
        "structured_yaml": "The knowledge graph is presented as a structured YAML format. Each entity is a key, and the value is a dictionary of relations and objects.",
        "structured_json": "The knowledge graph is presented as a structured JSON format. Each entity is a key, and the value is a dictionary of relations and objects.",
        "rdf_turtle": "The knowledge graph is presented as an RDF Turtle format."
    }

    def __init__(self, type="list_of_edges"):
        self.type = type

    def get_description(self):
        return self.FORMAT_DESCRIPTION[self.type]

    def convert(self, kg: KnowledgeGraph):
        """Converts a knowledge graph into a textual representation"""
        match self.type:
            case "list_of_edges":
                text = self.to_list_of_edges(kg)
            case "structured_yaml":
                text = self.to_structured_yaml(kg)
            case "structured_json":
                text = self.to_structured_json(kg)
            case "rdf_turtle1":
                text = self.to_rdf_turtle1(kg)
            case "rdf_turtle2":
                text = self.to_rdf_turtle2(kg)
            case "rdf_turtle3":
                text = self.to_rdf_turtle3(kg)
            case "json_ld1":
                text = self.to_json_ld1(kg)
            case "json_ld2":
                text = self.to_json_ld2(kg)
            case "json_ld3":
                text = self.to_json_ld3(kg)
            case "nt":
                text = self.to_nt(kg)
            case _:
                raise ValueError(f"Unknown text format: {self.type}")
        
        return text

    # Method to generate triplets (head, relation, tail) from the ego graph
    def get_triplets(self, kg):
        """gets the triplets in the kg"""
        triplets = []
        for head, tail, relation in kg.graph.edges(data='relation'):
            triplets.append((kg.entities[head].label, relation, kg.entities[tail].label))
        return triplets

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
    
    def to_structured_yaml(self, kg: KnowledgeGraph):
        """Converts the knowledge graph to a structured YAML format."""
        structured_data = {}

        for head, relation, tail in self.get_triplets(kg):
            if head not in structured_data:
                structured_data[head] = {}
            if relation not in structured_data[head]:
                structured_data[head][relation] = []
            structured_data[head][relation].append(tail)

        return yaml.dump(structured_data, default_flow_style=False)
    
    def to_structured_json(self, kg: KnowledgeGraph):
        """Converts the knowledge graph to a structured JSON format."""
        structured_data = {}

        for head, relation, tail in self.get_triplets(kg):
            if head not in structured_data:
                structured_data[head] = {}
            if relation not in structured_data[head]:
                structured_data[head][relation] = []
            structured_data[head][relation].append(tail)

        return json.dumps(structured_data, indent=2)
    
    def make_rdflib_graph(self, kg: KnowledgeGraph, use_entity_ids=False, use_relation_ids=False):
        g = Graph()
        ex = Namespace("http://example.org/countries#")
        g.bind("ex", ex)
        g.bind("rdf", RDF)
        g.bind("rdfs", RDFS)

        if use_entity_ids:
            # Add entity definitions with rdfs:label
            for entity_id, entity in kg.entities.items():
                entity_uri = URIRef(f"http://example.org/countries#{urllib.parse.quote(str(entity_id))}")
                g.add((entity_uri, RDF.type, ex.Country))
                g.add((entity_uri, RDFS.label, Literal(entity.label)))

        if use_relation_ids:
            # Add relation definitions with rdfs:label
            for relation_id, relation in kg.relations.items():
                relation_uri = URIRef(f"http://example.org/countries#{urllib.parse.quote('R' + str(relation_id))}")
                g.add((relation_uri, RDF.type, RDF.Property))
                g.add((relation_uri, RDFS.label, Literal(relation.label)))

        # Add edges using entity IDs and relation IDs
        for head, tail, edge_data in kg.graph.edges(data=True):
            head = str(head) if use_entity_ids else str(kg.entities[head].label)
            head_uri = URIRef(f"http://example.org/countries#{urllib.parse.quote(head)}")
            
            relation = 'R' + str(edge_data['relation_id']) if use_relation_ids else str(edge_data['relation'])
            relation_uri = URIRef(f"http://example.org/countries#{urllib.parse.quote(relation)}")
            
            tail = str(tail) if use_entity_ids else str(kg.entities[tail].label)
            tail_uri = URIRef(f"http://example.org/countries#{urllib.parse.quote(tail)}")

            g.add((head_uri, relation_uri, tail_uri))

        return g


    def to_rdf_turtle1(self, kg: KnowledgeGraph):
        """Converts the knowledge graph to RDF Turtle format with URI encoded labels."""
        g = self.make_rdflib_graph(kg, use_entity_ids=False, use_relation_ids=False)

        return g.serialize(format="turtle")
    
    def to_rdf_turtle2(self, kg: KnowledgeGraph):
        """Converts the knowledge graph to RDF Turtle format using node IDs and URI encoded relations."""
        g = self.make_rdflib_graph(kg, use_entity_ids=True, use_relation_ids=False)

        return g.serialize(format="turtle")
    
    def to_rdf_turtle3(self, kg: KnowledgeGraph):
        """Converts the knowledge graph to RDF Turtle format using node IDs and relation IDs."""
        g = self.make_rdflib_graph(kg, use_entity_ids=True, use_relation_ids=True)

        return g.serialize(format="turtle")
    
    def to_json_ld1(self, kg: KnowledgeGraph):
        """Converts the knowledge graph to JSON-LD format with URI encoded labels."""
        g = self.make_rdflib_graph(kg, use_entity_ids=False, use_relation_ids=False)

        context = {
            "@context": {
                "ex": "http://example.org/countries#",
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                "label": "rdfs:label",
                "type": "@type"
            }
        }

        return g.serialize(format="json-ld", indent=2, context=context)
    
    def to_json_ld2(self, kg: KnowledgeGraph):
        """Converts the knowledge graph to JSON-LD format using node IDs and URI encoded relations."""
        g = self.make_rdflib_graph(kg, use_entity_ids=True, use_relation_ids=False)

        context = {
            "@context": {
                "ex": "http://example.org/countries#",
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                "label": "rdfs:label",
                "type": "@type"
            }
        }

        return g.serialize(format="json-ld", indent=2, context=context)
    
    def to_json_ld3(self, kg: KnowledgeGraph):
        """Converts the knowledge graph to JSON-LD format using node IDs and relation IDs."""
        g = self.make_rdflib_graph(kg, use_entity_ids=True, use_relation_ids=True)

        context = {
            "@context": {
                "ex": "http://example.org/countries#",
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                "label": "rdfs:label",
                "type": "@type"
            }
        }

        return g.serialize(format="json-ld", indent=2, context=context)
    
    def to_nt(self, kg: KnowledgeGraph):
        """Converts the knowledge graph to N-Triples format."""
        g = self.make_rdflib_graph(kg, use_entity_ids=False, use_relation_ids=False)

        return g.serialize(format="nt")



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

    # Create an instance of KnowledgeGraphTextPresenter and extract triplets
    presenter = KnowledgeGraphTextPresenter("rdf_turtle3")

    # Get textual summary of the knowledge graph subset
    text_kg = presenter.convert(kg)
    print("\nTextual Summary of Knowledge Graph Subset:")
    print(text_kg)
    breakpoint()
