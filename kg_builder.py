import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


# Class to represent each entity (node in the graph)
class Entity:
    def __init__(self, entity_id, wikidata_id, label):
        self.entity_id = entity_id  # Entity ID from entities.tsv or nodes.tsv
        self.wikidata_id = wikidata_id  # WikiData ID
        self.label = label  # The name of the entity

    def __repr__(self):
        return f"Entity(id={self.entity_id}, wikidata_id={self.wikidata_id}, label={self.label})"

# Class to represent relations between entities (edges in the graph)
class Relation:
    def __init__(self, relation_id, wikidata_id, label):
        self.relation_id = relation_id  # Relation ID from relations.tsv
        self.wikidata_id = wikidata_id  # WikiData relation ID
        self.label = label  # The name of the relation

    def __repr__(self):
        return f"Relation(id={self.relation_id}, label={self.label})"


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()  # Directed graph
        self.entities = {}  # Stores Entity objects by entity_id
        self.relations = {}  # Stores Relation objects by relation_id

    # Load entities (nodes) from entities.tsv or nodes.tsv
    def load_entities(self, file_path):
        df = pd.read_csv(file_path, sep='\t')
        for _, row in df.iterrows():
            entity = Entity(row['entityID'], row['wikidataID'], row['label'])
            self.entities[row['entityID']] = entity
            self.graph.add_node(row['entityID'], label=row['label'], wikidata_id=row['wikidataID'])

    # Load relations from relations.tsv
    def load_relations(self, file_path):
        df = pd.read_csv(file_path, sep='\t')
        for _, row in df.iterrows():
            relation = Relation(row['relationID'], row['wikidataID'], row['label'])
            self.relations[row['relationID']] = relation

    # Load edges from edges.tsv and add them to the graph
    def load_edges(self, file_path):
        df = pd.read_csv(file_path, sep='\t')
        for _, row in df.iterrows():
            head = row['headEntity']
            tail = row['tailEntity']
            relation_id = row['relation']
            if relation_id in self.relations:
                relation_label = self.relations[relation_id].label
                self.graph.add_edge(head, tail, relation=relation_label)

    # Load attributes from attributes.tsv and add them as edges
    def load_attributes(self, file_path):
        df = pd.read_csv(file_path, sep='\t')
        for _, row in df.iterrows():
            head = row['headEntity']
            tail = row['tailEntity']
            relation_id = row['relation']
            if relation_id in self.relations:
                relation_label = self.relations[relation_id].label
                self.graph.add_edge(head, tail, relation=relation_label)

    # Print graph information
    def print_graph_info(self):
        print("Graph Nodes (Entities):", self.graph.nodes(data=True))
        print("Graph Edges (Relations):", self.graph.edges(data=True))

    # Visualize the graph (optional)
    def visualize_graph(self):
        nx.draw(self.graph, with_labels=True, node_color='lightblue', font_weight='bold')



if __name__ == "__main__":
    kg = KnowledgeGraph()

    # Load entities and nodes
    kg.load_entities('data/entities.tsv')
    kg.load_entities('data/nodes.tsv')

    # Load relations
    kg.load_relations('data/relations.tsv')

    # Load edges and attributes
    kg.load_edges('data/edges.tsv')
    kg.load_attributes('data/attributes.tsv')

    # Print graph information
    kg.print_graph_info()


