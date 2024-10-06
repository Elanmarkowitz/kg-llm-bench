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


    # Method to get and visualize ego graph with radius 1
    def get_ego_graph(self, entity_id, radius=1):
        if entity_id in self.graph.nodes:
            ego_g = nx.ego_graph(self.graph, entity_id, radius=radius)
            print(f"Ego Graph centered on entity {entity_id} with radius {radius}:")
            print("Nodes in the subgraph:", ego_g.nodes(data=True))
            print("Edges in the subgraph:", ego_g.edges(data=True))

            # Prepare node labels using human-readable labels
            node_labels = {node: self.graph.nodes[node]['label'] for node in ego_g.nodes}
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


class KnowledgeGraphTextPresenter:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    # Method to generate triplets (head, relation, tail) from the ego graph
    def get_triplets(self, entity_id, radius=1):
        if entity_id in self.kg.graph.nodes:
            ego_g = nx.ego_graph(self.kg.graph, entity_id, radius=radius)
            triplets = []
            for head, tail, data in ego_g.edges(data=True):
                relation = data.get('relation', 'unknown')
                triplets.append((self.kg.graph.nodes[head]['label'], relation, self.kg.graph.nodes[tail]['label']))
            return triplets
        else:
            print(f"Entity ID {entity_id} not found in the graph.")
            return []

    # Method to convert triplets into human-readable sentences
    def get_triplet_sentences(self, entity_id, radius=1):
        triplets = self.get_triplets(entity_id, radius)
        sentences = []
        for head, relation, tail in triplets:
            sentence = f"{head} {relation} {tail}."
            sentences.append(sentence)
        return sentences

    # Method to generate a textual summary from a subset of the knowledge graph
    def get_summary(self, entity_id, radius=1):
        sentences = self.get_triplet_sentences(entity_id, radius)
        summary = " ".join(sentences)
        return summary


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
