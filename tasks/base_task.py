import random
from kg_builder import KnowledgeGraph, KnowledgeGraphTextPresenter

class BaseTask:

    def __init__(self, task_name):
        self.task_name = task_name

    def run(self, kg, seed_entities, config):
        raise NotImplementedError('You must implement the run method in your task class')
    

class TripleRetrievalTask(BaseTask):

    def __init__(self, task_name):
        super().__init__(task_name)

    def run(self, kg: KnowledgeGraph, seed_entities):
        # Retrieve triples based on seed entities
        breakpoint()
        text_presenter = KnowledgeGraphTextPresenter(kg)
        triplets = [text_presenter.get_triplets(e) for e in seed_entities]
        text_kg = text_presenter.get_triplet_sentences(triplets)
        triple_sample = random.choice(triplets)
        if random.randint(0, 1) == 0:
            triples = triple_sample
        else:
            corrupted_triplet = list(triple_sample)
            corrupt_index = random.choice([0, 1, 2])
            corrupted_triplet[corrupt_index] = random.choice(kg.entities if corrupt_index != 1 else kg.relations)
            triples = tuple(corrupted_triplet)
        return triples
    

# sample_kg: (KnowledgeGraph -> KnowledgeGraph)
# text_presenter: (KnowledgeGraph -> str representing the KG)

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
    seed_entities = [10]
    task = TripleRetrievalTask('triple_retrieval')
    triples = task.run(kg, seed_entities)
    print(triples)
    text_kg = presenter.get_triplet_sentences(triples)
    print(text_kg)