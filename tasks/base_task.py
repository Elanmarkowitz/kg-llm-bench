import random
from kg_builder import KnowledgeGraph, KnowledgeGraphTextPresenter
from samplers import graph_samplers

class BaseTask:
    """Tasks are the main function that runs things. They handle sampling the kg, pseudonimizing, creating task question, 
    creating the question, making the llm request, and evaluating the response."""

    def __init__(self, task_name):
        self.task_name = task_name

    def run(self, kg, seed_entities, config):
        raise NotImplementedError('You must implement the run method in your task class')
    

class TripleRetrievalTask(BaseTask):

    def __init__(self, task_name, conversion_config, llm_config, pseudonomizer_config):
        super().__init__(task_name)
        self.text_presenter = KnowledgeGraphTextPresenter(conversion_config)
        self.llm = LLM(llm_config)
        self.pseudonomizer = Pseudonomizer(pseudonomizer_config)



    def run(self, kg: KnowledgeGraph, seed_entities, max_edges=100):
        # Retrieve triples based on seed entities
        breakpoint()
        sampled_kg = graph_samplers.sample_ego_graph_from_kg(kg, seed_entities, radius=1)
        sampled_kg = graph_samplers.prune_kg(sampled_kg, max_edges, max_degree=20)

        text_kg = self.text_presenter.to_list_of_sentences(sampled_kg)

        triple_sample = random.choice(sampled_kg.graph.edges(data=True))
        if random.randint(0, 1) == 0:
            triple = triple_sample
        else:
            corrupted_triplet = list(triple_sample)
            corrupt_index = random.choice([0, 1, 2])
            corrupted_triplet[corrupt_index] = random.choice(kg.entities if corrupt_index != 1 else kg.relations)
            triple = tuple(corrupted_triplet)
        
        question = f"Is the following triplet fact present in the knowledge graph? ({triple[0]}, triple[])"
    

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