from tasks.base_task import BaseTask
from copy import deepcopy
import os
import random
from kg_builder import KnowledgeGraph
from samplers import graph_samplers


class ShortestPathTask(BaseTask):

    def __init__(self, conversion_config, llm_config, pseudonomizer_config, dataset_file=None, results_file=None):
        super().__init__("ShortestPath",
                         conversion_config,
                         llm_config,
                         pseudonomizer_config,
                         dataset_file,
                         results_file)

    def run(self):
        self.results = deepcopy(self.data)
        for instance in self.results:
            if instance is None: # To Do: check for the edge cases.
                continue
            prompt = instance['prompt']
            response = self.model(prompt)
            instance['response'] = response
            instance['score'] = self.evaluate_response(response, instance['answer'])
        self.save_results()

    def evaluate_response(self, response, answer):
        #TODO: fix eval
        for answer_option in answer:
            response_path = response.replace("SHORTEST PATH:", "").strip()
            response_path = response_path.strip('[]').split(',')
            response_path = [node.strip().strip("'").strip('"') for node in response_path]

            answer_path = answer_option.replace("SHORTEST PATH:", "").strip()
            answer_path = answer_path.strip('[]').split(',')
            answer_path = [node.strip().strip("'").strip('"') for node in answer_path]
            if response_path == answer_path:
                return 1.0
        return 0.0

    def construct_instances(self, kg: KnowledgeGraph, num_instances=10, num_seed_entities=2, max_edges=100):
        """Constructs instances for the task."""
        for instance in range(num_instances):
            seed_entities = random.sample(list(kg.core_nodes.keys()), num_seed_entities)
            self.data.append(self.construct_instance(kg, seed_entities, instance, max_edges))

    def construct_instance(self, kg: KnowledgeGraph, seed_entities, instance_id=0, max_edges=100):
        ent1, ent2 = seed_entities[:2]
        shortest_paths = kg.get_shortest_paths(ent1, ent2)

        if not shortest_paths:
            return None

        seed_entities = list(set(seed_entities + [e for path in shortest_paths for e in path]))

        sampled_kg = graph_samplers.sample_ego_graph_from_kg(kg, seed_entities, radius=1)
        sampled_kg = graph_samplers.prune_kg(sampled_kg, max_edges=max_edges, max_degree=20)

        def add_path_to_graph(path):
            edge_data = []
            for e1, e2 in zip(path[:-1], path[1:]):
                relation = kg.graph.get_edge_data(e1, e2)
                edge_data.append((e1, relation, e2))
            # Add Edge data to the sampled_kg
            for e1, relation, e2 in edge_data:
                if not sampled_kg.has_edge(e1, e2):
                    sampled_kg.add_edge(e1, e2, relation=relation["relation"], relation_id=relation["relation_id"])

        def path_is_present(path):
            edge_data = []
            for e1, e2 in zip(path[:-1], path[1:]):
                relation = kg.graph.get_edge_data(e1, e2)
                edge_data.append((e1, relation, e2))
            # Add Edge data to the sampled_kg
            for e1, relation, e2 in edge_data:
                if not sampled_kg.has_edge(e1, e2):
                    return False
            return True

        add_path_to_graph(shortest_paths[0])
        shortest_paths = [shortest_paths[0]] + [path for path in shortest_paths[1:] if path_is_present(path)]
            
        text_kg = self.text_presenter.to_list_of_edges(sampled_kg)

        answer_paths = [[kg.entities[node].label for node in path] for path in shortest_paths]

        question = f"Your task is to find the shortest path from {answer_paths[0][0]} to {answer_paths[0][-1]}. For example, if the shortest path between Argentina and Mexico is through Bolivia and Colombia, then answer should be SHORTEST PATH: ['Argentina', 'Bolivia', 'Colombia', 'Mexico']. \n you should list your answer in the form list. \n\n What is the shortest path from {answer_paths[0][0]} to {answer_paths[0][-1]}? \n Answer: SHORTEST PATH:"

        prompt = self.structure_prompt(question, text_kg)
        answer = [f"SHORTEST PATH: {str(path)}" for path in answer_paths]

        return {
            'id': instance_id,
            'prompt': prompt,
            'question': question,
            'text_kg': text_kg,
            'shortest_paths': answer_paths,
            'answer': answer,
            'seed_entities': seed_entities,
            'kg': kg
        }

    def reformat_instances(self):
        """Reformats self.data using self.text_presenter."""
        for instance in self.data:
            kg = instance['kg']
            text_kg = self.text_presenter.to_list_of_edges(kg)
            instance['text_kg'] = text_kg

    def structure_prompt(self, question, text_kg):
        intro = f"Your job is to answer questions using the following knowledge graph. {self.text_presenter.get_description()}. You must rely exclusively on the information presented in the Knowledge Graph to answer questions."
        prompt = f"{intro}\n\nKnowledge Graph:\n{text_kg}\n\n{question}"
        return prompt

if __name__ == '__main__':
    kg = KnowledgeGraph()

    # Load entities and nodes
    kg.load_entities('data/countries/entities.tsv')
    kg.load_core_nodes('data/countries/nodes.tsv')

    # Load relations
    kg.load_relations('data/countries/relations.tsv')

    # Load edges and attributes
    kg.load_edges('data/countries/edges.tsv')
    kg.load_attributes('data/countries/attributes.tsv')


    conversion_config = {'type': "list_of_edges"}
    llm_config = {'model': 'gpt-4o-mini', 'provider': 'openai'}
    pseudonomizer_config = {'pseudonym_file': 'data/countries/pseudonym_data/country_pseudonyms.tsv'}


    task = ShortestPathTask(conversion_config, llm_config, pseudonomizer_config)
    seed_entities = random.sample(list(kg.core_nodes.keys()), 2)

    task.load_dataset()

    task.construct_instances(kg, num_instances=10, num_seed_entities=10, max_edges=100)

    task.save_base_data()
    task.save_dataset()

    task.run()
    print("Finished..")
