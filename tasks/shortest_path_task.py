from typing import List
from tqdm import tqdm
from tasks.base_task import BaseTask
from copy import deepcopy
import os
import random
from kg_builder import Entity, KnowledgeGraph
from samplers import graph_samplers
import ast

class ShortestPathTask(BaseTask):

    def __init__(self, conversion_config, llm_config, pseudonomizer_config, 
                 base_dataset_file=None, dataset_file=None, results_file=None):
        super().__init__("ShortestPath",
                         conversion_config,
                         llm_config,
                         pseudonomizer_config,
                         base_dataset_file,
                         dataset_file,
                         results_file)

    def evaluate_response(self, response, answer):
        response = response.replace("SHORTEST PATH:", "").strip()
        try:
            response_list = ast.literal_eval(response)
        except (SyntaxError, ValueError):
            return 0.0

        for answer_option in answer:
            try:
                answer_list = ast.literal_eval(answer_option.replace("SHORTEST PATH:", "").strip())
                if response_list == answer_list:
                    return 1.0
            except (SyntaxError, ValueError):
                continue
        return 0.0

    def construct_instance(self, kg: KnowledgeGraph, seed_entities, instance_id=0, max_edges=100):
        ent1, ent2 = seed_entities[:2]
        shortest_paths = kg.get_shortest_paths(ent1, ent2, depth=7)

        if not shortest_paths:
            raise ValueError("No shortest path found between seed entities")

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

        answer_paths: List[List[Entity]] = [[kg.entities[node] for node in path] for path in shortest_paths]
        answer = self.answer(answer_paths)
        question = self.question(answer_paths)
        pseudo_kg = self.pseudonymize_kg(sampled_kg)

        return {
            'id': instance_id,
            'question': question,
            'shortest_paths': answer_paths,
            'answer': answer,
            'seed_entities': seed_entities,
            'kg': sampled_kg,
            'pseudo_kg': pseudo_kg,
            'pseudonomizer_mapping': self.pseudonomizer.copy_mapping()
        }

    def construct_formatted_instances(self):
        """formats the kg and saves the formatted dataset (i.e. with actual prompt)"""
        self.formatted_data = deepcopy(self.base_data)
        for instance in self.formatted_data:
            assert 'kg_path' in instance
            kg = instance.pop('kg')

            # task specific read instructions
            instance['shortest_paths'] = [[Entity.from_dict(e) for e in path] for path in instance['shortest_paths']]

            if self.pseudonomizer:
                if 'pseudo_kg' in instance:
                    kg = instance.pop('pseudo_kg')
                else:
                    if not 'pseudonomizer_mapping' in instance:
                        raise ValueError("Pseudonomizer config set but no pseudonomizer mapping in the base data")
                    self.pseudonomizer.load_mapping(instance['pseudonomizer_mapping'])
                    kg = self.pseudonomizer.pseudonymize(kg)
                    # task specific pseudonomize conversions
                    instance['shortest_paths'] = [[self.pseudonomizer.map_entity(e) for e in path] for path in instance['shortest_paths']]
            
            # construct answer, question, text_kg, and prompt
            instance['answer'] = self.answer(instance['shortest_paths'])
            question = self.question(instance['shortest_paths'])
            text_kg = self.text_presenter.convert(kg)

            instance['text_kg'] = text_kg
            instance['prompt'] = self.structure_prompt(question, text_kg)
            instance['question'] = question
            # answer is a count so no change needed

    def question(self, answer_paths: List[List[Entity]]):
        return f"Your task is to find the shortest path from {answer_paths[0][0].label} to {answer_paths[0][-1].label}. You can use both incoming and outgoing edges. For example, if the shortest path between Argentina and Mexico is through Bolivia and Colombia, then answer should be SHORTEST PATH: ['Argentina', 'Bolivia', 'Colombia', 'Mexico']. \n you should list your answer in the form list. \n\n What is the shortest path from {answer_paths[0][0].label} to {answer_paths[0][-1].label}? \n Answer: SHORTEST PATH:"

    def answer(self, answer_paths: List[List[Entity]]):
        return [f"SHORTEST PATH: {str(self.ent_path_to_label_path(path))}" for path in answer_paths]

    def ent_path_to_label_path(self, path: List[Entity]):
        return [e.label for e in path]

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

    # try:
    #     task.load_base_dataset()
    # except ValueError:
    task.construct_base_instances(kg, num_instances=10, num_seed_entities=10, max_edges=100)
    task.save_base_dataset()
    
    try:
        task.load_formatted_dataset()
    except ValueError:
        task.construct_formatted_instances()
        task.save_formatted_dataset()

    try:
        task.run()
    except ValueError:
        print("No LLM configured, skipping run")

    # task.run()
    print("Finished..")
