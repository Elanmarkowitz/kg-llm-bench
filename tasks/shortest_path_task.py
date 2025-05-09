from typing import List
from tqdm import tqdm
from tasks.base_task import BaseTask, InstanceConstructionError
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
        self.use_flexible_eval = False

    def evaluate_response(self, response, answer):
        response = response.replace("SHORTEST PATH:", "").split('\n')[0].strip()
        if self.use_flexible_eval:
            import re
            match = re.search(r"SHORTEST PATH: \[(.*)\]", response)
            response = match.group(1).strip() if match else ""
        try:
            response_list = ast.literal_eval(response)
        except (SyntaxError, ValueError):
            response_list = response.strip('[]').split(',')
            response_list = [item.strip().strip("'") for item in response_list]
            return 0.0

        for answer_option in answer:
            try:
                answer_list = ast.literal_eval(answer_option.replace("SHORTEST PATH:", "").strip())
                if tuple(response_list) == tuple(answer_list):
                    return 1.0
            except (SyntaxError, ValueError):
                continue
        return 0.0

    def construct_instance(self, kg: KnowledgeGraph, seed_entities, instance_id=0, max_edges=100):
        ent1, ent2 = seed_entities[:2]
        
        shortest_paths = kg.get_shortest_paths(ent1, ent2, depth=7)
        
        if not shortest_paths:
            raise InstanceConstructionError("No shortest path found between seed entities")

        seed_entities = list(set(seed_entities + [e for path in shortest_paths for e in path]))

        sampled_kg = graph_samplers.sample_ego_graph_from_kg(kg, seed_entities, radius=1)
        sampled_kg = graph_samplers.prune_kg(sampled_kg, max_edges=max_edges, max_degree=20)

        def add_path_to_graph(path):
            edge_data = []
            for e1, e2 in zip(path[:-1], path[1:]):
                relation = kg.graph.get_edge_data(e1, e2)
                if relation:
                    edge_data.append((e1, relation, e2))
                else:
                    relation = kg.graph.get_edge_data(e2, e1)
                    edge_data.append((e2, relation, e1))
            # Add Edge data to the sampled_kg
            for e1, relation, e2 in edge_data:
                if not sampled_kg.has_edge(e1, e2):
                    sampled_kg.add_edge(e1, e2, relation=relation["relation"], relation_id=relation["relation_id"])
                    
                    # add missing entities and core nodes if removed during pruning
                    if e1 in kg.entities and e1 not in sampled_kg.entities:
                        sampled_kg.entities[e1] = deepcopy(kg.entities[e1])
                    if e2 in kg.entities and e2 not in sampled_kg.entities:
                        sampled_kg.entities[e2] = deepcopy(kg.entities[e2])
                    if e1 in kg.core_nodes and e1 not in sampled_kg.core_nodes:
                        sampled_kg.core_nodes[e1] = deepcopy(kg.core_nodes[e1])
                    if e2 in kg.core_nodes and e2 not in sampled_kg.core_nodes:
                        sampled_kg.core_nodes[e2] = deepcopy(kg.core_nodes[e2])

        def path_is_present(path):
            for e1, e2 in zip(path[:-1], path[1:]):
                if not sampled_kg.has_edge(e1, e2) and not sampled_kg.has_edge(e2, e1):
                    return False
            return True

        # ensure first shortest path is present in the sampled_kg, then filter out onces that no longer exist
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

    def format_instance(self, instance, text_kg):
        instance['shortest_paths'] = [[Entity.from_dict(e) for e in path] for path in instance['shortest_paths']]
        if self.pseudonomizer:
            instance['shortest_paths'] = [[self.pseudonomizer.map_entity(e) for e in path] for path in instance['shortest_paths']]
        instance['answer'] = self.answer(instance['shortest_paths'])
        question = self.question(instance['shortest_paths'])
        instance['prompt'] = self.structure_prompt(question, text_kg)
        instance['question'] = question

    def question(self, answer_paths: List[List[Entity]]):
        return f"Your task is to find the shortest path from {answer_paths[0][0].label} to {answer_paths[0][-1].label}. For example, if the shortest path in the knowledge graph between Argentina and Mexico is through Bolivia and Colombia, then your response should be \"SHORTEST PATH: ['Argentina', 'Bolivia', 'Colombia', 'Mexico']\". Your response must begin with 'SHORTEST_PATH:' followed by the list of entities in the path following the format shown before. Note that you can use both incoming and outgoing edges to form a path. \n\n What is the shortest path from {answer_paths[0][0].label} to {answer_paths[0][-1].label}? \n Answer: SHORTEST PATH:"

    def answer(self, answer_paths: List[List[Entity]]):
        return [f"SHORTEST PATH: {str(self.ent_path_to_label_path(path))}" for path in answer_paths]

    def ent_path_to_label_path(self, path: List[Entity]):
        return [e.label for e in path]


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
