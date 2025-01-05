from tasks.base_task import BaseTask
from copy import deepcopy
import random
from kg_builder import KnowledgeGraph
from samplers import graph_samplers
import re


class AggByRelationTask(BaseTask):
    """This task involves doing aggregations (count) by relationship for an entity"""
    def __init__(self, conversion_config, llm_config, pseudonomizer_config, dataset_file=None, results_file=None):
        super().__init__("AggByRelation",
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
        match = re.search(r'Answer:\s*(\d+)', response)
        extracted_number = int(match.group(1)) if match else None
        if extracted_number is None:
            return 0  # or some other logic to handle no number found
        return 1 if extracted_number == int(answer[0]) else 0

    def construct_instances(self, kg: KnowledgeGraph, num_instances=10, num_seed_entities=2, max_edges=100):
        """Constructs instances for the task."""
        for instance in range(num_instances):
            seed_entities = random.sample(list(kg.core_nodes.keys()), num_seed_entities)
            self.data.append(self.construct_instance(kg, seed_entities, instance, max_edges))

    def construct_instance(self, kg: KnowledgeGraph, seed_entities, instance_id=0, max_edges=100):
        sampled_kg = graph_samplers.sample_ego_graph_from_kg(kg, seed_entities, radius=2)
        
        # filter ent-rel pairs with only one edge
        entities_with_multiple_edges = {ent for ent in sampled_kg.entities if len(sampled_kg.graph.edges(ent)) > 1}
        sampled_kg.entities = {ent: sampled_kg.entities[ent] for ent in entities_with_multiple_edges}
        sampled_kg.graph = sampled_kg.graph.subgraph(entities_with_multiple_edges).copy()

        sampled_kg = graph_samplers.prune_kg(sampled_kg, max_edges=max_edges, max_degree=20)

        text_kg = self.text_presenter.to_list_of_edges(sampled_kg)
        
        def agg_edges_by_relation_for(ent, direction='outgoing'):
            if direction == 'outgoing':
                edges = sampled_kg.graph.out_edges(ent)
            elif direction == 'incoming':
                edges = sampled_kg.graph.in_edges(ent)
            else:
                raise ValueError("Direction must be either 'incoming' or 'outgoing'")
            
            aggregated_edges = {}  # maps from relation to edges for anchor ent
            for edge in edges:
                if direction == 'outgoing':
                    target = edge[1]
                else:
                    target = edge[0]
                edge_data = sampled_kg.graph.get_edge_data(edge[0], edge[1])
                relation = edge_data['relation']
                if relation not in aggregated_edges:
                    aggregated_edges[relation] = []
                aggregated_edges[relation].append((ent, target, edge_data))
            return aggregated_edges
        
        anchor_relation_direction_count = []
        for ent in sampled_kg.entities.keys():
            agg_edges = agg_edges_by_relation_for(ent, 'outgoing')
            for rel, edges in agg_edges.items():
                anchor_relation_direction_count.append((ent, rel, 'outgoing', len(edges)))
            agg_edges = agg_edges_by_relation_for(ent, 'incoming')
            for rel, edges in agg_edges.items():
                anchor_relation_direction_count.append((ent, rel, 'incoming', len(edges)))

        valid_options = [item for item in anchor_relation_direction_count if item[-1] > 1]

        if not valid_options:
            raise ValueError("No suitable aggregation relations")
        
        selected_option = random.choice(valid_options)
        anchor_ent, relation, direction, count = selected_option

        question = f"Using the provided knowledge graph only answer the following question. How many {direction} relations of type '{relation}' does {kg.entities[anchor_ent].label} have? Answer in the format 'Answer: <number>'."
        prompt = self.structure_prompt(question, text_kg)

        answer = [str(count)]

        return {
            'id': instance_id,
            'prompt': prompt,
            'question': question,
            'anchor_ent': anchor_ent,
            'relation': relation,
            'direction': direction,
            'text_kg': text_kg,
            'answer': answer,
            'seed_entities': seed_entities,
            'kg': sampled_kg
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

    task = AggByRelationTask(conversion_config, llm_config, pseudonomizer_config)
    seed_entities = random.sample(list(kg.core_nodes.keys()), 2)

    task.load_dataset()

    task.construct_instances(kg, num_instances=10, num_seed_entities=10, max_edges=100)

    task.save_base_data()
    task.save_dataset()

    # task.run()
    print("Finished..")
