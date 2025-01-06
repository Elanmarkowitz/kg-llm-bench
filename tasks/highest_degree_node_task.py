from tasks.base_task import BaseTask
from copy import deepcopy
import random
from kg_builder import KnowledgeGraph
from samplers import graph_samplers
import re


class HighestDegreeNodeTask(BaseTask):
    """This task involves finding the entity with the highest degree (incoming, outgoing, or total) relations."""

    def __init__(self, conversion_config, llm_config, pseudonomizer_config, dataset_file=None, results_file=None):
        super().__init__("HighestDegreeNode",
                         conversion_config,
                         llm_config,
                         pseudonomizer_config,
                         dataset_file,
                         results_file)

    def run(self):
        self.results = deepcopy(self.data)
        for instance in self.results:
            if instance is None:
                continue
            prompt = instance['prompt']
            response = self.model(prompt)
            instance['response'] = response
            instance['score'] = self.evaluate_response(response, instance['answer'])
        self.save_results()

    def evaluate_response(self, response, answer):
        match = re.search(r'Answer:\s*([\w\s]+)', response)
        extracted_string = match.group(1) if match else None
        if extracted_string is None:
            return 0  # or some other logic to handle no string found
        answer = answer[0].replace("Answer:", "").strip()
        return 1 if extracted_string == answer else 0

    def construct_instances(self, kg: KnowledgeGraph, num_instances=10, num_seed_entities=2, max_edges=100):
        """Constructs instances for the task."""
        for instance in range(num_instances):
            seed_entities = random.sample(list(kg.core_nodes.keys()), num_seed_entities)
            self.data.append(self.construct_instance(kg, seed_entities, instance, max_edges))

    def construct_instance(self, kg: KnowledgeGraph, seed_entities, instance_id=0, max_edges=100):
        sampled_kg = graph_samplers.sample_ego_graph_from_kg(kg, seed_entities, radius=2)
        sampled_kg = graph_samplers.prune_kg(sampled_kg, max_edges=max_edges, max_degree=20)

        text_kg = self.text_presenter.to_list_of_edges(sampled_kg)

        # Determine the highest degree node by randomly selecting the direction of the edge
        edge_direction = random.choice(['outgoing', 'incoming', 'total'])
        if edge_direction == 'outgoing':
            highest_degree_node = max(sampled_kg.graph.nodes, key=lambda n: sampled_kg.graph.out_degree(n))
        elif edge_direction == 'incoming':
            highest_degree_node = max(sampled_kg.graph.nodes, key=lambda n: sampled_kg.graph.in_degree(n))
        else:
            highest_degree_node = max(sampled_kg.graph.nodes, key=lambda n: sampled_kg.graph.degree(n))

        highest_degree_label = kg.entities[highest_degree_node].label

        question = f"Using the provided knowledge graph only answer the following question. Which entity has the highest number of {edge_direction} relations in the provided knowledge graph? Answer in the format 'Answer: <entity>'."
        prompt = self.structure_prompt(question, text_kg)

        answer = [f"Answer: {highest_degree_label}"]

        return {
            'id': instance_id,
            'prompt': prompt,
            'question': question,
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

    task = HighestDegreeNodeTask(conversion_config, llm_config, pseudonomizer_config)
    seed_entities = random.sample(list(kg.core_nodes.keys()), 2)

    task.load_dataset()

    task.construct_instances(kg, num_instances=10, num_seed_entities=10, max_edges=100)

    task.save_base_data()
    task.save_dataset()

    task.run()
    print("Finished..")
