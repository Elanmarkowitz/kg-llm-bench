from tasks.base_task import BaseTask, InstanceConstructionError
from copy import deepcopy
import random
from kg_builder import Entity, KnowledgeGraph, Relation
from samplers import graph_samplers
import re


class AggNeighborPropertiesTask(BaseTask):
    """This task involves aggregating neighbor properties for an entity"""
    def __init__(self, conversion_config, llm_config, pseudonomizer_config, 
                 base_dataset_file=None, dataset_file=None, results_file=None):
        super().__init__("AggNeighborProperties",
                         conversion_config,
                         llm_config,
                         pseudonomizer_config,
                         base_dataset_file,
                         dataset_file,
                         results_file)

    def evaluate_response(self, response, answer):
        match = re.search(r'Answer:\s*(\d+)', response)
        extracted_number = int(match.group(1)) if match else None
        if extracted_number is None:
            return 0  # or some other logic to handle no number found
        return 1 if extracted_number == int(answer[0]) else 0


    def construct_instance(self, kg: KnowledgeGraph, seed_entities, instance_id=0, max_edges=100):
        sampled_kg = graph_samplers.sample_ego_graph_from_kg(kg, seed_entities, radius=2)
        
        entities_with_multiple_edges = {ent for ent in sampled_kg.entities if len(sampled_kg.get_neighbors(ent)) > 1}
        sampled_kg.entities = {ent: sampled_kg.entities[ent] for ent in entities_with_multiple_edges}
        sampled_kg.graph = sampled_kg.graph.subgraph(entities_with_multiple_edges).copy()
        sampled_kg = graph_samplers.refine_entities_and_relations_for_sample(sampled_kg)

        sampled_kg = graph_samplers.prune_kg(sampled_kg, max_edges=max_edges, max_degree=20)

        anchor_relation_counts = []

        for anchor_ent in sampled_kg.entities.keys():
            # get neighbors
            neighbors = sampled_kg.get_neighbors(anchor_ent, fwd=True, bkw=True)
            for relation in sampled_kg.relations.values():
                # get count of neighbors that have an edge of type relation
                neighbors_with_relation = [
                    neighbor for neighbor in neighbors  # note this is only outgoing edges
                    if self.ent_has_relation(sampled_kg, neighbor, relation)
                ]
                count = len(neighbors_with_relation)
                if count > 0:
                    anchor_relation_counts.append((anchor_ent, relation.label, count))

        if not anchor_relation_counts:
            raise InstanceConstructionError("No suitable anchor entity and relation found")

        # Select a random option from the valid anchor_relation_counts
        counts_set =  set([count for _,_,count in anchor_relation_counts])
        selected_count = random.choice(list(counts_set))
        options = [(anchor, relation, count) for anchor, relation, count in anchor_relation_counts
                   if count == selected_count]
        selected_option = random.choice(options)
        anchor_ent, relation, count = selected_option

        question = self.question(sampled_kg.entities[anchor_ent], relation)

        pseudo_kg = self.pseudonymize_kg(sampled_kg)

        answer = [str(count)]

        return {
            'id': instance_id,
            'question': question,
            'anchor_ent': sampled_kg.entities[anchor_ent],
            'relation': relation,
            'answer': answer,
            'seed_entities': seed_entities,
            'kg': sampled_kg,
            'pseudo_kg': pseudo_kg,
            'pseudonomizer_mapping': self.pseudonomizer.copy_mapping()
        }

    def ent_has_relation(self, sampled_kg, anchor_ent, relation):
        """
        Return true if the anchor ent has an outgoing edge of type relation
        
        Args:
            sampled_kg: Knowledge graph object containing the DiGraph
            anchor_ent: The entity node to check neighbors for
            relation: Relation object containing the label to match
        
        Returns:
            bool: True if the entity has at least one outgoing edge of the specified relation type
        """
        # Get all successors (nodes with incoming edges from anchor_ent)
        successors = sampled_kg.graph.successors(anchor_ent)
        
        # Check each successor for the specified relation
        for neighbor in successors:
            edge_data = sampled_kg.graph.get_edge_data(anchor_ent, neighbor)
            if edge_data and edge_data.get('relation') == relation.label:
                return True
                
        return False

    def construct_formatted_instances(self):
        self.formatted_data = deepcopy(self.base_data)
        for instance in self.formatted_data:
            assert 'kg_path' in instance
            kg = instance.pop('kg')
            instance['anchor_ent'] = Entity.from_dict(instance['anchor_ent'])
            if self.pseudonomizer:
                if 'pseudo_kg' in instance:
                    kg = instance.pop('pseudo_kg')
                else:
                    if not 'pseudonomizer_mapping' in instance:
                        raise ValueError("Pseudonomizer config set but no pseudonomizer mapping in the base data")
                    self.pseudonomizer.load_mapping(instance['pseudonomizer_mapping'])
                    kg = self.pseudonomizer.pseudonymize(kg)
                    # task specific conversions
                    instance['anchor_ent'] = self.pseudonomizer.map_entity(instance['anchor'])
                # answer is Yes/No so no change needed
            
            question = self.question(instance['anchor_ent'], instance['relation'])
            text_kg = self.text_presenter.convert(kg)

            instance['text_kg'] = text_kg
            instance['prompt'] = self.structure_prompt(question, text_kg)
            instance['question'] = question
            # answer is a count so no change needed

    def question(self, anchor_ent: Entity, relation: str):
        return f"Using the provided knowledge graph only answer the following question. How many of the directly connected entities to '{anchor_ent.label}' have an outgoing property of type '{relation}' in the knowledge graph? You must answer in the format 'Answer: <number>'."

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

    task = AggNeighborPropertiesTask(conversion_config, llm_config, pseudonomizer_config)
    seed_entities = random.sample(list(kg.core_nodes.keys()), 2)

    try:
        task.load_base_dataset()
    except ValueError:
        task.construct_base_instances(kg, num_instances=10, num_seed_entities=1, max_edges=500)
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