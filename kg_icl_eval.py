# # -*- coding: utf-8 -*-
# """kg_icl_eval

# Automatically generated by Colab.

# Original file is located at
#     https://colab.research.google.com/drive/1akr8iRVk-f6C65hQxYPX0lEedsjjZOWc
# """



# from google.colab import drive
# drive.mount('/content/drive')

# # !pip install torch

# # %%capture
# !pip install transformers datasets

# # %%capture
# !pip install jsbeautifier

# # %%capture
# # !pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
# !pip install torch_geometric torch_sparse pyg_lib

# import torch
# torch.__version__

# import torch_geometric
# from pathlib import Path

# DATADIR = Path("/content/drive/MyDrive/KG_ICL_EVAL")
# DATADIR

# """# KG Construction

# ## Downloading initial dataset
# """

# # !wget https://graphs.telecom-paris.fr/data/WikiDataSets/countries.tar.gz
# # !mv countries.tar.gz /content/drive/MyDrive/KG_ICL_EVAL/countries.tar.gz
# # !tar -xvf /content/drive/MyDrive/KG_ICL_EVAL/countries.tar.gz -C /content/drive/MyDrive/KG_ICL_EVAL/

# import pandas as pd
# import os
# import copy

# COUNTRIES = DATADIR / 'countries'



# edges = pd.read_csv(os.path.join(COUNTRIES, 'edges.tsv'), sep='\t')
# entities = pd.read_csv(os.path.join(COUNTRIES, 'entities.tsv'), sep='\t')
# relations = pd.read_csv(os.path.join(COUNTRIES, 'relations.tsv'), sep='\t')
# attributes = pd.read_csv(os.path.join(COUNTRIES, 'attributes.tsv'), sep='\t')
# nodes = pd.read_csv(os.path.join(COUNTRIES, 'nodes.tsv'), sep='\t')

# entity2qid = {k:v for k, v in zip(entities['entityID'], entities['wikidataID'])}
# entity2label = {k:v for k, v in zip(entities['entityID'], entities['label'])}
# relation2pid = {k:v for k, v in zip(relations['relationID'], relations['wikidataID'])}
# relation2label = {k:v for k, v in zip(relations['relationID'], relations['label'])}
# core_nodes = torch.tensor([e for e in nodes['entityID']])

# COUNTRIES

# from torch_geometric.data import Data
# from torch_geometric.typing import Optional, Tensor, OptTensor, Union

# class KGMappingData:
#     def __init__(self, core_nodes, entity2qid, entity2label, relation2pid, relation2label):
#         self.core_nodes = core_nodes
#         self.entity2qid = entity2qid
#         self.entity2label = entity2label
#         self.relation2pid = relation2pid
#         self.relation2label = relation2label

#     def combine(self, other, disjoint=False):
#         assert not disjoint, "Comnining disjoint KGs not implemented yet"
#         core_nodes = torch.cat([self.core_nodes, other.core_nodes]).unique()
#         entity2qid = copy.deepcopy(self.entity2qid)
#         entity2qid.update(other.entity2qid)
#         entity2label = copy.deepcopy(self.entity2label)
#         entity2label.update(other.entity2label)
#         relation2label = copy.deepcopy(self.relation2label)
#         relation2label.update(other.relation2label)
#         relation2pid = copy.deepcopy(self.relation2pid)
#         relation2pid.update(other.relation2pid)
#         return KGMappingData(core_nodes, entity2qid, entity2label, relation2pid, relation2label)

#     def convert_entity_to_label(self, ent, override=None):
#         if override is not None and ent in override:
#             return override[ent]
#         return self.entity2label[ent]

#     def convert_relation_to_label(self, rel):
#         return self.relation2label[rel]


# class KG(Data):
#     def __init__(self,
#                  htr: OptTensor = None,
#                  x: Optional[Tensor] = None,
#                  edge_index: OptTensor = None,
#                  edge_attr: OptTensor = None,
#                  y: Optional[Union[Tensor, int, float]] = None,
#                  pos: OptTensor = None,
#                  time: OptTensor = None,
#                  kg_mapping: KGMappingData = None,
#                  **kwargs):
#         super().__init__(x, edge_index[:], edge_attr[:], y, pos, time, **kwargs)
#         self.htr = htr[:]
#         self.kg_mapping = kg_mapping
#         self.sort()

#     def sort(self):
#         sort_indices = self.edge_index[0].sort().indices
#         self.edge_index = self.edge_index[:, sort_indices]
#         self.edge_attr = self.edge_attr[sort_indices]
#         self.htr = self.htr[:, sort_indices]
#         sort_indices = self.edge_index[1].sort().indices
#         self.edge_index = self.edge_index[:, sort_indices]
#         self.edge_attr = self.edge_attr[sort_indices]
#         self.htr = self.htr[:, sort_indices]
#         return self

#     @staticmethod
#     def load(directory: Path):
#         edges = pd.read_csv(directory/'edges.tsv', sep='\t')
#         entities = pd.read_csv(directory/'entities.tsv', sep='\t')
#         relations = pd.read_csv(directory/'relations.tsv', sep='\t')
#         attributes = pd.read_csv(directory/'attributes.tsv', sep='\t')
#         nodes = pd.read_csv(directory/'nodes.tsv', sep='\t')

#         entity2qid = {k:v for k, v in zip(entities['entityID'], entities['wikidataID'])}
#         entity2label = {k:v for k, v in zip(entities['entityID'], entities['label'])}
#         relation2pid = {k:v for k, v in zip(relations['relationID'], relations['wikidataID'])}
#         relation2label = {k:v for k, v in zip(relations['relationID'], relations['label'])}
#         core_nodes = torch.tensor([e for e in nodes['entityID']])

#         kg_mapping = KGMappingData(core_nodes, entity2qid, entity2label, relation2pid, relation2label)

#         kg = edges_to_pyg(edges, kg_mapping)
#         attribute_kg = edges_to_pyg(attributes, kg_mapping)

#         full_kg = combine_graphs(kg, attribute_kg)
#         kg_inv = invert_graph(full_kg)
#         kg_bidir = combine_graphs(full_kg, kg_inv)

#         return kg_bidir

#     @staticmethod
#     def standardize(h, t, r):
#         if isinstance(h, torch.Tensor):
#             h_new = torch.where(r < 0, t, h)
#             t_new = torch.where(r < 0, h, t)
#             r_new = torch.where(r < 0, (-1 * r) - 1, r)
#             return h_new, t_new, r_new
#         if r < 0:
#             r = -r - 1
#             h, t = t, h
#         return h, t, r

# from torch_geometric.data import Data

# num_nodes = len(entities)
# num_relations = len(relations)

# kg_mapping = KGMappingData(core_nodes, entity2qid, entity2label, relation2pid, relation2label)

# def edges_to_pyg(edges, kg_mapping):
#     edge_index = [[h, t] for h, t in zip(edges['headEntity'], edges['tailEntity'])]
#     edge_index = torch.tensor(edge_index).t().contiguous()
#     edge_attr = torch.tensor([r for r in edges['relation']])
#     htr = torch.cat([edge_index, edge_attr.reshape(1, -1)], dim=0)
#     return KG(edge_index=edge_index, edge_attr=edge_attr, htr=htr, kg_mapping=kg_mapping)

# def edges_to_data(edges):
#     edge_index = [[h, t] for h, t in zip(edges['headEntity'], edges['tailEntity'])]
#     edge_index = torch.tensor(edge_index).t().contiguous()
#     edge_attr = torch.tensor([r for r in edges['relation']])
#     htr = torch.cat([edge_index, edge_attr.reshape(1, -1)], dim=0)
#     return edge_index, edge_attr, htr

# def invert_relations(r):
#     return torch.where(r >= 0, -1 * (r + 1), (-1 * r) - 1)

# print(invert_relations(torch.tensor([0,1,2,-1,-2])))

# def invert_graph(kg):
#     edge_index_inv = kg.edge_index[[1, 0]]
#     edge_attr_inv = invert_relations(kg.edge_attr)
#     htr_inv = torch.cat([edge_index_inv, edge_attr_inv.reshape(1, -1)], dim=0)
#     return KG(edge_index=edge_index_inv,
#               edge_attr=edge_attr_inv,
#               htr=htr_inv, kg_mapping=kg.kg_mapping)


# def combine_graphs(kg1, kg2):
#     edge_index = torch.cat([kg1.edge_index, kg2.edge_index], dim=-1)
#     edge_attr = torch.cat([kg1.edge_attr, kg2.edge_attr], dim=-1)
#     htr = torch.cat([kg1.htr, kg2.htr], dim=-1)
#     kg_mapping = kg1.kg_mapping.combine(kg2.kg_mapping)
#     return KG(edge_index=edge_index, edge_attr=edge_attr, htr=htr, kg_mapping=kg_mapping)

# kg = edges_to_pyg(edges, kg_mapping)
# # kg = kg_sorted(kg)
# attribute_kg = edges_to_pyg(attributes, kg_mapping)
# full_kg = combine_graphs(kg, attribute_kg)

# kg_inv = invert_graph(full_kg)
# kg_bidir = combine_graphs(full_kg, kg_inv)


# # edge_index, edge_attr, htr = edges_to_data(edges)

# # sort_indices = edge_index[0].sort().indices
# # edge_index = edge_index[:, sort_indices]
# # edge_attr = edge_attr[sort_indices]

# # sort_indices = edge_index[1].sort().indices
# # edge_index = edge_index[:, sort_indices]
# # edge_attr = edge_attr[sort_indices]

# # htr = torch.cat([edge_index, edge_attr.reshape(1, -1)], dim=0)

# # print(edge_index)
# # print(edge_attr)
# # print(htr)

# # data = Data(edge_index=edge_index, edge_attr=edge_attr)

# kg_bidir.edge_index

# import math

# from torch_geometric.sampler import NeighborSampler, HGTSampler, NodeSamplerInput

# sampler = NeighborSampler(kg_bidir, [2,2], is_sorted=True)
# # Either need to use is_sorted=True or use sampler.edge_permutation
# # print(sampler.edge_permutation)
# # edge_attr = edge_attr[sampler.edge_permutation]
# # edge_index = edge_index[:,sampler.edge_permutation]
# # htr = htr[:,sampler.edge_permutation]

# # outgoing_sampler = NeighborSampler(kg_inv, [3,1,1,1])

# batch = NodeSamplerInput(None, torch.tensor([0]))

# sampled_output = sampler.sample_from_nodes(batch)


# def standardize(h, t, r):
#     if r < 0:
#         r = -r - 1
#         h, t = t, h
#     return h, t, r

# for h, t, r in kg_bidir.htr[:,sampled_output.edge].t():
#     # print(h,t,r)
#     h, t, r = standardize(h, t, r)
#     # print(h,t,r)
#     print(entity2label[int(h)], relation2label[abs(int(r))], entity2label[int(t)])

# # for h, t in kg.edge_index[:,sampled_output.edge].t():
# #     # print(h,t,r)
# #     # h, t, r = standardize(h, t, r)
# #     # print(h,t,r)
# #     print(entity2label[int(h)], entity2label[int(t)])


# for h in sampled_output.node:
#   print(entity2label[int(h)])

# print(kg_bidir.htr[:,sampled_output.edge].t())
# print(kg_bidir.edge_index[:,sampled_output.edge].t())
# sampled_output.__dict__

# entity2qid[5937]

# relations

# # generated with https://www.name-generator.org.uk/?i=1e8qboz1
# fake_countries = pd.read_csv(os.path.join(DATADIR, 'fake_country_names.tsv'))
# fake_countries

# """# KG-to-Text Formats
# These can be over names or IDs
# - List of triples
# - YAML
# - JSON

# RDF



# """

# def to_list_of_edges(kg, sampled_output, pseudonym_mapping=None):
#     edges = kg_bidir.htr[:,sampled_output.edge].t()
#     text_edges = [(kg.kg_mapping.convert_entity_to_label(int(h), override=pseudonym_mapping),
#                    kg.kg_mapping.relation2label[abs(int(r))],
#                    kg.kg_mapping.convert_entity_to_label(int(t), override=pseudonym_mapping)) for h,t,r in edges]
#     text_edges = ["(" + ", ".join(e) + ")" for e in text_edges]
#     text_kg = "\n".join(text_edges)
#     text_kg = f"Knowledge Graph Edges:\n{text_kg}"
#     return text_kg

# text_kg = to_list_of_edges(kg, sampled_output)

# print(text_kg)

# class KGNeighborSampler:
#     def __init__(self, kg: KG, fanouts, num_roots):
#         self.pyg_sampler = NeighborSampler(kg, fanouts, is_sorted=True)
#         self.kg = kg
#         self.num_roots = num_roots
#         self.fanouts = fanouts
#         self.starting_nodes = kg.kg_mapping.core_nodes
#         assert self.num_roots <= len(self.kg.kg_mapping.core_nodes), 'num_roots must not be greater than num core nodes in KG'

#     def sample_subgraph(self):
#         idx = torch.randperm(len(self.starting_nodes))[:self.num_roots]
#         batch = self.starting_nodes[idx]
#         sampler_input = NodeSamplerInput(None, batch)
#         sampled_output = self.pyg_sampler.sample_from_nodes(sampler_input)
#         return sampled_output

# """# Tasks

# Path existence
# - Does a path exist between A and B

# Path identification
# - Find entities connected by a path of p1,p2,p3

# Degree of type
# - What entity has the most edges of type p1

# Comparison
# - Which entity has greater degree

# Simple Logical Filtering
# - Get people with at least one edge of type p1
# """

# import json

# class Config(dict):
#     def __init__(self, config_dict):
#         super().__init__(config)

#     @classmethod
#     def from_file(cls, config_file):
#         with open(config_file) as f:
#             config = json.load(f)
#         return cls(config)

#     def __getattr__(self, name):
#         try:
#             return self[name]
#         except KeyError:
#             raise AttributeError(f"Config has no attribute '{name}'")

#     def __setattr__(self, name, value):
#         self[name] = value

#     def __delattr__(self, name):
#         try:
#             del self[name]
#         except KeyError:
#             raise AttributeError(f"Config has no attribute '{name}'")

# # How to define task structure

# # Single fact recall
# import random
# from abc import ABC, abstractmethod


# TASKDATA = DATADIR/'datasets'


# class Task(ABC):
#     def __init__(self) -> None:
#         super().__init__()
#         self.config = None
#         self.data = None

#     @abstractmethod
#     def create_question_instances(self, kg, config):
#         pass

#     @abstractmethod
#     @staticmethod
#     def score_answer(answer, label):
#         pass

#     def create_dataset(self, source, config, dataset_name):
#         filepath = self.check_file_path(dataset_name)
#         print(f'Creating dataset with config:\n{json.dumps(config, indent=2)}')
#         kg = KG.load(DATADIR/source)
#         question_instances = self.create_question_instances(kg, config)
#         dataset = {
#             'config': config,
#             'data': question_instances
#         }
#         with open(filepath, 'w') as f:
#             f.write(jsbeautifier.beautify(json.dumps(dataset)))
#         print(f'Saving dataset at {filepath}')

#     def load_dataset(self, dataset_name):
#         filepath = self.get_file_path(dataset_name)
#         with open(filepath, 'r') as f:
#             dataset = json.load(f)
#         self.config = Config(dataset['config'])
#         self.data = dataset['data']

#     def get_file_path(self, dataset_name):
#         name = dataset_name if '.json' in dataset_name else (dataset_name + '.json')
#         savedir = TASKDATA/self.TASK_NAME
#         savedir.mkdir(parents=True, exist_ok=True)
#         filepath = savedir/name
#         return filepath

#     def check_file_path(self, dataset_name):
#         filepath = self.get_file_path(dataset_name)
#         if filepath.exists():
#             response = input(f"A datset with filepath {filepath} already exists. Would you like to overwrite? (y/n)")
#             if response not in ['y', 'Y', 'yes', 'Yes']:
#                 quit()
#         return filepath


# class SingleFactTrueOrFalse(Task):
#     # -> prompt
#     # -> kg2text
#     # -> how_to_sample_subgraph
#     # -> how to generate instance from kg sample (positive or negative)
#     # -> how to score responses

#     TASK_NAME = 'SingleFactTrueOrFalse'

#     PROMPT = "Your task is to answer questions about the following knowledge graph. Treat the provided knowledge graph as complete. Do not use any external information\n\n{kg}\n\nQuestion: {question}"

#     QUESTION_TEMPLATE1 = "Does the edge ({h}, {r}, {t}) exist in the knowledge graph? (Yes/No)"
#     QUESTION_TEMPLATE2 = "Only using explicitly stated information, does {h} have relation {r} to {t}? (Yes/No)"
#     QUESTION_TEMPLATE3 = "According to the knowledge graph, is ({h}, {r}, {t}) true? (Yes/No)"
#     QUESTION_TEMPLATES = [QUESTION_TEMPLATE1, QUESTION_TEMPLATE2, QUESTION_TEMPLATE3]

#     def create_question_instances(self, kg: KG, config: Config):
#         subgraph_sampler = KGNeighborSampler(kg, config.fanouts, config.num_roots)

#         instances = []

#         for i in range(config.num_questions):
#             # sample subgraph
#             sampled_output = subgraph_sampler.sample_subgraph()
#             # sample True or False
#             true_question = random.choice([True, False])
#             # sample edge
#             edge_sample = random.choice(sampled_output.edge)
#             subgraph = kg.htr[:, sampled_output.edge]
#             question_edge = kg.htr[:, edge_sample]
#             if not true_question: # corrupt edge if question answer is False
#                 while question_edge in subgraph.t():
#                     corrupt_point = random.choice([0,1,2])
#                     question_edge[corrupt_point] = random.choice(kg.htr[corrupt_point])

#             pseudonym_mapping = {}
#             unique_entities = set(subgraph[:1].unique().tolist())
#             unique_core_nodes = unique_entities.intersection(set(kg.kg_mapping.core_nodes.tolist()))
#             num_core = len(unique_core_nodes)
#             subgraph_pseudonyms = fake_countries['names'].sample(num_core) # TODO: replace with kg.kg_mapping.pseudonyms

#             for node, pseudonym in zip(unique_core_nodes, subgraph_pseudonyms):
#                 pseudonym_mapping[node] = pseudonym

#             h,t,r = kg.standardize(*question_edge)

#             h_label = kg.kg_mapping.convert_entity_to_label(int(h), override=pseudonym_mapping)
#             t_label = kg.kg_mapping.convert_entity_to_label(int(t), override=pseudonym_mapping)
#             r_label = kg.kg_mapping.convert_relation_to_label(int(r))

#             question = random.choice(self.QUESTION_TEMPLATES)
#             question = question.format(h=h_label, r=r_label, t=t_label)
#             print(question)
#             print(true_question)

#             instance = {}
#             instance["id"] = i
#             instance["question"] = question
#             instance["label"] = "yes" if true_question else "no"
#             instance["subgraph"] = subgraph.tolist()
#             instance["pseudonym_mapping"] = pseudonym_mapping
#             instances.append(instance)
#         return instances

#     @staticmethod
#     def score_answer(answer, label):
#         return 1.0 if answer.lower() == label.lower() else 0.0

# class A:
#     def test(self):
#         print(self.VAR)

# class B(A):
#     VAR = 3

# b = B()
# b.test()

# from pprint import pprint
# import jsbeautifier

# config = Config.from_file(DATADIR/"single_fact_true_or_false_config.json")
# print(config)
# task = SingleFactTrueOrFalse()
# task.create_dataset('countries', config, 'testing.json')
# task2 = SingleFactTrueOrFalse()
# task2.load_dataset('testing.json')
# print(task2.data)

# # For LLM class start by just using langchain, can use model.invoke(text) -> result.content

# # Runner:
#   # Task
#   # kg source
#   # task hyperparameters
#     # other task specific hyperparameters (e.g. path length, sampling type)
#   # LLM
#   # text2kg


#   # Will load or create task dataset based on task, kg source, and task hyperparams
#   # Will then run Task using LLM and text2kg
#   # Will score the output
#   # Will save results (output and scores)
#     # Results file based on Task, text2kg, LLM (source and hyperparams should be saved as well)

# """# Evaluate based on context size

# What will results look like?

# Main results:

# ---------------------- Tasks....

# text2kg formats

# ... (average acc across models)

# ...

# # (Future work) Tasks that use Text + KG

# Fact checking between KG and Text

# Question answering based on text and KG retrieval
# """