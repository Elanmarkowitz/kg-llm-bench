from copy import deepcopy
import random
import networkx as nx

from kg_builder import KnowledgeGraph


def sample_ego_graph_from_kg(kg: KnowledgeGraph, seed_entities, radius=1):
    """
    Samples the ego graph of a given radius around the seed entities.
    
    :param seed_entities: List of seed entities (nodes).
    :param radius: Radius of the ego graph.
    :return: A subgraph containing the ego graph.
    """
    kg_copy = deepcopy(kg)
    ego_graphs = []
    for seed in seed_entities:
        ego_graph = nx.ego_graph(kg.graph, seed, radius=radius)
        ego_graphs.append(ego_graph)
    
    # Combine all ego graphs into one subgraph
    combined_graph = nx.compose_all(ego_graphs)
    kg_copy.graph = combined_graph
    refine_entities_and_relations_for_sample(kg_copy)
    return kg_copy
    

def prune_kg(kg: KnowledgeGraph, 
             max_edges=None, 
             max_degree=None,
             max_out_degree=None,
             max_in_degree=None):
    """
    Prunes the graph by removing edges and nodes based on the specified criteria.
    
    :param max_edges: Maximum number of edges to keep.
    :param max_degree: Maximum degree of nodes, sample edges to keep below.
    :return: The pruned graph.
    """
    kg_copy = deepcopy(kg)
    graph = kg_copy.graph

    # Process edges to ensure max out-degree is met
    if max_out_degree is not None:
        for node in graph.nodes:
            out_degree = graph.out_degree(node)
            if out_degree > max_out_degree:
                edges = list(graph.out_edges(node))
                edges_to_remove = random.sample(edges, out_degree - max_out_degree)
                graph.remove_edges_from(edges_to_remove)

    # Process nodes for in-degree
    if max_in_degree is not None:
        for node in graph.nodes:
            in_degree = graph.in_degree(node)
            if in_degree > max_in_degree:
                edges = list(graph.in_edges(node))
                edges_to_remove = random.sample(edges, in_degree - max_in_degree)
                graph.remove_edges_from(edges_to_remove)

    # Process nodes for total degree
    if max_degree is not None:
        for node in graph.nodes:
            degree = graph.degree(node)
            if degree > max_degree:
                edges = list(graph.in_edges(node)) + list(graph.out_edges(node))
                edges_to_remove = random.sample(edges, degree - max_degree)
                graph.remove_edges_from(edges_to_remove)
    
    # Remove edges based on the maximum number of edges
    if max_edges is not None and max_edges < len(graph.edges):
        edges_to_remove = random.sample(list(graph.edges()), len(graph.edges()) - max_edges)
        graph.remove_edges_from(edges_to_remove)
    kg_copy.graph = graph  # not neccessary, but for clarity
    
    refine_entities_and_relations_for_sample(kg_copy)

    return kg_copy


def refine_entities_and_relations_for_sample(kg: KnowledgeGraph) -> KnowledgeGraph:
    selected_nodes = kg.graph.nodes
    selected_relations = set(rel for _, _, rel in kg.graph.edges(data='relation'))
    kg.entities = {idx: ent for idx, ent in kg.entities.items() if idx in selected_nodes}
    kg.core_nodes = {idx: ent for idx, ent in kg.core_nodes.items() if idx in selected_nodes}
    kg.relations = {idx: rel for idx, rel in kg.relations.items() if rel.label in selected_relations}
    return kg