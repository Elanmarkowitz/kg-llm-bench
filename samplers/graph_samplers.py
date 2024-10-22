from copy import deepcopy
import random
import networkx as nx

from kg_builder import KnowledgeGraph


def sample_ego_graph_from_kg(self, kg: KnowledgeGraph, seed_entities, radius=1):
    """
    Samples the ego graph of a given radius around the seed entities.
    
    :param seed_entities: List of seed entities (nodes).
    :param radius: Radius of the ego graph.
    :return: A subgraph containing the ego graph.
    """
    ego_graphs = []
    for seed in seed_entities:
        ego_graph = nx.ego_graph(kg.graph, seed, radius=radius)
        ego_graphs.append(ego_graph)
    
    # Combine all ego graphs into one subgraph
    combined_graph = nx.compose_all(ego_graphs)
    return combined_graph
    

def prune_kg(kg: KnowledgeGraph, max_edges, max_degree):
    """
    Prunes the graph by removing edges and nodes based on the specified criteria.
    
    :param max_edges: Maximum number of edges to keep.
    :param max_degree: Maximum degree of nodes, sample edges to keep below.
    :return: The pruned graph.
    """
    kg_copy = deepcopy(kg)
    graph = kg_copy.graph

    # Remove nodes based on the maximum degree
    high_degree_nodes = [node for node, degree in graph.degree() if degree > max_degree]
    # Sample edges to keep based on the maximum degree
    edges_to_keep = set()
    for node in high_degree_nodes:
        edges = list(graph.edges(node))
        if len(edges) > max_degree:
            edges_to_keep.update(random.sample(edges, max_degree))
        else:
            edges_to_keep.update(edges)
    
    # Remove all edges not in the sampled set
    edges_to_remove = set(graph.edges()) - edges_to_keep
    graph.remove_edges_from(edges_to_remove)
    
    # Remove edges based on the maximum number of edges
    edges_to_remove = random.sample(list(graph.edges()), len(graph.edges()) - max_edges)
    graph.remove_edges_from(edges_to_remove)
    
    return kg_copy