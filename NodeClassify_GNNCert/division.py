import numpy as np
import networkx as nx
import hashlib
import torch
import random
from copy import deepcopy, copy

DEFAULT=0

# Define feature functions
def get_node_tags(graph):
    return np.array(graph.node_tags)

def get_degree_centrality(graph):
    return nx.degree_centrality(graph.g).values()

def get_betweenness_centrality(graph):
    return nx.betweenness_centrality(graph.g).values()

def get_closeness_centrality(graph):
    return nx.closeness_centrality(graph.g).values()

def get_eigenvector_centrality(graph):
    return nx.eigenvector_centrality(graph.g).values()

def get_node_id(n):
    return np.arange(n)



# Map feature method to corresponding function
features_func_map = {
    "tags": get_node_tags,
    "degree": get_degree_centrality,
    "betweenness": get_betweenness_centrality,
    "closeness": get_closeness_centrality,
    "eigenvector": get_eigenvector_centrality,
    "id": get_node_id
}

# Define hash functions
def md5_hash(x):
    return int(hashlib.md5(str(x).encode()).hexdigest(), 16)

def sha256_hash(x):
    return int(hashlib.sha256(str(x).encode()).hexdigest(), 16)

def sha512_hash(x):
    return int(hashlib.sha512(str(x).encode()).hexdigest(), 16)

# Map hash method to corresponding function
hash_func_map = {
    "hash": hash,
    "md5": md5_hash,
    "sha256": sha256_hash,
    "sha512": sha512_hash
}

def get_edge_hash(edge_idx, hash_func, node_id):
    """
    Compute hash values for the edges in a given graph

    Returns:
    edge_hash (list): A list of hash values for the edges in the graph
    """
    num_edges=edge_idx.shape[1]
    node_categories = [f"{node_id[edge_idx[0,i]]};{node_id[edge_idx[1,i]]}" for i in range(num_edges)]
    edge_hash = [hash_func(category) for category in node_categories]
    return edge_hash

def get_node_hash(hash_func, node_id):
    """
    Compute hash values for the NODEs in a given graph

    Parameters:
    hash_func (function): The hash function to use
    node_id: Node ids
    Returns:
    node_hash (list): A list of hash values for the NODEs in the graph
    """
    node_hash = [hash_func(str(id)) for id in node_id]

    return node_hash


def graph_feature_division(n,attr_idx,edge_idx,hash_type,Ts=1,Tf=30):
    """
    Divide a graph into subgraphs based on their NODE hash values

    Parameters:
    graph (SV2Graph): The input graph
    args (argparse.Namespace): A namespace containing the required arguments

    Returns:
    subgraphs (list): A list of subgraphs obtained by dividing the input graph
    """
    # Compute the features and NODE hash values for the graph
    hash_func = hash_func_map.get(hash_type)
    node_id = get_node_id(n)
    node_hash = get_node_hash(hash_func, node_id)
    # Divide the graph based on the NODE hash values
    # group = [h % Tf for h in node_hash] (for n nodes)
    group = {}
    for i,h in enumerate(node_hash):
        group[i] = h % Tf
    # Create subgraphs for each group
    subgraphs_attrs = []
    for i in range(Tf):
        i=0
        subgraph_attr = copy(attr_idx.cpu())
        group_map_idx = torch.tensor([group[idx.item()] for idx in subgraph_attr[0, :]])
        subgraph_attr[1,group_map_idx != i] = DEFAULT
        subgraphs_attrs.append(subgraph_attr.to(attr_idx.device))
    return [edge_idx]*Tf,subgraphs_attrs

def graph_structure_division(n,attr_idx,edge_idx,hash_type,Ts=30,Tf=1):
    """
    Divide a graph into subgraphs based on their edge hash values
    Returns:
    subgraphs (list): A list of subgraphs obtained by dividing the input graph
    """
    hash_func = hash_func_map.get(hash_type)
    # Compute the features and edge hash values for the graph
    node_id = get_node_id(n)
    edge_hash = get_edge_hash(edge_idx, hash_func, node_id)
    # Divide the graph based on the edge hash values
    group = np.array([h % Ts for h in edge_hash])
    # Create subgraphs for each group
    subgraph_edges = []
    for i in range(Ts):
        # subgraph_edge = copy(edge_idx)
        subgraph_edge = edge_idx[:,group == i]
        subgraph_edges.append(subgraph_edge)
        # print(subgraph_edge.shape[1])
    return subgraph_edges,[attr_idx]*Ts


def graph_feature_structure_division(graph, args):
    """
    Divide a graph into subgraphs based on their edge & node hash values

    Parameters:
    graph (SV2Graph): The input graph
    args (argparse.Namespace): A namespace containing the required arguments

    Returns:
    subgraphs (list): A list of subgraphs obtained by dividing the input graph
    """
    # Preserve the original features, since edge_division may affect them (e.g., degree).
    features = args.features_func(graph)
    node_hash = get_node_hash(graph.g, hash_func, features)
    group = np.array([h % args.num_group for h in node_hash])
    # Divide the graph based on its structure.
    graphs = graph_structure_division(graph, args)
    # Divide each subgraph based on features.
    subgraphs = []
    for graph in graphs:
        for i in range(args.num_group):
            subgraph = copy(graph)
            subgraph.node_features[group != i] = DEFAULT
            subgraphs.append(subgraph)
    
    return subgraphs

# Map division method to corresponding function
division_func_map = {
    "feature": graph_feature_division,
    "structure": graph_structure_division,
    "both": graph_feature_structure_division,
}

def spread_function(divided_edges, divided_attrs, n_subgraphs, Td, division_method, seed):
    # to specify a mapping from [dk] to [dk]^d
    # n_subgraphs = k*d
    random.seed(seed)
    R = random.sample(range(n_subgraphs), Td)
    num_division = len(divided_edges)

    # for every division, pread it into d subsets. (repeat for d times)
    # h_spread(j)={j+r_t mod kd | t \in [d]}
    spread_idx = [[] for i in range(n_subgraphs)]
    subgraph_edges = [[] for i in range(n_subgraphs)]
    subgraph_attrs = [[] for i in range(n_subgraphs)]
    for i in range(num_division):
        for r_t in R:
            spread_idx[(i + r_t) % n_subgraphs].append(i)
            subgraph_edges[(i + r_t) % n_subgraphs].append(divided_edges[i])
            subgraph_attrs[(i + r_t) % n_subgraphs].append(divided_attrs[i])
    if division_method=='feature':
        subgraph_edges = divided_edges
        subgraph_attrs = [torch.cat(attrs, dim=1) for attrs in subgraph_attrs]
    elif division_method=='structure':
        subgraph_edges = [torch.cat(edges, dim=1) for edges in subgraph_edges]
        subgraph_attrs = divided_attrs
    else:
        subgraph_edges = [torch.cat(edges, dim=1) for edges in subgraph_edges]
        subgraph_attrs = [torch.cat(attrs, dim=1) for attrs in subgraph_attrs]
    return spread_idx, subgraph_edges, subgraph_attrs

