from typing import List, Union
import torch
import numpy as np
from sparse_auditvotes.sparsegraph import SparseGraph
from torch_sparse import coalesce
from torch_geometric.data import Data, Batch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
from torch_sparse import SparseTensor

def load_and_standardize(file_name):
    """
    Run gust.standardize() + make the attributes binary.

    Parameters
    ----------
    file_name
        Name of the file to load.
    Returns
    -------
    graph: gust.SparseGraph
        The standardized graph

    """
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        if 'type' in loader:
            del loader['type']
        graph = SparseGraph.from_flat_dict(loader)

    graph.standardize(no_self_loops=False)

    # binarize
    graph._flag_writeable(True)
    graph.adj_matrix[graph.adj_matrix != 0] = 1
    graph.attr_matrix[graph.attr_matrix != 0] = 1
    graph._flag_writeable(False)

    return graph

def split(labels, n_per_class=20, seed=0):
    """
    Randomly split the training data.

    Parameters
    ----------
    labels: array-like [n_nodes]
        The class labels
    n_per_class : int
        Number of samples per class
    seed: int
        Seed

    Returns
    -------
    split_train: array-like [n_per_class * nc]
        The indices of the training nodes
    split_val: array-like [n_per_class * nc]
        The indices of the validation nodes
    split_test array-like [n_nodes - 2*n_per_class * nc]
        The indices of the test nodes
    """
    np.random.seed(seed)
    nc = labels.max() + 1

    split_train, split_val = [], []
    for l in range(nc):
        perm = np.random.permutation((labels == l).nonzero()[0])
        split_train.append(perm[:n_per_class])
        split_val.append(perm[n_per_class:2 * n_per_class])

    split_train = np.random.permutation(np.concatenate(split_train))
    split_val = np.random.permutation(np.concatenate(split_val))

    assert split_train.shape[0] == split_val.shape[0] == n_per_class * nc

    split_test = np.setdiff1d(np.arange(len(labels)), np.concatenate((split_train, split_val)))

    return split_train, split_val, split_test


def split_inductive(labels, n_per_class=20, seed=None, balance_test=True, test_ratio=0.2):
    """
    Randomly split the training data.

    Parameters
    ----------
    labels: array-like [num_nodes]
        The class labels
    n_per_class : int
        Number of samples per class
    balance_test: bool
        wether to balance the classes in the test set; if true, take 10% of all nodes as test set
    seed: int
        Seed

    Returns
    -------
    split_labeled: array-like [n_per_class * nc]
        The indices of the training nodes
    split_val: array-like [n_per_class * nc]
        The indices of the validation nodes
    split_test: array-like [n_per_class * nc]
        The indices of the test nodes
    split_unlabeled: array-like [num_nodes - 3*n_per_class * nc]
        The indices of the unlabeled nodes
    """
    if seed is not None:
        np.random.seed(seed)
    nc = labels.max() + 1
    if balance_test:
        # compute n_per_class
        bins = np.bincount(labels)
        n_test_per_class = np.ceil(test_ratio * bins)
    else:
        n_test_per_class = np.ones(nc) * n_per_class

    split_labeled, split_val, split_test = [], [], []
    for label in range(nc):
        perm = np.random.permutation((labels == label).nonzero()[0])
        split_labeled.append(perm[:n_per_class])
        split_val.append(perm[n_per_class: 2 * n_per_class])
        split_test.append(perm[2 * n_per_class: 2 * n_per_class + n_test_per_class[label].astype(int)])

    split_labeled = np.random.permutation(np.concatenate(split_labeled))
    split_val = np.random.permutation(np.concatenate(split_val))
    split_test = np.random.permutation(np.concatenate(split_test))

    assert split_labeled.shape[0] == split_val.shape[0] == n_per_class * nc

    split_unlabeled = np.setdiff1d(np.arange(len(labels)), np.concatenate((split_labeled, split_val, split_test)))

    print(
        f'number of samples:\n - labeled train: {split_labeled.shape[0]} \n - unlabeled train: {split_unlabeled.shape[0]} \n - val: {split_val.shape[0]} \n - test: {split_test.shape[0]} ')
    # split the nodes into labeled training nodes, and unlabeled training nodes, validation nodes, and testing nodes. These nodes do not overlap
    return split_labeled, split_unlabeled, split_val, split_test


def filter_data_for_idx(attr, adj,device, labels, idx):
    '''filters attr, adj and labels for idx; also returns mapping from idx to corresponding indices in new objects'''
    n=adj.shape[0]
    adj_filtered = adj[idx,:]
    adj_filtered = adj_filtered[:,idx]

    # mapping indicating new indices mapping_proj[k] is the new index of k if k in idx end -1 else
    n_idx = len(idx)
    mapping_proj = -1*torch.ones(n).long()
    mapping_proj[idx] = torch.arange(n_idx)

    # map attr
    attr = attr[idx]
    labels = labels[idx]
    attr_idx = torch.LongTensor(np.stack(attr.nonzero())).to(device)
    edge_idx = torch.LongTensor(np.stack(adj_filtered.nonzero())).to(device)

    return attr_idx, edge_idx, labels, mapping_proj

def to_undirected(edge_idx, n):
    """
    Keep only edges that appear in both directions.

    Parameters
    ----------
    edge_idx : torch.Tensor [2, ?]
        The indices of the edges
    n : int
        Number of nodes

    Returns
    -------
    edge_idx : torch.Tensor [2, ?]
        The indices of the edges that appear in both directions
    """
    joined = torch.cat((edge_idx, edge_idx[[1, 0]]), 1)
    edge_idx, value = coalesce(joined, torch.ones_like(joined[0]), n, n, 'add')

    # keep only the edges that appear twice
    edge_idx = edge_idx[:, value > 1]

    return edge_idx
