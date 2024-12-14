from typing import List, Union
import torch
import numpy as np
from sparse_auditvotes.sparsegraph import SparseGraph
from torch_sparse import coalesce
from torch_geometric.data import Data, Batch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
import time

def get_time():
    torch.cuda.synchronize()
    return time.time()

def binary_perturb(data, pf_minus, pf_plus):
    """
    Randomly flip bits.

    Parameters
    ----------
    data: torch.Tensor [b, ?, ?]
        The indices of the non-zero elements
    pf_minus: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero
    pf_plus : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one

    Returns
    -------
    data_perturbed: torch.Tensor [b, ?, ?]
        The indices of the non-zero elements after perturbation
    """

    to_del = torch.cuda.BoolTensor(data.shape).bernoulli_(1 - pf_minus)
    to_add = torch.cuda.BoolTensor(data.shape).bernoulli_(pf_plus)

    data_perturbed = data * to_del + (1 - data) * to_add
    return data_perturbed


def retain_k_elements(data_idx, k, undirected, shape=None):
    """
    Randomly retain k (non-zero) elements.

    Parameters
    ----------
    data_idx: torch.Tensor [2, ?]
        The indices of the non-zero elements.
    k : int
        Number of elements to remove.
    undirected : bool
        If true for every (i, j) also perturb (j, i).
    shape: (int, int)
        If shape=None only retain k non-zero elements,
        else retain k of all possible shape[0]*shape[0] pairs (including zeros).

    Returns
    -------
    per_data_idx: torch.Tensor [2, ?]
        The indices of the non-zero elements after perturbation.
    """
    if undirected:
        data_idx = data_idx[:, data_idx[0] < data_idx[1]]

    if shape is not None:
        n, m = shape
        if undirected:
            # undirected makes sense only for square (adjacency matrices)
            assert n == m 
            total_pairs = n*(n+1)//2
        else:
            total_pairs = n*m

        k = k*data_idx.shape[1]//total_pairs

    rnd_idx = torch.randperm(data_idx.shape[1]).to(args.device)[:k]
    per_data_idx = data_idx[:, rnd_idx]

    if undirected:
        per_data_idx = torch.cat((per_data_idx, per_data_idx[[1, 0]]), 1)

    return per_data_idx


def sparse_perturb(data_idx, pf_minus, pf_plus, n, m, undirected):
    """
    Randomly flip bits.

    Parameters
    ----------
    data_idx: torch.Tensor [2, ?]
        The indices of the non-zero elements
    pf_minus: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero
    pf_plus : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one
    n : int
        The shape of the tensor
    m : int
        The shape of the tensor
    undirected : bool
        If true for every (i, j) also perturb (j, i)

    Returns
    -------
    perturbed_data_idx: torch.Tensor [2, ?]
        The indices of the non-zero elements after perturbation
    """
    if undirected:
        # select only one direction of the edges, ignore self loops
        data_idx = data_idx[:, data_idx[0] < data_idx[1]]

    w_existing = torch.ones_like(data_idx[0])
    to_del = torch.cuda.BoolTensor(data_idx.shape[1]).bernoulli_(pf_minus)
    w_existing[to_del] = 0

    nadd = np.random.binomial(n * m, pf_plus)  # 6x faster than PyTorch
    nadd_with_repl = int(np.log(1 - nadd / (n * m)) / np.log(1 - 1 / (n * m)))
    to_add = data_idx.new_empty([2, nadd_with_repl])
    to_add[0].random_(n * m)
    to_add[1] = to_add[0] % m
    to_add[0] = to_add[0] // m
    if undirected:
        # select only one direction of the edges, ignore self loops
        assert n == m
        to_add = to_add[:, to_add[0] < to_add[1]]

    w_added = torch.ones_like(to_add[0])

    # if an edge already exists but has been removed do not add it back
    # hence we coalesce with the min value
    joined, weights = coalesce(torch.cat((data_idx, to_add), 1),
                               torch.cat((w_existing, w_added), 0),
                               n, m, 'min')

    per_data_idx = joined[:, weights > 0]

    if undirected:
        per_data_idx = torch.cat((per_data_idx, per_data_idx[[1, 0]]), 1)

    return per_data_idx


def sparse_perturb_adj_batch(data_idx, nnodes, pf_minus, pf_plus, undirected):
    """
    Randomly flip bits.

    Parameters
    ----------
    data_idx: torch.Tensor [2, ?]
        The indices of the non-zero elements
    nnodes : array_like, dtype=int
        Number of nodes per graph
    pf_minus: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero
    pf_plus : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one
    undirected : bool
        If true for every (i, j) also perturb (j, i)

    Returns
    -------
    perturbed_data_idx: torch.Tensor [2, ?]
        The indices of the non-zero elements after perturbation
    """
    if undirected:
        # select only one direction of the edges, ignore self loops
        data_idx = data_idx[:, data_idx[0] < data_idx[1]]

    w_existing = torch.ones_like(data_idx[0])
    to_del = torch.cuda.BoolTensor(data_idx.shape[1]).bernoulli_(pf_minus)
    w_existing[to_del] = 0

    offsets = torch.cat((nnodes.new_zeros(1), torch.cumsum(nnodes, dim=0)[:-1]))
    nedges = torch.cumsum(nnodes**2, dim=0)
    offsets2 = torch.cat((nedges.new_zeros(1), nedges[:-1]))
    nedges_total = nedges[-1].item()
    nadd = np.random.binomial(nedges_total, pf_plus)  # 6x faster than PyTorch
    nadd_with_repl = int(np.log(1 - nadd / nedges_total) / np.log(1 - 1 / nedges_total))
    to_add = data_idx.new_empty([2, nadd_with_repl])
    to_add[0].random_(nedges_total)
    add_batch = (to_add[0][:, None] >= nedges[None, :]).sum(1)
    to_add[0] -= offsets2[add_batch]
    to_add[1] = to_add[0] % nnodes[add_batch]
    to_add[0] = to_add[0] // nnodes[add_batch]
    to_add += offsets[add_batch][None, :]
    if undirected:
        # select only one direction of the edges, ignore self loops
        to_add = to_add[:, to_add[0] < to_add[1]]

    w_added = torch.ones_like(to_add[0])

    # if an edge already exists but has been removed do not add it back
    # hence we coalesce with the min value
    nnodes_total = torch.sum(nnodes)
    joined, weights = coalesce(torch.cat((data_idx, to_add), 1),
                               torch.cat((w_existing, w_added), 0),
                               nnodes_total, nnodes_total, 'min')

    per_data_idx = joined[:, weights > 0]

    if undirected:
        per_data_idx = torch.cat((per_data_idx, per_data_idx[[1, 0]]), 1)

    # Check that there are no off-diagonal edges
    # batch0 = (to_add[0][:, None] >= nnodes.cumsum(0)[None, :]).sum(1)
    # batch1 = (to_add[1][:, None] >= nnodes.cumsum(0)[None, :]).sum(1)
    # assert torch.all(batch0 == batch1)

    return per_data_idx


def sparse_perturb_multiple(data_idx, pf_minus, pf_plus, n, m, undirected, nsamples, offset_both_idx):
    """
    Randomly flip bits.

    Parameters
    ----------
    data_idx: torch.Tensor [2, ?]
        The indices of the non-zero elements
    pf_minus: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero
    pf_plus : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one
    n : int
        The shape of the tensor
    m : int
        The shape of the tensor
    undirected : bool
        If true for every (i, j) also perturb (j, i)
    nsamples : int
        Number of perturbed samples
    offset_both_idx : bool
        Whether to offset both matrix indices (for adjacency matrix)

    Returns
    -------
    perturbed_data_idx: torch.Tensor [2, ?]
        The indices of the non-zero elements of multiple concatenated matrices
        after perturbation
    """

    if undirected:
        # select only one direction of the edges, ignore self loops
        data_idx = data_idx[:, data_idx[0] < data_idx[1]]

    idx_copies = copy_idx(data_idx, n, nsamples, offset_both_idx)
    w_existing = torch.ones_like(idx_copies[0])
    to_del = torch.cuda.BoolTensor(idx_copies.shape[1]).bernoulli_(pf_minus)
    w_existing[to_del] = 0

    if offset_both_idx:
        assert n == m
        # nadd_persample_np = np.random.binomial(n * m, pf_plus, size=nsamples)  # 6x faster than PyTorch
        # nadd_persample = torch.LongTensor(nadd_persample_np).to(data_idx.device)
        # I think that it might have something wrong:-----
        # nadd_persample_with_repl = torch.round(torch.log(1 - nadd_persample / (n * m))
        #                                        / np.log(1 - 1 / (n * m))).long()
        # nadd_with_repl = nadd_persample_with_repl.sum()
        # to_add = data_idx.new_empty([2, nadd_with_repl])
        # to_add[0].random_(n * m)
        # to_add[1] = to_add[0] % m
        # to_add[0] = to_add[0] // m
        # to_add = offset_idx(to_add, nadd_persample_with_repl, m, [0, 1])
        #-------------------------------------------------
        # use no replacement sampling
        if pf_plus==0:
            to_add=torch.empty([2,0],dtype=torch.long,device=data_idx.device)
        else:
            # nadd_persample_np = np.random.binomial(n * m, pf_plus, size=nsamples)  # 6x faster than PyTorch
            # nadd_persample = torch.LongTensor(nadd_persample_np).to(data_idx.device)
            # nadd_persample_sum = nadd_persample.sum()
            # to_add_flat = torch.cat([torch.multinomial(torch.ones(n * m, device=data_idx.device), num, replacement=False) for num in nadd_persample_np])
            rands = [torch.empty(1, n * m).uniform_(0, 1) for i in range(nsamples)]
            rands_bool = [rand<=pf_plus for rand in rands]
            nadd_persample = torch.tensor([rand.sum() for rand in rands_bool]).to(data_idx.device)
            to_add_flat = torch.cat([rand.nonzero()[:,1] for rand in rands_bool], 0)
            to_add = torch.stack([to_add_flat // m, to_add_flat % m], 0).to(data_idx.device)
            to_add = offset_idx(to_add, nadd_persample, m, [0, 1])
            if undirected:
                # select only one direction of the edges, ignore self loops
                to_add = to_add[:, to_add[0] < to_add[1]]
    else:
        nadd = np.random.binomial(nsamples * n * m, pf_plus)  # 6x faster than PyTorch
        nadd_persample_sum = nadd.sum()
        # nadd_with_repl = int(np.round(np.log(1 - nadd / (nsamples * n * m))
        #                               / np.log(1 - 1 / (nsamples * n * m))))
        # to_add = data_idx.new_empty([2, nadd_with_repl])
        # to_add[0].random_(nsamples * n * m)
        # to_add[1] = to_add[0] % m
        # to_add[0] = to_add[0] // m
        to_add_flat = torch.randperm(nsamples * n * m)[:nadd_persample_sum]
        to_add = torch.stack([to_add_flat // m, to_add_flat % m], 0).to(data_idx.device)

    w_added = torch.ones_like(to_add[0])

    if offset_both_idx:
        mb = nsamples * m
    else:
        mb = m

    # if an edge already exists but has been removed do not add it back
    # hence we coalesce with the min value
    # The coalesce function is used to combine duplicate indices in a sparse tensor and sum their values.
    joined, weights = coalesce(torch.cat((idx_copies, to_add), 1),
                               torch.cat((w_existing, w_added), 0),
                               nsamples * n, mb, 'min')

    per_data_idx = joined[:, weights > 0]

    if undirected:
        per_data_idx = torch.cat((per_data_idx, per_data_idx[[1, 0]]), 1)

    # Check that there are no off-diagonal edges
    # if offset_both_idx:
    #     batch0 = to_add[0] // n
    #     batch1 = to_add[1] // n
    #     assert torch.all(batch0 == batch1)

    return per_data_idx

#These are too slowly:----
# 1) to_add_flat = torch.cat([torch.randperm(n * m)[:num] for num in nadd_persample])
# 2) rand_tensors = [np.random.choice(range(0, n * m),
#                                  size=num,
#                                  p=np.ones(n * m) / (n * m),
#                                  replace=False) for num in nadd_persample_np]
# to_add_flat = torch.cat([torch.from_numpy(rand_tensor) for rand_tensor in rand_tensors])

#This is fast but:-----
#RuntimeError: number of categories cannot exceed 2^24:
# nadd_persample_np = np.random.binomial(n * m, pf_plus, size=nsamples)  # 6x faster than PyTorch
# nadd_persample = torch.LongTensor(nadd_persample_np).to(data_idx.device)
# nadd_persample_sum = nadd_persample.sum()
# to_add_flat = torch.cat([torch.multinomial(torch.ones(n * m, device=data_idx.device), num, replacement=False) for num in nadd_persample_np])

def accuracy(labels, logits, idx):
    return (labels[idx] == logits[idx].argmax(1)).sum().item() / len(idx)


def accuracy_majority(labels, votes, idx):
    return (votes.argmax(1)[idx] == labels[idx]).mean()




def sample_perturbed_mnist(data, sample_config):
    pf_minus = sample_config.get('pf_minus_att', 0)
    pf_plus = sample_config.get('pf_plus_att', 0)
    return binary_perturb(data, pf_minus, pf_plus)


def sample_one_graph(attr_idx, edge_idx, sample_config, n, d):
    """
    Perturb the structure and node attributes.

    Parameters
    ----------
    attr_idx: torch.Tensor [2, ?]
        The indices of the non-zero attributes.
    edge_idx: torch.Tensor [2, ?]
        The indices of the edges.
    sample_config: dict
        Configuration specifying the sampling probabilities
    n : int
        Number of nodes
    d : int
        Number of features

    Returns
    -------
    attr_idx: torch.Tensor [2, ?]
        The indices of the non-zero attributes after perturbation.
    edge_idx: torch.Tensor [2, ?]
        The indices of the edges after perturbation.
    """
    pf_plus_adj = sample_config.get('pf_plus_adj', 0)
    pf_plus_att = sample_config.get('pf_plus_att', 0)

    pf_minus_adj = sample_config.get('pf_minus_adj', 0)
    pf_minus_att = sample_config.get('pf_minus_att', 0)

    per_attr_idx = sparse_perturb(data_idx=attr_idx, n=n, m=d, undirected=False,
                                  pf_minus=pf_minus_att, pf_plus=pf_plus_att)

    per_edge_idx = sparse_perturb(data_idx=edge_idx, n=n, m=n, undirected=True,
                                  pf_minus=pf_minus_adj, pf_plus=pf_plus_adj)

    return per_attr_idx, per_edge_idx


def sample_batch_pyg(data, sample_config):
    """
    Perturb the structure and node attributes.

    Parameters
    ----------
    data: torch_geometric.data.Batch
        Dataset containing the attributes, edge indices, and batch-ID
    sample_config: dict
        Configuration specifying the sampling probabilities

    Returns
    -------
    per_data: torch_geometric.Dataset
        Dataset containing the perturbed graphs
    """
    pf_plus_adj = sample_config.get('pf_plus_adj', 0)
    pf_plus_att = sample_config.get('pf_plus_att', 0)

    pf_minus_adj = sample_config.get('pf_minus_adj', 0)
    pf_minus_att = sample_config.get('pf_minus_att', 0)

    per_x = binary_perturb(data.x, pf_minus_att, pf_plus_att)

    per_edge_index = sparse_perturb_adj_batch(
            data_idx=data.edge_index, nnodes=torch.bincount(data.batch),
            pf_minus=pf_minus_adj, pf_plus=pf_plus_adj,
            undirected=True)

    per_data = Batch(batch=data.batch, x=per_x, edge_index=per_edge_index)

    return per_data


def sample_multiple_graphs(attr_idx, edge_idx, sample_config, n, d, nsamples):
    """
    Perturb the structure and node attributes.

    Parameters
    ----------
    attr_idx: torch.Tensor [2, ?]
        The indices of the non-zero attributes.
    edge_idx: torch.Tensor [2, ?]
        The indices of the edges.
    sample_config: dict
        Configuration specifying the sampling probabilities
    n : int
        Number of nodes
    d : int
        Number of features
    nsamples : int
        Number of samples

    Returns
    -------
    attr_idx: torch.Tensor [2, ?]
        The indices of the non-zero attributes after perturbation.
    edge_idx: torch.Tensor [2, ?]
        The indices of the edges after perturbation.
    """
    pf_plus_adj = sample_config.get('pf_plus_adj', 0)
    pf_plus_att = sample_config.get('pf_plus_att', 0)

    pf_minus_adj = sample_config.get('pf_minus_adj', 0)
    pf_minus_att = sample_config.get('pf_minus_att', 0)

    if pf_minus_att + pf_plus_att > 0:
        per_attr_idx = sparse_perturb_multiple(data_idx=attr_idx, n=n, m=d, undirected=False,
                                               pf_minus=pf_minus_att, pf_plus=pf_plus_att,
                                               nsamples=nsamples, offset_both_idx=False)
    else:
        per_attr_idx = copy_idx(idx=attr_idx, dim_size=n, ncopies=nsamples, offset_both_idx=False)

    if pf_minus_adj + pf_plus_adj > 0:
        per_edge_idx = sparse_perturb_multiple(data_idx=edge_idx, n=n, m=n, undirected=True,
                                               pf_minus=pf_minus_adj, pf_plus=pf_plus_adj,
                                               nsamples=nsamples, offset_both_idx=True)
    else:
        per_edge_idx = copy_idx(idx=edge_idx, dim_size=n, ncopies=nsamples, offset_both_idx=True)


    return per_attr_idx, per_edge_idx


def collate(attr_idx_list: List[torch.LongTensor],
            edge_idx_list: List[torch.LongTensor], n: int, d: int):
    attr_idx = torch.cat(attr_idx_list, dim=1)
    edge_idx = torch.cat(edge_idx_list, dim=1)

    attr_lens = attr_idx.new_tensor([idx.shape[1] for idx in attr_idx_list])
    edge_lens = edge_idx.new_tensor([idx.shape[1] for idx in edge_idx_list])
    attr_idx = offset_idx(attr_idx, attr_lens, n, [0])
    edge_idx = offset_idx(edge_idx, edge_lens, n, [0, 1])

    return attr_idx, edge_idx


def copy_idx(idx: torch.LongTensor, dim_size: int, ncopies: int, offset_both_idx: bool):
    idx_copies = idx.repeat(1, ncopies)

    offset = dim_size * torch.arange(ncopies, dtype=torch.long,
                                     device=idx.device)[:, None].expand(ncopies, idx.shape[1]).flatten()

    if offset_both_idx:
        idx_copies += offset[None, :]
    else:
        idx_copies[0] += offset

    return idx_copies


def offset_idx(idx_mat: torch.LongTensor, lens: torch.LongTensor, dim_size: int, indices: List[int] = [0]):
    offset = dim_size * torch.arange(len(lens), dtype=torch.long,
                                     device=idx_mat.device).repeat_interleave(lens, dim=0)

    idx_mat[indices, :] += offset[None, :]
    return idx_mat


def get_mnist_dataloaders(batch_size, random_seed=0, num_workers=-1, pin_memory=False, root='../dataset_cache', shuffle=True):
    dataset_dev = datasets.MNIST(
        root=root, train=True, download=True, transform=transforms.ToTensor())
    dataset_test = datasets.MNIST(
        root=root, train=False, download=True, transform=transforms.ToTensor())

    x_dev_bin = (dataset_dev.data > 0.5).float()
    x_test_bin = (dataset_test.data > 0.5).float()
   
    indices = np.arange(len(dataset_dev))
    nvalid = 5000

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[nvalid:], indices[:nvalid]


    dataset_train_bin = TensorDataset(x_dev_bin[train_idx], dataset_dev.targets[train_idx])
    dataset_val_bin = TensorDataset(x_dev_bin[valid_idx], dataset_dev.targets[valid_idx])
    dataset_test_bin = TensorDataset(x_test_bin, dataset_test.targets)

    n_images = {'train': len(train_idx),
                'val': len(valid_idx),
                'test': len(dataset_test)}

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        dataset_train_bin, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory)
    dataloaders['val'] = torch.utils.data.DataLoader(
        dataset_val_bin, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory)

    dataloaders['test'] = torch.utils.data.DataLoader(
        dataset_test_bin, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return dataloaders, n_images

class MultipleOptimizer():
    """ a class that wraps multiple optimizers """
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def update_lr(self, op_index, new_lr):
        """ update the learning rate of one optimizer
        Parameters: op_index: the index of the optimizer to update
                    new_lr:   new learning rate for that optimizer """
        for param_group in self.optimizers[op_index].param_groups:
            param_group['lr'] = new_lr

def sample_positive_edges(edge_idx, sample_ratio=0.3):
    """
    Sample a percentage of positive edges from the given edge index.

    Args:
        edge_idx (torch.Tensor): Tensor containing edge indices.
        sample_ratio (float): Ratio of edges to sample (default is 0.3).

    Returns:
        torch.Tensor: Tensor containing sampled edge indices.
    """
    num_edges = edge_idx.size(1)
    num_samples = int(num_edges * sample_ratio)
    # Randomly permute the indices and select the first num_samples indices
    permuted_indices = torch.randperm(num_edges)[:num_samples]
    # Select the sampled edges
    sampled_edge_idx = edge_idx[:, permuted_indices]
    return sampled_edge_idx


def sample_negative_edges(edge_idx, num_nodes, num_samples=1000):
    """
    Sample a percentage of negative edges from the given edge index.

    Args:
        edge_idx (torch.Tensor): Tensor containing edge indices.
        num_nodes (int): Number of nodes in the graph.
        sample_ratio (float): Ratio of edges to sample (default is 0.3).

    Returns:
        torch.Tensor: Tensor containing sampled negative edge indices.
    """
    # Create a set of all possible edges
    all_possible_edges = set((i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j)
    # Convert edge_idx to a set of tuples
    positive_edges = set((edge_idx[0, i].item(), edge_idx[1, i].item()) for i in range(edge_idx.size(1)))
    # Identify negative edges
    negative_edges = list(all_possible_edges - positive_edges)
    # Randomly sample negative edges
    sampled_negative_edges = torch.tensor(negative_edges)[torch.randperm(len(negative_edges))[:num_samples]]
    return sampled_negative_edges.t()

def sample_positive_edges_test(edge_idx, pos_edge_idx, sample_ratio=0.3):
    """
    Sample a percentage of positive edges from the given edge index, excluding training edges.

    Args:
        edge_idx (torch.Tensor): Tensor containing edge indices.
        pos_edge_idx (torch.Tensor): Tensor containing training edge indices.
        sample_ratio (float): Ratio of edges to sample (default is 0.3).

    Returns:
        torch.Tensor: Tensor containing sampled edge indices.
    """
    # Convert edge_idx and pos_edge_idx to sets of tuples
    all_edges = set((edge_idx[0, i].item(), edge_idx[1, i].item()) for i in range(edge_idx.size(1)))
    training_edges = set((pos_edge_idx[0, i].item(), pos_edge_idx[1, i].item()) for i in range(pos_edge_idx.size(1)))

    # Exclude training edges
    remaining_edges = list(all_edges - training_edges)

    # Calculate number of samples
    num_samples = int(len(remaining_edges) * sample_ratio)

    # Randomly sample edges
    sampled_edges = torch.tensor(remaining_edges)[torch.randperm(len(remaining_edges))[:num_samples]]

    return sampled_edges.t()