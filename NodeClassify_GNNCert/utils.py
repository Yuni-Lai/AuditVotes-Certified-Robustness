import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import scipy.sparse as sp
from sparsegraph import SparseGraph
from models import *
from torch_geometric.utils import add_remaining_self_loops,to_dense_adj
import pandas as pd
import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt
import seaborn as sns
rc('text', usetex=True)
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
plt.subplots_adjust(left=0, right=0.1, top=0.1, bottom=0)
plt.style.use('classic')
MEDIUM_SIZE = 25
BIGGER_SIZE = 27
plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
plt.rcParams['legend.title_fontsize'] = BIGGER_SIZE

def init_random_seed(SEED=2021):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    os.environ['PYTHONHASHSEED'] = str(SEED)
init_random_seed()


def model_setup(d,nc,features,idx,edge_proportion,args):
    if args.model.lower() == 'gcn':
        model = GCN(n_features=d, n_classes=nc, n_hidden=args.n_hidden, p_dropout=args.p_dropout).to(args.device)
    elif args.model.lower() == 'gat':
        model = GAT(n_features=d, n_classes=nc, n_hidden=args.n_hidden // 8,
                    k_heads=8, p_dropout=args.p_dropout).to(
            args.device)  # divide the number of hidden units by the number of heads to match the overall number of paramters
    elif args.model.lower() == 'appnp':
        model = APPNPNet(n_features=d, n_classes=nc, n_hidden=args.n_hidden,
                         k_hops=10, alpha=0.15, p_dropout=args.p_dropout).to(args.device)
    elif args.model.lower() == 'gcnjaccard_aug':
        model = GCNJaccard_Aug(n_features=d, n_classes=nc, n_hidden=args.n_hidden, edge_proportion=edge_proportion, Ts=args.Ts, p_dropout=args.p_dropout, device=args.device).to(args.device)
        model.set_jaccard_matrix(features,idx.copy(),args.device)
    elif args.model.lower() == 'gauggcn':
        model = GAugGCN(n_features=d, n_classes=nc, n_hidden=args.n_hidden, dim_z=32, p_dropout=args.p_dropout).to(args.device)
    elif args.model.lower() == 'fauggcn':
        model = FAugGCN(n_features=d, n_classes=nc, n_hidden=args.n_hidden, dim_z=32, edge_proportion=edge_proportion, Ts=args.Ts, p_dropout=args.p_dropout).to(args.device)
    elif args.model.lower() == 'sauggcn':
        model = SAugGCN(n_features=d, n_classes=nc, n_hidden=args.n_hidden, dim_z=32, num_wegihts=5,
                        edge_proportion=edge_proportion, Ts=args.Ts,
                        p_dropout=args.p_dropout).to(args.device)
    else:
        raise ValueError(f"Model {args.model} not implemented.")
    return model

def set_detector_savedir(args):
    if args.detector=='DGMM':
        detector_savedir = f'./Detector_DGMM/{args.dataset}/Ts{args.Ts}_Tf{args.Tf}_Td{args.Td}/'
    else:
        detector_savedir = f'./Detector_Quality/{args.dataset}/Ts{args.Ts}_Tf{args.Tf}_Td{args.Td}/'
    if not os.path.exists(detector_savedir):
        os.makedirs(detector_savedir)
    return detector_savedir

def train_detector_model(model, attr_idx, edge_idx, n, d, nc, idx,labels,detector_savedir,args):
    if args.detector == 'DGMM':
        detector = train_DGMM_detector(model, attr_idx, edge_idx, n, d, nc, args.device,
                                       f'{detector_savedir}/{args.detector}.pth')
    elif args.detector == 'Conf':
        detector = 'Conf_Filter'  # 'filter samples with prediction confidence'
        #test_conf_AUC(model, n, d, attr_idx, edge_idx, labels, idx, args)
    elif args.detector == 'Homo':
        detector = 'Homo_Filter'  # 'filter samples with prediction homophily'
    elif args.detector in  ['Prox1','Prox2']:
        detector = 'Prox_Filter'  # 'filter samples with Prox'
    elif args.detector in ['JSD']:
        detector = 'JSD_Filter'  # 'filter samples with JSDivergence'
    elif args.detector in ['NSP']:
        detector = 'NSP_Filter'  # 'filter samples with Neighbor similarity'
    else:
        if len(args.detector.split('_'))>1 and args.detector.split('_')[-1] == 'Conf':
            # the detector input will graph+confidence of each nodes.
            detector = train_Conf_detector(model, n, d, attr_idx, edge_idx, labels, idx,
                                           args, f'{detector_savedir}/{args.detector}.pth')
        elif len(args.detector.split('+'))>1:
            # the detector input will graph+features of each nodes.
            detector = train_Combine_detector(model, n, d, attr_idx, edge_idx, labels, idx,
                                           args, f'{detector_savedir}/{args.detector}.pth')
        else:
            detector = train_GAD_detector(model, n, d, attr_idx, edge_idx, labels, idx,
                                              args, f'{detector_savedir}/{args.detector}.pth')
    return detector

def get_smoothing_config(pf_plus_adj,pf_plus_att,pf_minus_adj,pf_minus_att,mean_softmax,n_samples_pre_eval,n_samples_eval):
    sample_config = {
        'pf_plus_adj': pf_plus_adj,
        'pf_plus_att': pf_plus_att,
        'pf_minus_adj': pf_minus_adj,
        'pf_minus_att': pf_minus_att,
    }
    # if we need to sample at least once and at least one flip probability is non-zero
    if (pf_plus_adj + pf_plus_att + pf_minus_adj + pf_minus_att > 0):
        sample_config_train = sample_config
        sample_config_train['mean_softmax'] = mean_softmax
    else:
        sample_config_train = None

    sample_config_eval = sample_config.copy()
    sample_config_eval['n_samples'] = n_samples_eval
    sample_config_pre_eval = sample_config.copy()
    sample_config_pre_eval['n_samples'] = n_samples_pre_eval
    return sample_config_train,sample_config_pre_eval,sample_config_eval



def load_data(path):
    graph = np.load(path)
    A = sp.csr_matrix((np.ones(graph['A'].shape[1]).astype(int), graph['A']))
    data = (np.ones(graph['X'].shape[1]), graph['X'])
    X = sp.csr_matrix(data, dtype=np.float32).todense()
    y = graph['y']
    n, d = X.shape
    nc = y.max() + 1
    return A, X, y, n, d, nc


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



def get_degrees(edge_index):
    adj_dense = torch.squeeze(to_dense_adj(edge_index))
    adj_dense.fill_diagonal_(0)
    (adj_dense==adj_dense.T).all()
    degrees = adj_dense.sum(0).cpu().numpy().astype(np.int16)
    return degrees

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

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
        np.random.seed(seed+l)
        perm = np.random.RandomState(seed=seed).permutation((labels == l).nonzero()[0])
        split_train.append(perm[:n_per_class])
        split_val.append(perm[n_per_class:2 * n_per_class])


    split_train = np.random.RandomState(seed=seed).permutation(np.concatenate(split_train))
    split_val = np.random.RandomState(seed=seed).permutation(np.concatenate(split_val))

    assert split_train.shape[0] == split_val.shape[0] == n_per_class * nc

    split_test = np.setdiff1d(np.arange(len(labels)), np.concatenate((split_train, split_val)))
    print("Number of samples per class:", n_per_class)
    print("Training-validation-testing Size:", len(split_train),len(split_val),len(split_test))
    return split_train, split_val, split_test

def normalize(adj):
    degree = torch.sum(adj,dim=0)
    D_half_norm = torch.pow(degree, -0.5)
    D_half_norm = torch.nan_to_num(D_half_norm, nan=0.0, posinf=0.0, neginf=0.0)
    D_half_norm = torch.diag(D_half_norm)
    DAD = torch.mm(torch.mm(D_half_norm,adj), D_half_norm)
    return DAD

def count_arr(predictions, nclass):
    nodes_n=predictions.shape[0]
    counts = np.zeros((nodes_n,nclass), dtype=int)
    for n,idx in enumerate(predictions):
        counts[n,idx] += 1
    return counts

def listSubset(A_list,index_list):
    '''take out the elements of a list (A_list) by a index list (index_list)'''
    return [A_list[i] for i in index_list]

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def Conf_Filter(embeddings,n,batch_size):
    probs_vectors = F.log_softmax(embeddings, dim=1)
    probs = torch.max(probs_vectors,dim=1).values
    probs_vector = -probs
    probs=probs_vector.reshape(batch_size,n)
    thresh = torch.quantile(probs, 0.8, dim=0)#np.quantile(probs, 0.8)
    abnormal_node_idx = torch.where(probs >= thresh)
    anom_preds = torch.zeros((batch_size,n))
    anom_preds[abnormal_node_idx] = 1
    return anom_preds.view(-1).numpy(), probs_vector.cpu().numpy()

def accuracy(labels, logits, idx):
    return (labels[idx] == logits[idx].argmax(1)).sum().item() / len(idx)

def accuracy_majority(labels, votes, idx):
    return (votes.argmax(1)[idx] == labels[idx]).mean()

def get_batch_idx(batch_subgraph_edges,batch_subgraph_attrs,idx_subgraph,labels,n,batch_size):
    edge_idxs=[]
    for i,edge in enumerate(batch_subgraph_edges):
        edge_idxs.append(edge+i*n)
    edge_idx_batch=torch.cat(edge_idxs,dim=1)

    attr_idxs = []
    for i,attr in enumerate(batch_subgraph_attrs):
        attr_idxs.append(attr+torch.tensor([i*n,0]).unsqueeze(1).to(attr.device))
    attr_idx_batch = torch.cat(attr_idxs, dim=1)

    idx_subgraph_batch = torch.hstack([idx_subgraph + i for i in range(batch_size)])
    labels_batch = labels.repeat(batch_size)
    return edge_idx_batch,attr_idx_batch,idx_subgraph_batch,labels_batch

def get_batch_idx2(batch_subgraph_edges,batch_subgraph_attrs,n):
    edge_idxs=[]
    for i,edge in enumerate(batch_subgraph_edges):
        edge_idxs.append(edge+i*n)
    edge_idx_batch=torch.cat(edge_idxs,dim=1)

    attr_idxs = []
    for i,attr in enumerate(batch_subgraph_attrs):
        attr_idxs.append(attr+torch.tensor([i*n,0]).unsqueeze(1).to(attr.device))
    attr_idx_batch = torch.cat(attr_idxs, dim=1)

    return edge_idx_batch,attr_idx_batch


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