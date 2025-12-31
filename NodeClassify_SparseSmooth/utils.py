import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import scipy.sparse as sp
from sparse_auditvotes.models import *
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

def get_time():
    torch.cuda.synchronize()
    return time.time()

def model_setup(d,nc,features,idx,edge_proportion,args):
    if args.model.lower() == 'gcn':
        nc_model = GCN(n_features=d, n_classes=nc, n_hidden=args.n_hidden, p_dropout=args.p_dropout).to(args.device)
    elif args.model.lower() == 'gat':
        nc_model = GAT(n_features=d, n_classes=nc, n_hidden=args.n_hidden // 8,
                    k_heads=8, p_dropout=args.p_dropout).to(
            args.device)  # divide the number of hidden units by the number of heads to match the overall number of paramters
    elif args.model.lower() == 'appnp':
        nc_model = APPNPNet(n_features=d, n_classes=nc, n_hidden=args.n_hidden,
                         k_hops=10, alpha=0.15, p_dropout=args.p_dropout).to(args.device)
    else:
        raise ValueError(f"Model {args.model} not implemented.")

    if args.augmenter.lower() == 'jacaug':
        ep_model = JacAug(device=args.device)
    elif args.augmenter.lower() == 'faeaug':
        ep_model = FAEAug(dim_feats=d, dim_h=args.n_hidden, dim_z=32, ae=True).to(args.device)
    elif args.augmenter.lower() == 'simaug':
        ep_model = SimAug(input_size=d, num_wegihts=5).to(args.device)

    if args.augmenter!='':
        model = AuditVotes(nc_net=nc_model,ep_net=ep_model,edge_proportion=edge_proportion, p_plus=args.pf_plus_adj, p_minus=args.pf_minus_adj)
    else:
        model = nc_model
    return model

def get_smoothing_config(pf_plus_adj,pf_plus_att,pf_minus_adj,pf_minus_att,mean_softmax,n_samples_pre_eval,n_samples_eval):
    sample_config = {
        'pf_plus_adj': pf_plus_adj,
        'pf_plus_att': pf_plus_att,
        'pf_minus_adj': pf_minus_adj,
        'pf_minus_att': pf_minus_att,
    }
    # if we need to sample at least once and at least one flip probability is non-zero

    sample_config_train = sample_config
    sample_config_train['mean_softmax'] = mean_softmax

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

