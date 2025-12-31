import os
import sys
sys.path.append("..")
import numpy as np
import pprint
import time
from tqdm import tqdm, trange
import concurrent.futures
import pandas as pd
import pickle
import random
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import warnings
from training import *
from utils import *
from division import hash_func_map, division_func_map, spread_function
from data import load_and_standardize, split, split_inductive, filter_data_for_idx
from certify import certify
from predict import predict_smooth_gnn
from visualize import *

warnings.filterwarnings("ignore")
pp = pprint.PrettyPrinter(depth=4)
# Model Settings=======================================
parser = argparse.ArgumentParser(description='certify GNN Finite Aggregation')
parser.add_argument('-device', type=str, default='gpu', help="device type")
parser.add_argument('-gpuID', type=int, default=9)
parser.add_argument('-seed', type=int, default=2021)
parser.add_argument('-early_stopping', action='store_true', default=True)
parser.add_argument('-patience', type=int, default=200, help='patience for early stopping')
parser.add_argument('-max_epochs', type=int, default=3000, help='training epoch')
parser.add_argument('-lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('-weight_decay', type=float, default=1e-3, help='weight_decay rate')
parser.add_argument('-model', type=str, default='SAugGCN', choices=['GCN','GCNJaccard_Aug','GAugGCN','FAugGCN','SAugGCN'], help='GNN models')
parser.add_argument('-n_hidden', type=int, default=128, help='size of hidden layer')
parser.add_argument('-p_dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('-n_per_class', type=int, default=50, help='sample numebr per class')
parser.add_argument('-force_training', action='store_true', default=True,
                    help="force training even if pretrained model exist")
parser.add_argument('-batch_size_eval', type=int, default=5)  # 10,50,100
parser.add_argument('-batch_size_train', type=int, default=5)
# Detector setting---------------------
parser.add_argument('-certify_mode', type=str, default='Vanilla', choices=['Vanilla', 'WithDetect'])
parser.add_argument('-detector', type=str, default='Conf',
                    choices=['DGMM', 'BWGNN', 'RFGraph', 'RGCN', 'PCGNN', 'GATSep', 'GCN', 'SVM', 'Conf'])
parser.add_argument('-conf_thre', type=float, default=0.8, help='1-threshold for confidence filter')
parser.add_argument('-batch_size_detect', type=int, default=10, help='number of random graph at once')
parser.add_argument('-batch_number', type=int, default=300, help='number of batchs')
# Certify setting----------------------
parser.add_argument('-division-method', type=str,
                    default="structure", choices=["structure", "feature", "both"], help="division method")
parser.add_argument('-Ts', type=int, default=20, help='groups of structure division')
parser.add_argument('-Tf', type=int, default=1, help='groups of ndoe feature division')
parser.add_argument('-Td', type=int, default=1, help='number of repeat for each division,Td=1 by default')
parser.add_argument('-certify_type', type=str, default='r', choices=['r','r_a', 'r_d'], help='certify delete or add manipulation')
parser.add_argument('-hash', type=str, default='md5',
                    choices=["add", "hash", "md5", "sha256", "sha512", "sha1"], help="Hash function used for division")
parser.add_argument('-mean_softmax', action='store_true', default=False)
parser.add_argument('-analyze_result', action='store_true', default=True)
parser.add_argument('-force_cert', action='store_true', default=True,
                    help="force certifying even if randomized result exist")
# Dir setting-------------------------
parser.add_argument('-dataset', type=str, default='cora_ml', choices=['cora_ml', 'citeseer', 'pubmed','amazon'])
parser.add_argument('-output_dir', type=str, default='')
args = parser.parse_args()
if args.certify_mode == "Vanilla":
    args.detector = ''
# Others------------------------------
if torch.cuda.is_available() and args.device == 'gpu':
    args.device = torch.device(f'cuda:{args.gpuID}')
    print(f"---using GPU---cuda:{args.gpuID}----")
else:
    print("---using CPU---")
    args.device = torch.device("cpu")
print('\n')
print(f"initailing random seed {args.seed}\n")
init_random_seed(int(args.seed))

n_subgraphs = args.Ts* args.Tf* args.Td
args.batch_size_eval=min(args.batch_size_eval,n_subgraphs)

if args.dataset == "cora_ml":
    args.data_dir = "../Data/cora_ml/cora_ml.npz"
    test_ratio = 0.2
elif args.dataset == "citeseer":
    args.data_dir = "../Data/citeseer/citeseer.npz"
    test_ratio = 0.2
elif args.dataset == "pubmed":
    args.data_dir = "../Data/pubmed/pubmed.npz"
    test_ratio = 0.6
    args.batch_size_eval = 1
    args.batch_size_train = 1

args.output_dir = f'./results_{args.dataset}_{args.model}/{args.certify_mode}{args.detector}/Ts{args.Ts}_Tf{args.Tf}_Td{args.Td}/'
args.model_dir = f'./results_{args.dataset}_{args.model}/{args.model}_Ts{args.Ts}_Tf{args.Tf}_Td{args.Td}.pth'
pprint.pprint(vars(args), width=1)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
# =======================================================
## Load load dataset-------------------
graph = load_and_standardize(args.data_dir)
edge_idx = torch.LongTensor(np.stack(graph.adj_matrix.nonzero())).to(args.device)
attr_idx = torch.LongTensor(np.stack(graph.attr_matrix.nonzero())).to(args.device)
labels = torch.LongTensor(graph.labels).to(args.device)
n, d = graph.attr_matrix.shape
nc = graph.labels.max() + 1
idx = {}
# idx['train'], idx['val'], idx['test'] = split(labels=graph.labels, n_per_class=args.n_per_class, seed=args.seed)
edge_proportion=graph.adj_matrix.sum()/n**2
## prepare inductive subgraphs------------

idx['train'], idx['unlabeled'], idx['val'], idx['test'] = split_inductive(graph.labels, n_per_class=args.n_per_class,
                                                                          balance_test=True,test_ratio=test_ratio, seed=args.seed)

#if balance_test=True, the distribution of test is different from val.
## Load models------------------------------------------
model = model_setup(d, nc, graph.attr_matrix.todense(), idx, edge_proportion,args)

## Training the model with subgraphs division--------------
division_func = division_func_map.get(args.division_method, None)
divided_edges, divided_attrs = division_func(n, attr_idx, edge_idx, args.hash,args.Ts * args.Td, args.Tf * args.Td)
spread_idx, subgraph_edges, subgraph_attrs = spread_function(divided_edges, divided_attrs, n_subgraphs, args.Td, args.division_method, args.seed)

# 1) when training, the subgraph is consist of labeled training nodes, and unlabeled training nodes
subgraph_attrs_train, subgraph_edges_train, labels_train, mapping_proj_train = filter_data_for_idx(subgraph_attrs, subgraph_edges, n,d, args.device,
                                                                    labels, np.concatenate([idx['train'], idx['unlabeled']]))
idx_train_train = mapping_proj_train[idx['train']].to(args.device)  # idx of training nodes in training graph
# 2) when validatition, the subgraph is consist of labeled training nodes, labeled validation nodes, and unlabeled training nodes
subgraph_attrs_val, subgraph_edges_val, labels_val, mapping_proj_val = filter_data_for_idx(subgraph_attrs, subgraph_edges, n,d, args.device,labels,
                                                            np.concatenate([idx['train'], idx['val'], idx['unlabeled']]))
idx_val_val = mapping_proj_val[idx['val']].to(args.device)  # idx of val nodes in val graph
# 3) when testing, the input is the whole graph, but test on testing nodes.

## Training the model with smoothing perturbation samples
if not os.path.exists(args.model_dir) or args.force_training:
    start_time = get_time()
    args.batch_size_train = min(args.batch_size_train, n_subgraphs)
    if args.model in ['GAugGCN','FAugGCN','SAugGCN']:
        best_hyperparams = train_Aug_gnn(model=model, subgraph_edges_train=subgraph_edges_train, subgraph_attrs_train=subgraph_attrs_train, labels_train=labels_train,
                                  n=n, d=d, nc=nc, subgraph_edges_val=subgraph_edges_val, subgraph_attrs_val=subgraph_attrs_val, labels_val=labels_val,
                                  idx_train=idx_train_train, idx_val=idx_val_val,attr_idx=attr_idx, edge_idx=edge_idx,idx=idx, lr=args.lr, weight_decay=args.weight_decay,
                                  patience=args.patience, max_epochs=args.max_epochs,
                                  pretrain_ep=250,pretrain_nc=0,display_step=50,
                                  batch_size=args.batch_size_train, early_stopping=args.early_stopping)
        with open(f'{args.model_dir.split(".pth")[0]}_best_hyperparams.json', 'w') as f:
            json.dump(best_hyperparams, f)
    elif args.model in ['GCNJaccard_Aug']:
        best_hyperparams = train_Jac_gnn(model=model, subgraph_edges_train=subgraph_edges_train, subgraph_attrs_train=subgraph_attrs_train, labels_train=labels_train,
                                  n=n, d=d, nc=nc, subgraph_edges_val=subgraph_edges_val, subgraph_attrs_val=subgraph_attrs_val, labels_val=labels_val,
                                  idx_train=idx_train_train, idx_val=idx_val_val,lr=args.lr, weight_decay=args.weight_decay,
                                  patience=args.patience, max_epochs=args.max_epochs,
                                  display_step=50,batch_size=args.batch_size_train, early_stopping=args.early_stopping)
        with open(f'{args.model_dir.split(".pth")[0]}_best_hyperparams.json', 'w') as f:
            json.dump(best_hyperparams, f)
    else:
        trace_val = train_gnn(model=model, subgraph_edges_train=subgraph_edges_train, subgraph_attrs_train=subgraph_attrs_train, labels_train=labels_train,
                              n=n, d=d, nc=nc, subgraph_edges_val=subgraph_edges_val, subgraph_attrs_val=subgraph_attrs_val, labels_val=labels_val,
                              idx_train=idx_train_train, idx_val=idx_val_val, lr=args.lr, weight_decay=args.weight_decay,
                              patience=args.patience, max_epochs=args.max_epochs, display_step=50,
                              batch_size=args.batch_size_train, early_stopping=args.early_stopping)
    torch.save(model, args.model_dir)
    print('model saved to:', args.model_dir)
    end_time = get_time()
    print('training time:', end_time - start_time)
else:
    model = torch.load(args.model_dir,map_location=args.device)

## Load detector-------------------------------------------
detector_savedir = set_detector_savedir(args)
if args.certify_mode == 'WithDetect' and (
        args.force_training == True or not os.path.exists(f'{detector_savedir}/{args.detector}.pth')):
    detector = train_detector_model(model, attr_idx, edge_idx, n, d, nc, idx, labels,detector_savedir, args)

if args.certify_mode == 'WithDetect':
    if args.detector not in ['Conf', 'Homo', 'Prox1', 'Prox2', 'JSD', 'NSP']:  # These filters do not contain any trained model
        detector = torch.load(f'{detector_savedir}/{args.detector}.pth')
    else:
        detector = 'Simple_Filter'  # 'filter samples with simple features', it does not need *.pth model.
else:
    detector = None

if args.analyze_result == True and args.certify_mode == 'WithDetect':
    analysis_data = {'anomaly_score': [], 'anomaly_pred': [], 'class_pred': [], 'class_label': []}
else:
    analysis_data = None

if not os.path.exists(f'{args.output_dir}/smoothing_result.pkl') or args.force_cert:
    start_time = get_time()
    votes,analysis_data = predict_smooth_gnn(args=args, subgraph_edges=subgraph_edges, subgraph_attrs=subgraph_attrs,
                                              model=model, n=n, d=d, nc=nc,
                                              batch_size=args.batch_size_eval, detector=detector,
                                              analysis_data=analysis_data)
    if analysis_data != None:
        analysis_data['class_label'].extend(graph.labels.tolist() * n_subgraphs)
        analyze_result(analysis_data, args.output_dir, args)
    end_time = get_time()
    print('testing time:', end_time - start_time)
    f = open(f'{args.output_dir}/smoothing_result.pkl', 'wb')
    pickle.dump(votes, f)
    f.close()
    print(f'Save result to {args.output_dir}/smoothing_result.pkl')
else:
    f = open(f'{args.output_dir}/smoothing_result.pkl', 'rb')
    votes = pickle.load(f)
    f.close()

df, results_summary = certify(votes, graph.labels, idx, args)
certified_curve(df, args.output_dir, args)

# table = df.pivot_table(index='parameters', columns='rho', values='certified accuracy')
# print(table.loc[:, [20, 50, 100, 120, 140]])
f = open(f'{args.output_dir}/certify_result.pkl', 'wb')
pickle.dump(df, f)
f.close()
print(f'Save result to {args.output_dir}/certify_result.pkl')


pp.pprint(results_summary)
print(f'{args.dataset}_{args.detector}')
print(df[df.loc[:, args.certify_type].isin([0, 3, 5, 7, 10, 20, 30])])

if args.certify_mode == 'WithDetect':
    if args.model=="GCNJaccard":
        types = ['GCN', 'GCNJaccard']
    elif args.model=="GCNJaccard_Aug":
        types = ['GCN', 'GCNJaccard_Aug']#'GCNJaccard',
    elif args.model=="GAugGCN":
        types = ['GCN', 'GCNJaccard_Aug','GAugGCN']#'GCNJaccard',
    elif args.model=="FAugGCN":
        types = ['GCN', 'GCNJaccard_Aug', "FAugGCN"]
    elif args.model=="SAugGCN":
        types = ['GCN', 'GCNJaccard_Aug', "FAugGCN", "SAugGCN"]
    else:
        if args.detector in ['Conf','Homo','JSD','Prox1','Prox2','NSP']:
            types = ['Vanilla']
    if len(args.detector.split('_')[-1].split('+')) > 1:#Conf+something
        types.append('WithDetectConf')

    df_combine, votes_vanilla=get_combine_df(args,types)
    # add the current df also
    df_combine = pd.concat([df_combine, df], ignore_index=True)
    merged_certified_curve(df_combine, args.output_dir, args)
    votes_vanilla=votes_vanilla.sum(0)
    votes=votes.sum(0)
    pA_distribution(votes_vanilla[idx['test'], :], votes[idx['test'], :], labels[idx['test']], args)

    print(f'{args.dataset}_{args.detector}')
    print(df_combine[df_combine.loc[:, args.certify_type].isin([0, 3, 5, 7, 10, 20, 30])])
    print(df_combine[df_combine.loc[:, args.certify_type].isin([5])])

elif args.model in ["GCNJaccard_Aug","GAugGCN","FAugGCN","SAugGCN"]:
    if args.model=="GCNJaccard_Aug":
        types = ['GCN','GCNJaccard_Aug']#'GCNJaccard'
    elif args.model=="GAugGCN":
        types = ['GCN', 'GCNJaccard_Aug', "GAugGCN"]
    elif args.model=="FAugGCN":
        types = ['GCN', 'GCNJaccard_Aug', "FAugGCN"]#
    elif args.model=="SAugGCN":
        types = ['GCN', 'GCNJaccard_Aug', "FAugGCN", "SAugGCN"]#
    df_combine, _ = get_combine_df(args, types)
    merged_certified_curve(df_combine, args.output_dir, args)
