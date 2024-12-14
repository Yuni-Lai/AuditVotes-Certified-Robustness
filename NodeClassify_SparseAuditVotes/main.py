import os
import sys
sys.path.append("..")
import numpy as np
import pprint
from tqdm import tqdm, trange
import concurrent.futures
import pandas as pd
import pickle
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import warnings
import json
from sparse_auditvotes.training import train_gnn,train_Jac_gnn,train_Aug_gnn
from sparse_auditvotes.data import load_and_standardize, split, split_inductive, filter_data_for_idx
from torch_geometric.utils import add_self_loops
from sparse_auditvotes.certify import certify
from sparse_auditvotes.predict import predict_smooth_gnn
from visualize import analyze_result, certified_curve, merged_certified_curve,pA_distribution,get_combine_df
from utils import *

warnings.filterwarnings("ignore")
pp = pprint.PrettyPrinter(depth=4)
# Model Settings=======================================
parser = argparse.ArgumentParser(description='certify SparseSmooth with adversarial filters')
parser.add_argument('-device', type=str, default='gpu', help="device type")
parser.add_argument('-gpuID', type=int, default=7)
parser.add_argument('-seed', type=int, default=2021)
parser.add_argument('-early_stopping', action='store_true', default=True)
parser.add_argument('-patience', type=int, default=100, help='patience for early stopping')
parser.add_argument('-max_epochs', type=int, default=3000, help='training epoch')
parser.add_argument('-lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('-weight_decay', type=float, default=1e-3, help='weight_decay rate')
parser.add_argument('-model', type=str, default='APPNP', choices=['GCN','GAT','APPNP'], help='GNN models')
parser.add_argument('-augmenter', type=str, default='SimAug', choices=['','JacAug','FAEAug','SimAug'], help='augmentations')
parser.add_argument('-n_hidden', type=int, default=128, help='size of hidden layer')
parser.add_argument('-p_dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('-n_per_class', type=int, default=50, help='sample numebr per class')
parser.add_argument('-batch_size_eval', type=int, default=5)  # 5,10,50,100
parser.add_argument('-batch_size_train', type=int, default=1, help="for smooth logit training")
parser.add_argument('-force_training', action='store_true', default=False,
                    help="force training even if pretrained model exist")
# filter setting---------------------
parser.add_argument('-certify_mode', type=str, default='WithDetect', choices=['Vanilla', 'WithDetect'])
parser.add_argument('-filter', type=str, default='Conf',
                    choices=['Conf','Entr','Homo','Prox1','Prox2','JSD','NSP'])
parser.add_argument('-conf_thre', type=float, default=0.6, help='1-threshold for confidence filter')
parser.add_argument('-etr_thre', type=float, default=1.2, help='threshold for entropy filter')
parser.add_argument('-homo_thre', type=float, default=0.01, help='1-threshold for homophily filter')
parser.add_argument('-prox_thre', type=float, default=0.5, help='threshold for prox 1 or 2 filter')
parser.add_argument('-jsd_thre', type=float, default=1.0, help='threshold for jsd filter')
parser.add_argument('-nsp_thre', type=float, default=0.85, help='1-threshold for nsp filter')
parser.add_argument('-batch_size_detect', type=int, default=10, help='batch size of random graph in training')
parser.add_argument('-batch_number', type=int, default=800, help='number of batchs')
# Certify setting----------------------
parser.add_argument('-pf_plus_adj', type=float, default=0.2, help='probability of adding edges')
parser.add_argument('-pf_minus_adj', type=float, default=0.6, help='probability of deleting edges')
parser.add_argument('-pf_plus_att', type=float, default=0.0, help='probability of adding attributes')
parser.add_argument('-pf_minus_att', type=float, default=0.0, help='probability of deleting attributes')
parser.add_argument('-certify_type', type=str, default='r_a', choices=['r_a', 'r_d'], help='certify delete or add manipulation')
parser.add_argument('-n_samples_pre_eval', type=int, default=1000, help='number of smoothing samples pre-evalute')
parser.add_argument('-n_samples_eval', type=int, default=10000, help='number of smoothing samples evalute (N)')
parser.add_argument('-conf_alpha', type=float, default=0.001, help='confident alpha for statistic testing')
parser.add_argument('-mean_softmax', action='store_true', default=False, help="for smooth logit training")
parser.add_argument('-pre_votes', action='store_true', default=False)
parser.add_argument('-analyze_result', action='store_true', default=True)
parser.add_argument('-force_cert', action='store_true', default=True,
                    help="force certifying even if randomized result exist")
# Dir setting--------------------------
parser.add_argument('-dataset', type=str, default='citeseer', choices=['cora_ml', 'citeseer', 'pubmed'])
parser.add_argument('-output_dir', type=str, default='')
args = parser.parse_args()
# Others-------------------------------
if torch.cuda.is_available() and args.device == 'gpu':
    args.device = torch.device(f'cuda:{args.gpuID}')
    print(f"---using GPU---cuda:{args.gpuID}----")
else:
    print("---using CPU---")
    args.device = torch.device("cpu")
print('\n')
print(f"setting random seed {args.seed}\n")
init_random_seed(int(args.seed))

if args.certify_mode == "Vanilla":
    args.filter = '' #None
# args.model=args.model+args.augmenter

if args.dataset == "cora_ml":
    args.data_dir = "../Data/cora_ml/cora_ml.npz"
    test_ratio = 0.2
elif args.dataset == "citeseer":
    args.data_dir = "../Data/citeseer/citeseer.npz"
    test_ratio = 0.2
elif args.dataset == "pubmed":
    args.data_dir = "../Data/pubmed/pubmed.npz"
    args.batch_size_detect=1
    args.batch_size_eval=1
    test_ratio = 0.6

if args.model=="GCN":
    args.batch_size_eval = 1 # avoid OOM

if args.certify_type == 'r_a':
    args.conf_thre = 0.8

args.output_dir = f'./results_{args.dataset}_{args.model}{args.augmenter}/{args.certify_mode}{args.filter}/{args.pf_plus_adj}_{args.pf_plus_att}_{args.pf_minus_adj}_{args.pf_minus_att}_{args.n_samples_eval}/'
args.model_dir = f'./results_{args.dataset}_{args.model}{args.augmenter}/{args.model}{args.augmenter}_{args.pf_plus_adj}_{args.pf_plus_att}_{args.pf_minus_adj}_{args.pf_minus_att}.pth'
pprint.pprint(vars(args), width=1)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
# =======================================================
# smoothing samples config
args.N = args.n_samples_eval
sample_config_train, sample_config_pre_eval, sample_config_eval = get_smoothing_config(args.pf_plus_adj,
                                                                                       args.pf_plus_att,
                                                                                       args.pf_minus_adj,
                                                                                       args.pf_minus_att,
                                                                                       args.mean_softmax,
                                                                                       args.n_samples_pre_eval,
                                                                                       args.n_samples_eval)

## Load load dataset------------------------
graph = load_and_standardize(args.data_dir)
# graph.has_self_loops() #True
n, d = graph.attr_matrix.shape
nc = graph.labels.max() + 1
idx = {}
edge_idx = torch.LongTensor(np.stack(graph.adj_matrix.nonzero())).to(args.device)
attr_idx = torch.LongTensor(np.stack(graph.attr_matrix.nonzero())).to(args.device)
labels = torch.LongTensor(graph.labels).to(args.device)

edge_proportion=graph.adj_matrix.sum()/n**2
## prepare inductive subgraphs------------
idx['train'], idx['unlabeled'], idx['val'], idx['test'] = split_inductive(graph.labels, n_per_class=args.n_per_class,
                                                                          balance_test=True,test_ratio=test_ratio, seed=args.seed)
# 1) when training, the subgraph is consist of labeled training nodes, and unlabeled training nodes
attr_idx_train, edge_idx_train, labels_train, mapping_proj_train = filter_data_for_idx(graph.attr_matrix, graph.adj_matrix, args.device,
                                                                    labels, np.concatenate([idx['train'], idx['unlabeled']]))
idx_train_train = mapping_proj_train[idx['train']].to(args.device)  # idx of training nodes in training graph
# 2) when validatition, the subgraph is consist of labeled training nodes, labeled validation nodes, and unlabeled training nodes
attr_idx_val, edge_idx_val, labels_val, mapping_proj_val = filter_data_for_idx(graph.attr_matrix, graph.adj_matrix, args.device,labels,
                                                            np.concatenate([idx['train'], idx['val'], idx['unlabeled']]))
idx_val_val = mapping_proj_val[idx['val']].to(args.device)  # idx of val nodes in val graph
# 3) when testing, the input is the whole graph, but test on testing nodes.

## Load models------------------------------------------
model = model_setup(d, nc, graph.attr_matrix.todense(),idx, edge_proportion,args)


## Training the model with smoothing perturbation samples
if not os.path.exists(args.model_dir) or args.force_training:
    start_time = get_time()
    if args.augmenter in ['FAEAug','SimAug']:
        best_hyperparams = train_Aug_gnn(model=model, edge_idx_train=edge_idx_train, attr_idx_train=attr_idx_train, labels_train=labels_train,
                                  n=n,d=d, nc=nc, edge_idx_val=edge_idx_val, attr_idx_val=attr_idx_val, labels_val=labels_val,
                                  idx_train=idx_train_train, idx_val=idx_val_val,edge_idx=edge_idx,attr_idx=attr_idx,idx=idx, lr=args.lr, weight_decay=args.weight_decay,
                                  patience=args.patience*2, max_epochs=args.max_epochs,
                                  pretrain_ep=250,pretrain_nc=100,#100
                                  display_step=50,sample_config=sample_config_train,
                                  batch_size=args.batch_size_train, early_stopping=args.early_stopping)
        with open(f'{args.model_dir.split(".pth")[0]}_best_hyperparams.json', 'w') as f:
            json.dump(best_hyperparams, f)
    elif args.augmenter in ['JacAug']:
        best_hyperparams = train_Aug_gnn(model=model, edge_idx_train=edge_idx_train, attr_idx_train=attr_idx_train, labels_train=labels_train,
                                  n=n,d=d, nc=nc, edge_idx_val=edge_idx_val, attr_idx_val=attr_idx_val, labels_val=labels_val,
                                  idx_train=idx_train_train, idx_val=idx_val_val,edge_idx=edge_idx,attr_idx=attr_idx,idx=idx, lr=args.lr, weight_decay=args.weight_decay,
                                  patience=args.patience*2, max_epochs=args.max_epochs,
                                  pretrain_ep=0,pretrain_nc=0,
                                  display_step=50,sample_config=sample_config_train,
                                  batch_size=args.batch_size_train, early_stopping=args.early_stopping)
        with open(f'{args.model_dir.split(".pth")[0]}_best_hyperparams.json', 'w') as f:
            json.dump(best_hyperparams, f)
    else:
        trace_val = train_gnn(model=model, edge_idx_train=edge_idx_train, attr_idx_train=attr_idx_train, labels_train=labels_train,
                              d=d, nc=nc, edge_idx_val=edge_idx_val, attr_idx_val=attr_idx_val, labels_val=labels_val,
                              idx_train=idx_train_train, idx_val=idx_val_val, lr=args.lr, weight_decay=args.weight_decay,
                              patience=args.patience, max_epochs=args.max_epochs, display_step=50,
                              sample_config=sample_config_train,
                              batch_size=args.batch_size_train, early_stopping=args.early_stopping)
    torch.save(model, args.model_dir)
    print('model saved to:', args.model_dir)
    end_time = get_time()
    print('training time:', end_time - start_time)
else:
    model = torch.load(args.model_dir,map_location=args.device)


# print(torch.cuda.memory_summary(device=args.device))

if model.__class__.__name__ in ['GAugGCN','FAugGCN','SAugGCN']:
    print('Reconstruction acc,auc on whole graph:')
    model.eval()
    model.ep_net.test_all(edge_idx, attr_idx,n,d, range(n))#idx['test']
elif model.__class__.__name__ in ['JacGCN']:
    print('Reconstruction acc,auc on whole graph:')
    model.eval()
    model.test_all(edge_idx, attr_idx, n, d,range(n))#idx['test']


if args.analyze_result == True and args.certify_mode == 'WithDetect':
    analysis_data = {'anomaly_score': [], 'anomaly_pred': [], 'class_pred': [], 'class_label': []}
else:
    analysis_data = None


# from sparse_smoothing.utils import sample_multiple_graphs
# from filter_Quality.train_test import *
#
# homophilys_ori = get_homo(edge_idx, n, 1, labels)
# attr_idx_batch, edge_idx_batch = sample_multiple_graphs(
#                     attr_idx=attr_idx, edge_idx=edge_idx,
#                     sample_config=sample_config_eval, n=n, d=d, nsamples=5)
# if args.model in ['JacGCN','FAugGCN','SAugGCN']:
#     edge_idx_batch=model.get_aug_edge_idx(edge_idx_batch, n * 5)
# homophilys = get_homo(edge_idx_batch, n, 5, labels.repeat(5))
# print(homophilys_ori.mean(), homophilys.mean())

# model = torch.nn.DataParallel(model, device_ids=[5, 6, 7])
if not os.path.exists(f'{args.output_dir}/certify_result_{args.certify_type}.pkl') or args.force_cert:
# if not os.path.exists(f'{args.output_dir}/smoothing_result.pkl') or args.force_cert:
    votes, analysis_data = predict_smooth_gnn(args=args, attr_idx=attr_idx, edge_idx=edge_idx,
                                              sample_config=sample_config_eval,
                                              model=model, n=n, d=d, nc=nc,
                                              batch_size=args.batch_size_eval,
                                              analysis_data=analysis_data)
    if analysis_data != None:
        if args.dataset == 'pubmed':
            analysis_data['class_label'].extend(graph.labels.tolist() * args.batch_size_eval)
        else:
            analysis_data['class_label'].extend(graph.labels.tolist() * args.n_samples_eval)
        try:
            analyze_result(analysis_data, args.output_dir, args)
        except:
            print('analysis_data encounter some errors')
            pass
    if args.pre_votes == True:
        pre_votes, _ = predict_smooth_gnn(args=args, attr_idx=attr_idx, edge_idx=edge_idx,
                                          sample_config=sample_config_pre_eval,
                                          model=model, n=n, d=d, nc=nc,
                                          batch_size=args.batch_size_eval,
                                          analysis_data=None)
    else:
        pre_votes = votes

    f = open(f'{args.output_dir}/smoothing_result.pkl', 'wb')
    pickle.dump([pre_votes, votes], f)
    f.close()
    print(f'Save result to {args.output_dir}/smoothing_result.pkl')
else:
    f = open(f'{args.output_dir}/smoothing_result.pkl', 'rb')
    pre_votes, votes = pickle.load(f)
    f.close()

df, results_summary = certify(pre_votes, votes, graph.labels, idx, args)
certified_curve(df, args.output_dir, args)

pp.pprint(results_summary)
print(f'{args.dataset}_{args.filter}')
print(df[df.loc[:, args.certify_type].isin([0, 3, 5, 7, 10, 20, 30])])

# table = df.pivot_table(index='parameters', columns='rho', values='certified accuracy')
# print(table.loc[:, [20, 50, 100, 120, 140]])
f = open(f'{args.output_dir}/certify_result_{args.certify_type}.pkl', 'wb')#_{args.conf_thre}
pickle.dump(df, f)
f.close()
print(f'Save result to {args.output_dir}/certify_result_{args.certify_type}.pkl')


if args.certify_mode == 'WithDetect':
    if args.augmenter=="JacAug":
        types = [args.model, args.model+'JacAug']
    elif args.augmenter=="FAEAug":
        types = [args.model, args.model+'JacAug', args.model+"FAEAug"]
    elif args.augmenter=="SimAug":
        types = [args.model, args.model+'JacAug', args.model+"FAEAug", args.model+"SimAug"]
    else:
        if args.filter in ['Conf','Entr','Homo','JSD','Prox1','Prox2','NSP']:
            types = ['Vanilla']

    df_combine, votes_vanilla=get_combine_df(args,types)
    # add the current df also
    df_combine = pd.concat([df_combine, df], ignore_index=True)
    merged_certified_curve(df_combine, args.output_dir, args)
    pA_distribution(votes_vanilla[idx['test'], :], votes[idx['test'], :], labels[idx['test']], args)

    print(f'{args.dataset}_{args.filter}')
    print(df_combine[df_combine.loc[:, args.certify_type].isin([0, 3, 5, 7, 10, 20, 30])])
    print(df_combine[df_combine.loc[:, args.certify_type].isin([5])])

elif args.augmenter!='':
    if args.augmenter == "JacAug":
        types = [args.model, args.model+'JacAug']
    elif args.augmenter == "FAEAug":
        types = [args.model, args.model+'JacAug', args.model+"FAEAug"]
    elif args.augmenter == "SimAug":
        types = [args.model, args.model+'JacAug', args.model+"FAEAug", args.model+"SimAug"]
    df_combine, _ = get_combine_df(args, types)
    merged_certified_curve(df_combine, args.output_dir, args)
    print(df_combine[df_combine.loc[:, args.certify_type].isin([5])])