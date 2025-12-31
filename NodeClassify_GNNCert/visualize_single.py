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
parser.add_argument('-gpuID', type=int, default=3)
parser.add_argument('-seed', type=int, default=2021)
parser.add_argument('-early_stopping', action='store_true', default=True)
parser.add_argument('-patience', type=int, default=200, help='patience for early stopping')
parser.add_argument('-max_epochs', type=int, default=3000, help='training epoch')
parser.add_argument('-lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('-weight_decay', type=float, default=1e-3, help='weight_decay rate')
parser.add_argument('-model', type=str, default='FAugGCN', choices=['GCN','GCNJaccard_Aug','GAugGCN','FAugGCN','SAugGCN'], help='GNN models')
parser.add_argument('-n_hidden', type=int, default=128, help='size of hidden layer')
parser.add_argument('-p_dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('-n_per_class', type=int, default=50, help='sample numebr per class')
parser.add_argument('-force_training', action='store_true', default=False,
                    help="force training even if pretrained model exist")
parser.add_argument('-batch_size_eval', type=int, default=5)  # 10,50,100
parser.add_argument('-batch_size_train', type=int, default=5)
# Detector setting---------------------
parser.add_argument('-certify_mode', type=str, default='Vanilla', choices=['Vanilla', 'WithDetect'])
parser.add_argument('-detector', type=str, default='Conf',
                    choices=['DGMM', 'BWGNN', 'RFGraph', 'RGCN', 'PCGNN', 'GATSep', 'GCN', 'SVM', 'Conf'])
parser.add_argument('-conf_thre', type=float, default=0.5, help='1-threshold for confidence filter')
parser.add_argument('-batch_size_detect', type=int, default=10, help='number of random graph at once')
parser.add_argument('-batch_number', type=int, default=300, help='number of batchs')
# Certify setting----------------------
parser.add_argument('-division-method', type=str,
                    default="structure", choices=["structure", "feature", "both"], help="division method")
parser.add_argument('-Ts', type=int, default=30, help='groups of structure division')
parser.add_argument('-Tf', type=int, default=1, help='groups of ndoe feature division')
parser.add_argument('-Td', type=int, default=1, help='number of repeat for each division')
parser.add_argument('-certify_type', type=str, default='r', choices=['r','r_a', 'r_d'], help='certify delete or add manipulation')
parser.add_argument('-hash', type=str, default='md5',
                    choices=["add", "hash", "md5", "sha256", "sha512", "sha1"], help="Hash function used for division")
parser.add_argument('-mean_softmax', action='store_true', default=False)
parser.add_argument('-analyze_result', action='store_true', default=False)
parser.add_argument('-force_cert', action='store_true', default=False,
                    help="force certifying even if randomized result exist")
# Dir setting-------------------------
parser.add_argument('-dataset', type=str, default='citeseer', choices=['cora_ml', 'citeseer', 'pubmed'])
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
elif args.dataset == "citeseer":
    args.data_dir = "../Data/citeseer/citeseer.npz"
elif args.dataset == "pubmed":
    args.data_dir = "../Data/pubmed/pubmed.npz"

args.output_dir = f'./results_{args.dataset}_{args.model}/{args.certify_mode}{args.detector}/Ts{args.Ts}_Tf{args.Tf}_Td{args.Td}/'
args.model_dir = f'./results_{args.dataset}_{args.model}/{args.model}_Ts{args.Ts}_Tf{args.Tf}_Td{args.Td}.pth'
pprint.pprint(vars(args), width=1)




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
        types = ['GCN', "FAugGCN"]#'GCNJaccard_Aug'
    elif args.model=="SAugGCN":
        types = ['GCN', 'GCNJaccard_Aug', "FAugGCN", "SAugGCN"]
    df_combine, _ = get_combine_df(args, types)
    merged_certified_curve(df_combine, args.output_dir, args)
