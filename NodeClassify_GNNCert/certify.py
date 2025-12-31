import numpy as np
import pandas as pd
from utils import accuracy_majority
import torch
import random

model_map={'GCN':'GCN','GAT':'GAT','GCNJaccard_Aug':'GCN+JacAug','GAugGCN':'GCN+GAE','FAugGCN':'GCN+FAEAug','SAugGCN':'GCN+SimAug'}


def certify(votes,labels,idx,args):
    acc_majority = {}
    for split_name in ['train', 'val', 'test']:
        acc_majority[split_name] = accuracy_majority(
            votes=votes.sum(0), labels=labels, idx=idx[split_name])

    #only count the test set---------
    votes=votes[:,idx['test'],:]
    labels=labels[idx['test']]
    pred = votes.sum(0).argmax(axis=-1)
    n_sample=len(idx['test'])
    n_subsets=votes.shape[0]

    random.seed(args.seed)
    R = random.sample(range(n_subsets), args.Td)#shifts

    cert_accs=certify_FA(n_sample, n_subsets, R, args.Td, votes.transpose((1,0,2)),labels)

    result_dict = {'r': range(30), 'certified accuracy': cert_accs.tolist()}

    df = pd.DataFrame(result_dict)
    results_summary = {
        'acc_majority_train': acc_majority['train'],
        'acc_majority_val': acc_majority['val'],
        'acc_majority_test': acc_majority['test'],
    }
    df['certify_mode'] = f'{model_map[args.model]}'
    if args.detector != '':
        df['certify_mode'] = df['certify_mode'] + f'+{args.detector}'
    df['Td'] = args.Td
    return df, results_summary


def certify_FA(n_sample,n_subsets,shifts, d, votes,labels):
    num_classes = votes.shape[2]
    valid_vote_mask = torch.tensor((votes.sum(2)>0).astype('int'))
    valid_subsets = valid_vote_mask.sum(1)
    max_classes = torch.LongTensor(np.argmax(votes, axis=2))
    predictions = torch.zeros(max_classes.shape[0], num_classes)
    for i in range(max_classes.shape[1]):#[valid_vote_mask[:,i]]
        predictions[torch.arange(max_classes.shape[0]), max_classes[:, i]] += 1 * valid_vote_mask[:,i]
    predinctionsnp = predictions.numpy()
    idxsort = np.argsort(-predinctionsnp, axis=1, kind='stable')
    valsort = -np.sort(-predinctionsnp, axis=1, kind='stable')
    val = valsort[:, 0]
    idx = idxsort[:, 0]
    valsecond = valsort[:, 1]
    idxsecond = idxsort[:, 1]

    certs = torch.LongTensor(n_sample)

    # prepared for indexing
    shifted = [
        [(h + shift) % n_subsets for shift in shifts] for h in range(n_subsets)
    ]
    shifted = torch.LongTensor(shifted)

    invalid_vote_mask = torch.tensor((votes.sum(2) == 0))  # .astype('int'))
    for i in range(n_sample):
        if idx[i] != labels[i]:
            certs[i] = -1
            continue

        certs[i] = valid_subsets[i]
        label = int(labels[i])

        # max_classes corresponding to diff h
        mask = valid_vote_mask[i][shifted.view(-1)].view(-1, d) #Abstain votes
        max_classes_given_j = max_classes[i][shifted.view(-1)].view(-1, d)
        for c in range(num_classes):  # compute min radius respect to all classes
            if c != label:
                diff = predictions[i][labels[i]] - predictions[i][c] - (1 if c < label else 0)
                # if predictions[i].sum()<180:
                #     print(predictions[i].sum())
                deltas = (mask*(1 + (max_classes_given_j == label).long() - (max_classes_given_j == c).long())).sum(dim=1)
                deltas = deltas.sort(descending=True)[0]

                radius = 0
                while diff - deltas[radius] >= 0:
                    diff -= deltas[radius].item()
                    radius += 1
                certs[i] = min(certs[i], radius)

    cert_accs = np.array([(i <= certs).sum() for i in np.arange(30)]) / n_sample
    print('Smoothed accuracy: ' + str(cert_accs[0] * 100.) + '%')
    return cert_accs

