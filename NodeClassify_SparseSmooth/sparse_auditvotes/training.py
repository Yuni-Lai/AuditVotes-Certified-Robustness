import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from functools import partial
from sparse_auditvotes.utils import sample_multiple_graphs, accuracy
import time
import logging
import random
import json
from sparse_auditvotes.utils import MultipleOptimizer,sample_positive_edges,sample_negative_edges,sample_positive_edges_test
from torch_geometric.utils import add_self_loops

def get_time():
    torch.cuda.synchronize()
    return time.time()


def smooth_logits_gnn(attr_idx, edge_idx, model, sample_config, n, d, nc, batch_size=1, idx_nodes=None):
    n_samples = sample_config.get('n_samples', 1)
    mean_softmax = sample_config['mean_softmax']

    assert n_samples % batch_size == 0
    nbatches = n_samples // batch_size

    arng = torch.arange(n, dtype=torch.long,
                        device=attr_idx.device).repeat(batch_size)
    logits = torch.zeros([n, nc], dtype=torch.float, device=attr_idx.device)

    for _ in range(nbatches):
        attr_idx_batch, edge_idx_batch = sample_multiple_graphs(
            attr_idx=attr_idx, edge_idx=edge_idx,
            sample_config=sample_config, n=n, d=d, nsamples=batch_size)

        logits_batch = model(attr_idx=attr_idx_batch, edge_idx=edge_idx_batch,
                             n=batch_size * n, d=d)
        if mean_softmax:
            logits_batch = F.softmax(logits_batch, dim=1)
        logits = logits + scatter_add(logits_batch, arng, dim=0, dim_size=n)

    # divide by n_samples so we have the mean
    logits = logits / n_samples

    # go back to log space if we were averaging in probability space
    if mean_softmax:
        logits = torch.log(torch.clamp(logits, min=1e-20))

    return logits


def smooth_logits_pytorch(data, model, sample_config, sample_fn):
    n_samples = sample_config.get('n_samples', 1)
    logits = []
    for _ in range(n_samples):
        data_perturbed = sample_fn(data, sample_config)
        logits.append(model(data_perturbed))
    return torch.stack(logits).mean(0)


def train_gnn(model, edge_idx_train, attr_idx_train, labels_train,
              d, nc, edge_idx_val, attr_idx_val, labels_val,
              idx_train, idx_val, lr, weight_decay, patience, max_epochs,
              sample_config=None, display_step=50,
              batch_size=1, early_stopping=True):
    print('training GNN model------------')
    trace_val = []
    n_train=len(labels_train)
    n_val=len(labels_val)
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    if sample_config is not None:
        model_partial = partial(smooth_logits_gnn, model=model, sample_config=sample_config,
                                d=d, nc=nc, batch_size=batch_size)
    else:
        model_partial = partial(model, d=d)

    best_loss = np.inf
    last_time = get_time()

    for it in range(max_epochs):
        model.train()
        logits_train = model_partial(attr_idx=attr_idx_train, edge_idx=edge_idx_train, n=n_train)
        loss_train = F.cross_entropy(logits_train[idx_train], labels_train[idx_train])
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            logits_val = model_partial(attr_idx=attr_idx_val, edge_idx=edge_idx_val, n=n_val)
            loss_val = F.cross_entropy(logits_val[idx_val], labels_val[idx_val])
            trace_val.append(loss_val.item())

        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = it
            best_state = {key: value.cpu()
                          for key, value in model.state_dict().items()}
        else:
            if it >= best_epoch + patience and early_stopping:
                break

        if it % display_step == 0:
            acc_train = accuracy(labels_train, logits_train, idx_train)
            acc_val = accuracy(labels_val, logits_val, idx_val)

            current_time = get_time()
            # logging.info(f'Epoch {it:4}: loss_train: {loss_train.item():.5f}, loss_val: {loss_val.item():.5f} '
            #              f'acc_train: {acc_train:.5f}, acc_val: {acc_val:.5f} ({current_time - last_time:.3f}s)')
            print(f'Epoch {it:4}: loss_train: {loss_train.item():.5f}, loss_val: {loss_val.item():.5f} '
                           f'acc_train: {acc_train:.5f}, acc_val: {acc_val:.5f} ({current_time - last_time:.3f}s)')
            last_time = current_time
    
    print(f'best_epoch:{best_epoch}, acc_train:{acc_train}, acc_val:{acc_val}' )
    model.load_state_dict(best_state)
    return trace_val


#=======================newly added=========================
def pretrain_ep_net(model, edge_idx, pos_edge_idx,neg_edge_idx, attr_idx, n,d, sample_config,lr,weight_decay,n_epochs):
    """ pretrain the edge prediction network """
    print('pretraining edeg prediction model------------')
    # For testing edge sample---
    if n < 3000:
        pos_val = sample_positive_edges_test(edge_idx, pos_edge_idx, sample_ratio=1.0)
    else:
        pos_val = sample_positive_edges_test(edge_idx, pos_edge_idx, sample_ratio=0.1)
    neg_val = sample_negative_edges(edge_idx, n, num_samples=pos_val.shape[1]*10)
    best_auc = 0
    model.train()
    optimizer = torch.optim.Adam(model.ep_net.parameters(), lr=lr,weight_decay=weight_decay)#,weight_decay=weight_decay
    for epoch in range(n_epochs):
        z = model.ep_net.encoder(attr_idx, edge_idx, n, d)
        loss = model.ep_net.recon_loss(z, pos_edge_idx,n, neg_edge_idx)
        auc_val = model.ep_net.test(z, pos_val, neg_val)
        print(f'epoch: {epoch}, loss: {loss.item()}, val_auc: {auc_val}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if auc_val > best_auc:
            best_auc = auc_val
            best_epoch = epoch
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
    print(f'best_epoch:{best_epoch}, auc_val:{auc_val}')
    model.load_state_dict(best_state)

    adj_logit=model.ep_net(attr_idx, edge_idx, n, d)
    print(adj_logit)
    print('adj_logit>0.51:',(adj_logit>0.51).sum())
    # print(model.ep_net.fc_base.weight)

def pretrain_nc_net(model, edge_idx_train, attr_idx_train, edge_idx_val, attr_idx_val, d, idx_train,idx_val, labels_train,labels_val,sample_config, lr,weight_decay,n_epochs):
    """ pretrain the edge prediction network """
    print('pretraining node classification model------------')
    best_loss = np.inf
    n_train=len(labels_train)
    n_val=len(labels_val)
    model.train()
    optimizer = torch.optim.Adam(model.nc_net.parameters(),lr=lr)
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        logits_train = model.nc_net(attr_idx=attr_idx_train, edge_idx=edge_idx_train, n=n_train, d=d)
        loss_train = F.cross_entropy(logits_train[idx_train], labels_train[idx_train])
        loss_train.backward()
        optimizer.step()
        with torch.no_grad():
            logits_val = model.nc_net(attr_idx=attr_idx_val, edge_idx=edge_idx_val, n=n_val, d=d)
            loss_val = F.cross_entropy(logits_val[idx_val], labels_val[idx_val])
        print(f'epoch:{epoch}, train-loss:{loss_train.item()}, val-loss:{loss_val.item()}')
        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = epoch
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
    print(f'best_epoch:{best_epoch},best loss:{best_loss}')
    return best_state


def train_Jac_gnn(model, edge_idx_train, attr_idx_train, labels_train,
                  d, nc, edge_idx_val, attr_idx_val, labels_val,
                  idx_train, idx_val, lr, weight_decay,
                  pretrain_nc,patience, max_epochs,
                  display_step=50, sample_config=None, batch_size=1, early_stopping=True):
    print('training Jaccard GNN model------------')

    trace_val = []
    best_hyperparams = None
    best_performance = 0.0
    n_train = len(labels_train)
    n_val = len(labels_val)

    if pretrain_nc:  # pretrain GCN
        pretrained_state = pretrain_nc_net(model, edge_idx_train, attr_idx_train, edge_idx_val, attr_idx_val, d,
                                           idx_train, idx_val, labels_train, labels_val, sample_config, lr,
                                           weight_decay, pretrain_nc)
        model.load_state_dict(pretrained_state)

    # set adaptive augmentation thresholds---
    if n_train < 3000:
        attr_idx_batch, edge_idx_batch = sample_multiple_graphs(
            attr_idx=attr_idx_train, edge_idx=edge_idx_train, sample_config=sample_config, n=n_train, d=d,
            nsamples=5)
        model.set_daptive_thresholds(attr_idx_batch, edge_idx_batch, n_train, 5, d)
    else:
        attr_idx_batch, edge_idx_batch = sample_multiple_graphs(
            attr_idx=attr_idx_train, edge_idx=edge_idx_train, sample_config=sample_config, n=n_train, d=d,
            nsamples=1)
        model.set_daptive_thresholds(attr_idx_batch, edge_idx_batch, n_train, 1, d)

    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    if sample_config is not None:
        model_partial = partial(smooth_logits_gnn, model=model, sample_config=sample_config,
                                d=d, nc=nc, batch_size=batch_size)
    else:
        model_partial = partial(model, d=d)

    best_val = 0
    best_loss = np.inf
    last_time = get_time()

    for it in range(max_epochs):
        model.train()
        model.subgraph_mode(mode='train')
        logits_train = model_partial(attr_idx=attr_idx_train, edge_idx=edge_idx_train, n=n_train)
        loss_train = F.cross_entropy(logits_train[idx_train], labels_train[idx_train])
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            model.subgraph_mode(mode='val')
            logits_val = model_partial(attr_idx=attr_idx_val, edge_idx=edge_idx_val, n=n_val)
            loss_val = F.cross_entropy(logits_val[idx_val], labels_val[idx_val])
            trace_val.append(loss_val.item())

        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = it
            best_state = {key: value.cpu()
                          for key, value in model.state_dict().items()}
        else:
            if it >= best_epoch + patience and early_stopping:
                break

        if it % display_step == 0:
            acc_train = accuracy(labels_train, logits_train, idx_train)
            acc_val = accuracy(labels_val, logits_val, idx_val)
            current_time = get_time()
            print(f'Epoch {it:4}: loss_train: {loss_train.item():.5f}, loss_val: {loss_val.item():.5f} '
                  f'acc_train: {acc_train:.5f}, acc_val: {acc_val:.5f} ({current_time - last_time:.3f}s)')
            last_time = current_time
            if best_val < acc_val:
                best_val = acc_val

    print(f'best_epoch:{best_epoch}, best_val_acc:{best_val}')
    performance = best_val
    if performance > best_performance:
        best_performance = performance
        best_hyperparams = {
            'threshold1': model.threshold1,
            'threshold2': model.threshold2,
            'best_performance': best_performance}

    print(best_hyperparams)
    model.load_state_dict(best_state)
    model.subgraph_mode(mode='test')
    return trace_val

def train_Aug_gnn(model, edge_idx_train, attr_idx_train, labels_train,
              n,d, nc, edge_idx_val, attr_idx_val, labels_val,
              idx_train, idx_val, edge_idx, attr_idx, idx, lr, weight_decay, patience, max_epochs,
              pretrain_ep=30,pretrain_nc=30,
              sample_config=None, display_step=50,
              batch_size=1, early_stopping=True):
    best_hyperparams = None
    best_performance = 0.0

    trace_val = []
    model.train()
    n_train = torch.tensor(len(labels_train))
    n_val = torch.tensor(len(labels_val))

    # sample postive edges
    if n_train < 3000:
        pos_edge_idx = sample_positive_edges(edge_idx_train, sample_ratio = 0.9)
    else:
        pos_edge_idx = sample_positive_edges(edge_idx_train, sample_ratio = 0.1)
    # neg_edge_idx = sample_negative_edges(edge_idx_train, n_train, num_samples=pos_edge_idx.shape[1]*10)
    neg_edge_idx = None# it will be sampled randomly in the loss function
    if pretrain_ep:# pretrain augmentor
        pretrain_ep_net(model, edge_idx_train, pos_edge_idx, neg_edge_idx, attr_idx_train, n_train, d,sample_config,lr,weight_decay, pretrain_ep)
        print('reconstruction acc,auc, best_threshold on validation set:')
        model.ep_net.test_all(edge_idx_val, attr_idx_val, n_val, d, idx_val)
    if pretrain_nc:# pretrain GCN with nonsmooth graph
        pretrained_state=pretrain_nc_net(model, edge_idx_train, attr_idx_train, edge_idx_val, attr_idx_val, d, idx_train, idx_val, labels_train,labels_val ,sample_config,lr,weight_decay, pretrain_nc)
        model.load_state_dict(pretrained_state)

    model.set_adj_logit(attr_idx, edge_idx, n, d, idx)
    optimizer = torch.optim.Adam(model.nc_net.parameters(), lr=lr, weight_decay=weight_decay)

    # set adaptive augmentation thresholds---
    if n_train<3000:
        attr_idx_batch, edge_idx_batch = sample_multiple_graphs(
            attr_idx=attr_idx_train, edge_idx=edge_idx_train, sample_config=sample_config, n=n_train, d=d,
            nsamples=5)
        model.set_daptive_thresholds(attr_idx_batch, edge_idx_batch,n_train,5, d)
    else:
        attr_idx_batch, edge_idx_batch = sample_multiple_graphs(
            attr_idx=attr_idx_train, edge_idx=edge_idx_train, sample_config=sample_config, n=n_train, d=d,
            nsamples=1)
        model.set_daptive_thresholds(attr_idx_batch, edge_idx_batch,n_train,1, d)

    best_val = 0
    best_loss = np.inf
    last_time = get_time()
    for it in range(max_epochs):
        model.train()
        model.subgraph_mode(mode='train')
        optimizer.zero_grad()
        attr_idx_batch, edge_idx_batch = sample_multiple_graphs(
            attr_idx=attr_idx_train, edge_idx=edge_idx_train, sample_config=sample_config, n=n_train, d=d,
            nsamples=1)
        logits_train = model.forward(attr_idx=attr_idx_batch, edge_idx=edge_idx_batch, n=n_train, d=d)
        loss_nc = F.cross_entropy(logits_train[idx_train], labels_train[idx_train])
        loss_train = loss_nc
        loss_train.backward()
        optimizer.step()
        with torch.no_grad():
            model.eval()
            model.subgraph_mode(mode='val')
            attr_idx_batch, edge_idx_batch = sample_multiple_graphs(
                attr_idx=attr_idx_val, edge_idx=edge_idx_val, sample_config=sample_config, n=n_val, d=d,
                nsamples=1)
            logits_val = model.forward(attr_idx=attr_idx_batch, edge_idx=edge_idx_batch, n=n_val, d=d)
            loss_val = F.cross_entropy(logits_val[idx_val], labels_val[idx_val])
            trace_val.append(loss_val.item())

        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = it
            best_state = {key: value.cpu()
                          for key, value in model.state_dict().items()}
        else:
            if it >= best_epoch + patience and early_stopping:
                break
        if it % display_step == 0:
            acc_train = accuracy(labels_train, logits_train, idx_train)
            acc_val = accuracy(labels_val, logits_val, idx_val)
            current_time = get_time()
            print(f'Epoch {it:4}: loss_train: {loss_train.item():.5f}, loss_val: {loss_val.item():.5f} '
                  f'acc_train: {acc_train:.5f}, acc_val: {acc_val:.5f} ({current_time - last_time:.3f}s)')
            last_time = current_time
            if best_val < acc_val:
                best_val = acc_val

    print(f'best_epoch:{best_epoch}, best_val_acc:{best_val}')
    model.load_state_dict(best_state)
    performance = best_val
    if performance > best_performance:
        best_performance = performance
        best_hyperparams = {
            'threshold1': model.threshold1,
            'threshold2': model.threshold2,
            'best_performance':best_performance}
    print(best_hyperparams)
    model.subgraph_mode(mode='test')
    return best_hyperparams

# adj_orig = torch.sparse.FloatTensor(edge_idx_train,
#                                     torch.ones(edge_idx_train.size(1), device=edge_idx_train.device),
#                                     torch.Size([n_train, n_train])).to_dense()
# adj_orig.sum()
# adj_sample = torch.sparse.FloatTensor(edge_idx_batch, torch.ones(edge_idx_batch.size(1), device=edge_idx_batch.device),
#                                     torch.Size([n_train, n_train])).to_dense()
# adj_sample.sum()
# adj_orig.sum()+(n_train**2-adj_orig.sum())*0.1

