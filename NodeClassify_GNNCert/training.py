import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from functools import partial
from utils import accuracy
import time
import logging
from utils import *

def get_time():
    torch.cuda.synchronize()
    return time.time()

def train_gnn(model, subgraph_edges_train, subgraph_attrs_train,labels_train, n, d, nc,
              subgraph_edges_val, subgraph_attrs_val,labels_val, idx_train, idx_val, lr, weight_decay, patience, max_epochs,
              display_step=50, batch_size=10, early_stopping=True):
    '''
    model : torch.nn.Module. The GNN model.
    subgraph_attrs_train: list of torch.Tensor [2, ?]
        The indices of the non-zero attributes.
    subgraph_edges_train: list of torch.Tensor [2, ?]
        The indices of the edges.
    n : int. Number of nodes
    d : int. Dimension of features
    nc : int. Number of classes
    '''
    n_train = len(labels_train)
    n_val = len(labels_val)
    n_subgraphs=len(subgraph_edges_train)
    assert n_subgraphs % batch_size == 0
    nbatches = n_subgraphs // batch_size

    print(f'training GNN model with {n_subgraphs} division graphs ------------')
    trace_val = []
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = 0
    best_loss = 10000
    last_time = get_time()
    STOP = False
    for it in range(max_epochs):
        for batch_id in range(nbatches):
            batch_subgraph_edges = subgraph_edges_train[batch_id*batch_size:(batch_id+1)*batch_size]
            batch_subgraph_attrs = subgraph_attrs_train[batch_id*batch_size:(batch_id+1)*batch_size]
            edge_idx_batch,attr_idx_batch,idx_train_batch,labels_train_batch=get_batch_idx(batch_subgraph_edges,batch_subgraph_attrs,
                                                                                     idx_train,labels_train,n_train,batch_size)
            logits_train = model(attr_idx=attr_idx_batch, edge_idx=edge_idx_batch,
                                 n=batch_size * n_train, d=d)

            loss_train = F.cross_entropy(logits_train[idx_train_batch], labels_train_batch[idx_train_batch])
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            with torch.no_grad():
                batch_subgraph_edges = subgraph_edges_val[batch_id * batch_size:(batch_id + 1) * batch_size]
                batch_subgraph_attrs = subgraph_attrs_val[batch_id * batch_size:(batch_id + 1) * batch_size]
                edge_idx_batch, attr_idx_batch, idx_val_batch, labels_val_batch = get_batch_idx(
                    batch_subgraph_edges, batch_subgraph_attrs,
                    idx_val, labels_val, n_val, batch_size)
                model.eval()
                logits_val = model(attr_idx=attr_idx_batch, edge_idx=edge_idx_batch, n=batch_size * n_val, d=d)
                loss_val = F.cross_entropy(logits_val[idx_val_batch], labels_val_batch[idx_val_batch])
                trace_val.append(loss_val.item())

            if loss_val < best_loss:
                best_loss = loss_val
                best_epoch = (nbatches*it+batch_id)
                best_state = {key: value.cpu()
                              for key, value in model.state_dict().items()}
            else:
                if (nbatches*it+batch_id) >= best_epoch + patience and early_stopping:
                    STOP = True
                    break

            if (nbatches*it+batch_id) % display_step == 0:
                acc_train = accuracy(labels_train_batch, logits_train, idx_train_batch)
                acc_val = accuracy(labels_val_batch, logits_val, idx_val_batch)
                current_time = get_time()
                print(f'Epoch {(nbatches*it+batch_id):4}: loss_train: {loss_train.item():.5f}, loss_val: {loss_val.item():.5f} '
                      f'acc_train: {acc_train:.5f}, acc_val: {acc_val:.5f} ({current_time - last_time:.3f}s)')
                last_time = current_time
                if best_val < acc_val:
                    best_val = acc_val
        if STOP:
            break

    print(f'best_epoch:{best_epoch}, best_val_acc:{best_val}')
    model.load_state_dict(best_state)
    return trace_val

def train_Jac_gnn(model, subgraph_edges_train, subgraph_attrs_train,labels_train, n, d, nc,
              subgraph_edges_val, subgraph_attrs_val,labels_val, idx_train, idx_val, lr, weight_decay, patience, max_epochs,
              display_step=50, batch_size=10, early_stopping=True):
    '''
    model : torch.nn.Module. The GNN model.
    subgraph_attrs_train: list of torch.Tensor [2, ?]
        The indices of the non-zero attributes.
    subgraph_edges_train: list of torch.Tensor [2, ?]
        The indices of the edges.
    n : int. Number of nodes
    d : int. Dimension of features
    nc : int. Number of classes
    '''
    n_train = len(labels_train)
    n_val = len(labels_val)
    n_subgraphs=len(subgraph_edges_train)
    assert n_subgraphs % batch_size == 0
    nbatches = n_subgraphs // batch_size

    print(f'training GNN model with {n_subgraphs} division graphs ------------')
    trace_val = []
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = 0
    best_loss = 10000
    last_time = get_time()
    model.set_jaccard_mask()
    STOP = False
    for it in range(max_epochs):
        for batch_id in range(nbatches):
            model.train()
            model.subgraph_mode(mode='train')
            batch_subgraph_edges = subgraph_edges_train[batch_id*batch_size:(batch_id+1)*batch_size]
            batch_subgraph_attrs = subgraph_attrs_train[batch_id*batch_size:(batch_id+1)*batch_size]
            edge_idx_batch,attr_idx_batch,idx_train_batch,labels_train_batch=get_batch_idx(batch_subgraph_edges,batch_subgraph_attrs,
                                                                                     idx_train,labels_train,n_train,batch_size)

            logits_train = model(attr_idx=attr_idx_batch, edge_idx=edge_idx_batch,
                                 n=batch_size * n_train, d=d)

            loss_train = F.cross_entropy(logits_train[idx_train_batch], labels_train_batch[idx_train_batch])
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            with torch.no_grad():
                batch_subgraph_edges = subgraph_edges_val[batch_id * batch_size:(batch_id + 1) * batch_size]
                batch_subgraph_attrs = subgraph_attrs_val[batch_id * batch_size:(batch_id + 1) * batch_size]
                edge_idx_batch, attr_idx_batch, idx_val_batch, labels_val_batch = get_batch_idx(
                    batch_subgraph_edges, batch_subgraph_attrs,
                    idx_val, labels_val, n_val, batch_size)
                model.eval()
                model.subgraph_mode(mode='val')
                logits_val = model(attr_idx=attr_idx_batch, edge_idx=edge_idx_batch, n=batch_size * n_val, d=d)
                loss_val = F.cross_entropy(logits_val[idx_val_batch], labels_val_batch[idx_val_batch])
                trace_val.append(loss_val.item())

            if loss_val < best_loss:
                best_loss = loss_val
                best_epoch = (nbatches*it+batch_id)
                best_state = {key: value.cpu()
                              for key, value in model.state_dict().items()}
            else:
                if (nbatches*it+batch_id) >= best_epoch + patience and early_stopping:
                    STOP=True
                    break

            if (nbatches*it+batch_id) % display_step == 0:
                acc_train = accuracy(labels_train_batch, logits_train, idx_train_batch)
                acc_val = accuracy(labels_val_batch, logits_val, idx_val_batch)
                current_time = get_time()
                print(f'Epoch {(nbatches*it+batch_id):4}: loss_train: {loss_train.item():.5f}, loss_val: {loss_val.item():.5f} '
                      f'acc_train: {acc_train:.5f}, acc_val: {acc_val:.5f} ({current_time - last_time:.3f}s)')
                last_time = current_time
                if best_val < acc_val:
                    best_val = acc_val
        if STOP:
            break

    print(f'best_epoch:{best_epoch}, best_val_acc:{best_val}')
    model.load_state_dict(best_state)
    model.subgraph_mode(mode='test')
    return trace_val

def pretrain_ep_net(model, edge_idx, pos_edge_idx,neg_edge_idx, attr_idx, n,d,lr,weight_decay,n_epochs):
    """ pretrain the edge prediction network """
    print('pretraining edeg prediction model------------')
    # For testing edge sample---
    pos_test = sample_positive_edges_test(edge_idx, pos_edge_idx, sample_ratio=1.0)
    neg_test = sample_negative_edges(edge_idx, n, num_samples=pos_test.shape[1]*10)
    best_auc = 0
    model.train()
    optimizer = torch.optim.Adam(model.ep_net.parameters(), lr=lr,weight_decay=weight_decay)#,weight_decay=weight_decay
    for epoch in range(n_epochs):
        z = model.ep_net.encoder(attr_idx, edge_idx, n, d)
        loss = model.ep_net.recon_loss(z, pos_edge_idx,n, neg_edge_idx)
        # loss = model.ep_net.loss(attr_idx, edge_idx, n, d)
        auc_test = model.ep_net.test(z, pos_test, neg_test)
        print(f'epoch: {epoch}, loss: {loss.item()}, test_auc: {auc_test}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if auc_test > best_auc:
            best_auc = auc_test
            best_epoch = epoch
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
    print(f'best_epoch:{best_epoch}, auc_test:{auc_test}')
    model.load_state_dict(best_state)

    adj_logit=model.ep_net(attr_idx, edge_idx, n, d)
    print(adj_logit)
    print('adj_logit>0.51:',(adj_logit>0.5).sum())
    # print(model.ep_net.fc_base.weight)

def train_Aug_gnn(model, subgraph_edges_train, subgraph_attrs_train, labels_train,
              n,d, nc, subgraph_edges_val, subgraph_attrs_val, labels_val,
              idx_train, idx_val,attr_idx, edge_idx,idx, lr, weight_decay, patience, max_epochs,
              pretrain_ep=30,pretrain_nc=0,display_step=50,batch_size=1, early_stopping=True):

    trace_val = []
    model.train()
    n_train = torch.tensor(len(labels_train))
    n_val = torch.tensor(len(labels_val))
    n_subgraphs=len(subgraph_edges_train)
    assert n_subgraphs % batch_size == 0
    nbatches = n_subgraphs // batch_size

    #only use all the training graph for edge prediciton net pretraining---
    edge_idx_train = torch.cat(subgraph_edges_train, dim=1)#TODO: 有重复的边吗？有，self_loop重复了
    attr_idx_train = subgraph_attrs_train[0]
    edge_idx_val = torch.cat(subgraph_edges_val, dim=1)
    attr_idx_val = subgraph_attrs_val[0]
    # sample postive edges
    pos_edge_idx = sample_positive_edges(edge_idx_train, sample_ratio = 0.7)
    # neg_edge_idx = sample_negative_edges(edge_idx_train, n_train, num_samples=pos_edge_idx.shape[1]*10)
    neg_edge_idx = None# it will be sampled randomly in the loss function
    if pretrain_ep:# pretrain VGAE
        pretrain_ep_net(model,edge_idx_train, pos_edge_idx, neg_edge_idx, attr_idx_train, n_train, d,lr,weight_decay, pretrain_ep)
        print('reconstruction acc,auc on validation set:')
        model.ep_net.test_all(edge_idx_val, attr_idx_val, n_val, d, idx_val)

    model.set_adj_logit(attr_idx, edge_idx, n, d, idx)
    batch_id=0
    batch_subgraph_edges = subgraph_edges_train[batch_id * batch_size:(batch_id + 1) * batch_size]
    batch_subgraph_attrs = subgraph_attrs_train[batch_id * batch_size:(batch_id + 1) * batch_size]
    edge_idx_batch, attr_idx_batch, idx_train_batch, labels_train_batch = get_batch_idx(batch_subgraph_edges,
                                                                                        batch_subgraph_attrs,
                                                                                        idx_train, labels_train,
                                                                                        n_train, batch_size)
    model.set_daptive_thresholds(attr_idx_batch, edge_idx_batch, n_train, batch_size, d)

    optimizer = torch.optim.Adam(model.nc_net.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = 0
    best_performance = 0.0
    best_loss = np.inf
    last_time = get_time()
    STOP=False

    for it in range(max_epochs):
        for batch_id in range(nbatches):
            optimizer.zero_grad()
            model.train()
            model.subgraph_mode(mode='train')
            batch_subgraph_edges = subgraph_edges_train[batch_id * batch_size:(batch_id + 1) * batch_size]
            batch_subgraph_attrs = subgraph_attrs_train[batch_id * batch_size:(batch_id + 1) * batch_size]
            edge_idx_batch, attr_idx_batch, idx_train_batch, labels_train_batch = get_batch_idx(batch_subgraph_edges,
                                                                                                batch_subgraph_attrs,
                                                                                                idx_train, labels_train,
                                                                                                n_train, batch_size)
            logits_train = model.forward(attr_idx=attr_idx_batch, edge_idx=edge_idx_batch, n=n_train*batch_size, d=d)
            loss_nc = F.cross_entropy(logits_train[idx_train_batch], labels_train_batch[idx_train_batch])
            loss_train = loss_nc
            loss_train.backward()
            optimizer.step()
            with torch.no_grad():
                model.eval()
                model.subgraph_mode(mode='val')
                batch_subgraph_edges = subgraph_edges_val[batch_id * batch_size:(batch_id + 1) * batch_size]
                batch_subgraph_attrs = subgraph_attrs_val[batch_id * batch_size:(batch_id + 1) * batch_size]
                edge_idx_batch, attr_idx_batch, idx_val_batch, labels_val_batch = get_batch_idx(
                    batch_subgraph_edges, batch_subgraph_attrs,
                    idx_val, labels_val, n_val, batch_size)
                logits_val = model.forward(attr_idx=attr_idx_batch, edge_idx=edge_idx_batch, n=n_val*batch_size, d=d)
                loss_val = F.cross_entropy(logits_val[idx_val], labels_val[idx_val])
                trace_val.append(loss_val.item())

            if loss_val < best_loss:
                best_loss = loss_val
                best_epoch = nbatches*it+batch_id
                best_state = {key: value.cpu()
                              for key, value in model.state_dict().items()}
            else:
                if nbatches*it+batch_id >= best_epoch + patience and early_stopping:
                    STOP=True
                    break
            if (nbatches*it+batch_id) % display_step == 0:
                acc_train = accuracy(labels_train, logits_train, idx_train)
                acc_val = accuracy(labels_val, logits_val, idx_val)
                current_time = get_time()
                print(f'Epoch {nbatches*it+batch_id:4}: loss_train: {loss_train.item():.5f}, loss_val: {loss_val.item():.5f} '
                      f'acc_train: {acc_train:.5f}, acc_val: {acc_val:.5f} ({current_time - last_time:.3f}s)')
                last_time = current_time
                if best_val < acc_val:
                    best_val = acc_val
        if STOP:
            break

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
    print(best_hyperparams)
    model.subgraph_mode(mode='test')
    return best_hyperparams