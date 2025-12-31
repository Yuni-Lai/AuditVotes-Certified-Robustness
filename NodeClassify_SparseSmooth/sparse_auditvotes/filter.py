import sys
import numpy as np
import dgl
import torch
import torch.nn.functional as F
import torch.optim as optim
import scipy as sp
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import dgl.function as fn

def get_long_edges(graph):
    """Internal function for getting the edges of a graph as long tensors."""
    src, dst = graph.edges()
    return src.long(), dst.long()

def node_homophily(graph, y):
    r"""Homophily measure from `Geom-GCN: Geometric Graph Convolutional
    Networks <https://arxiv.org/abs/2002.05287>`__

    We follow the practice of a later paper `Large Scale Learning on
    Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods
    <https://arxiv.org/abs/2110.14446>`__ to call it node homophily.

    Mathematically it is defined as follows:

    .. math::
      \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \frac{ | \{u
      \in \mathcal{N}(v): y_v = y_u \} |  } { |\mathcal{N}(v)| },

    where :math:`\mathcal{V}` is the set of nodes, :math:`\mathcal{N}(v)` is
    the predecessors of node :math:`v`, and :math:`y_v` is the class of node
    :math:`v`.

    Parameters
    ----------
    graph : DGLGraph
        The graph.
    y : torch.Tensor
        The node labels, which is a tensor of shape (|V|).

    Returns
    -------
    float
        The node homophily value.

    Examples
    --------
    # >>> import dgl
    # >>> import torch
    #
    # >>> graph = dgl.graph(([1, 2, 0, 4], [0, 1, 2, 3]))
    # >>> y = torch.tensor([0, 0, 0, 0, 1])
    # >>> dgl.node_homophily(graph, y)
    0.6000000238418579
    """
    check_pytorch()
    with graph.local_scope():
        # Handle the case where graph is of dtype int32.
        src, dst = get_long_edges(graph)
        # Compute y_v = y_u for all edges.
        graph.edata["same_class"] = (y[src] == y[dst]).float()
        graph.update_all(
            fn.copy_e("same_class", "m"), fn.mean("m", "same_class_deg")
        )
        # return graph.ndata["same_class_deg"].mean(dim=0).item()
        # for isolated node/ singleton, we set the homophily to maximum value 1.0.
        graph.ndata["same_class_deg"][torch.where(graph.in_degrees()==0)]=1.0
        return graph.ndata["same_class_deg"]

def get_conf(embs_eval):
    probs_vectors = F.softmax(embs_eval, dim=1)
    Confs = torch.max(probs_vectors, dim=1).values
    return Confs

def get_entropy(embs_eval):
    probs_vectors = F.softmax(embs_eval, dim=1)
    entropy = torch.sum(-probs_vectors * torch.log(probs_vectors), dim=1)
    return entropy
def get_homo(edge_idx_batch,n,batch_size,predictions,log=False):
    graph = dgl.graph((edge_idx_batch[0, :], edge_idx_batch[1, :]), num_nodes=n * batch_size)
    if log:
        homophilys = torch.log(node_homophily(graph, predictions)+0.01)
    else:
        homophilys = node_homophily(graph, predictions)
    return homophilys

def prox_1(S, adj):
    n_nodes = adj.shape[0]
    node_degree = adj.sum(1).reshape(-1,1)
    pairwise_KL = (((S * torch.log(S)).sum(1)).reshape(-1,1).repeat(1,n_nodes) - (S @ torch.log(S.T)))
    tmp = (pairwise_KL * adj).sum(1)
    prox1=((1/node_degree) * tmp.reshape(-1,1))
    prox1=torch.nan_to_num(prox1, nan=1.0, posinf=1.0, neginf=0.0)
    return prox1.squeeze()

def prox_2(S, adj):
    n_nodes = adj.shape[0]
    node_degree = adj.sum(1).reshape(-1, 1)
    pairwise_KL = (((S * torch.log(S)).sum(1)).reshape(-1, 1).repeat(1, n_nodes) - (S @ torch.log(S.T)))
    tmp = ((adj.float() @ pairwise_KL) * adj).sum(1)
    prox2=(1 / (node_degree * (node_degree - 1)) * tmp.reshape(-1, 1))
    prox2 = torch.nan_to_num(prox2, nan=1.0, posinf=1.0, neginf=0.0)
    return prox2.squeeze()

def JSDivergence(S, adj, nclass):
    node_degree = adj.sum(1).reshape(-1, 1)
    tmp = ((1 / node_degree).repeat(1, nclass)) * (adj.float() @ S)
    tmp /= tmp.sum(1).reshape(-1, 1).repeat(1, nclass)
    tmp1 = (-tmp * torch.log(tmp)).sum(1)
    tmp2 = (1 / node_degree) * (adj.float() @ torch.distributions.Categorical(S).entropy().reshape(-1, 1))
    JSD = tmp1.reshape(-1, 1) - tmp2
    JSD = torch.nan_to_num(JSD, nan=1.0, posinf=1.0, neginf=0.0)
    return JSD.squeeze()

def get_prox1(embeddings,edge_idx_batch,n,batch_size):
    probs_vectors = F.softmax(embeddings, dim=1)
    graph = dgl.graph((edge_idx_batch[0, :], edge_idx_batch[1, :]), num_nodes=n * batch_size)
    prox1 = prox_1(probs_vectors, graph.adj().to_dense())
    prox1 = torch.log(prox1)
    prox1 = torch.nan_to_num(prox1, nan=0.0, neginf=0.0)
    return prox1

def get_prox2(embeddings,edge_idx_batch,n,batch_size):
    probs_vectors = F.softmax(embeddings, dim=1)
    graph = dgl.graph((edge_idx_batch[0, :], edge_idx_batch[1, :]), num_nodes=n * batch_size)
    prox2 = prox_2(probs_vectors, graph.adj().to_dense())
    prox2 = torch.log(prox2)
    prox2 = torch.nan_to_num(prox2, nan=0.0, neginf=0.0)
    return prox2

def get_JSD(embeddings,edge_idx_batch,n,nc,batch_size):
    probs_vectors = F.softmax(embeddings, dim=1)
    graph = dgl.graph((edge_idx_batch[0, :], edge_idx_batch[1, :]), num_nodes=n * batch_size)
    JSD = JSDivergence(probs_vectors, graph.adj().to_dense(),nc)
    JSD = torch.log(JSD)
    JSD = torch.nan_to_num(JSD, nan=0.0, posinf=0.0)
    return JSD

def get_NSP(edge_idx_batch,attr_idx_batch,n,d,batch_size):
    attr_sparse = torch.sparse.FloatTensor(attr_idx_batch, torch.ones(attr_idx_batch.size(1), device=attr_idx_batch.device),
                                           torch.Size([n*batch_size, d]))
    edge_sparse = torch.sparse.FloatTensor(edge_idx_batch, torch.ones(edge_idx_batch.size(1), device=attr_idx_batch.device),
                                           torch.Size([n*batch_size, n*batch_size]))

    NSP = neighbors_sim(attr_sparse.to_dense(),edge_sparse.to_dense(),order=1)
    return NSP

def get_features(embs_eval,n,d,batch_size,nc,edge_idx_batch,attr_idx_batch,predictions,features_list):
    features = []
    for feature_name in features_list:
        if feature_name == 'Conf':
            Confs = get_conf(embs_eval)
            features.append(Confs)
        if feature_name == 'Entr':
            entropy = get_entropy(embs_eval)
            features.append(entropy)
        if feature_name == 'Emb':
            features.extend(embs_eval.t())
        if feature_name == 'Homo':
            homophilys = get_homo(edge_idx_batch, n, batch_size, predictions)
            features.append(homophilys)
        if feature_name == 'Prox1':
            prox1 = get_prox1(embs_eval,edge_idx_batch, n, batch_size)
            features.append(prox1)
        if feature_name == 'Prox2':
            prox2 = get_prox2(embs_eval,edge_idx_batch, n, batch_size)
            features.append(prox2)
        if feature_name == 'JSD':
            JSD = get_JSD(embs_eval,edge_idx_batch, n, nc, batch_size)
            features.append(JSD)
        if feature_name == 'NSP':
            NSP=get_NSP(edge_idx_batch,attr_idx_batch,n,d,batch_size)
            features.append(NSP)
    features = torch.transpose(torch.stack(features), 0, 1)
    return features

def generate_conf_dataset(model,n,d,attr_idx,edge_idx,labels,idx,sample_config,args):
    from sparse_auditvotes.utils import sample_multiple_graphs
    attr_idx_batch, edge_idx_batch = sample_multiple_graphs(
        attr_idx=attr_idx, edge_idx=edge_idx,
        sample_config=sample_config, n=n, d=d,
        nsamples=args.batch_size_detect)
    with torch.no_grad():
        embs_eval = model(attr_idx=attr_idx_batch, edge_idx=edge_idx_batch, n=n * args.batch_size_detect, d=d)
        predictions = embs_eval.argmax(1)
        Confs = get_conf(embs_eval)
    detect_labels = (predictions != labels.repeat(args.batch_size_detect)).long()
    if args.model.split('_')[0] == "GCNJaccard":
        edge_idx_batch = model.Jaccard_mask_edge_idx(edge_idx_batch, n * args.batch_size_detect)
    data = Dataset_from_sparse(attr_idx_batch, edge_idx_batch, detect_labels, idx, n, d, args.batch_size_detect, 'randomgraph',
                               args.device)
    return data,Confs

def generate_combine_dataset(model,n,d,attr_idx,edge_idx,labels,idx,sample_config,features_list,args):
    from sparse_auditvotes.utils import sample_multiple_graphs
    attr_idx_batch, edge_idx_batch = sample_multiple_graphs(
        attr_idx=attr_idx, edge_idx=edge_idx,
        sample_config=sample_config, n=n, d=d,
        nsamples=args.batch_size_detect)
    nc=int(labels.max()+1)
    with torch.no_grad():
        embs_eval = model(attr_idx=attr_idx_batch, edge_idx=edge_idx_batch, n=n * args.batch_size_detect, d=d)
    predictions = embs_eval.argmax(1)
    if args.model == "GCNJaccard_Aug":
        edge_idx_batch = model.Jaccard_mask_edge_idx(edge_idx_batch, n * args.batch_size_detect)
    elif args.model in ["FAugGCN","SAugGCN"]:
        edge_idx_batch = model.get_aug_edge_idx(edge_idx_batch, n * args.batch_size_detect)
    features = get_features(embs_eval, n,d, args.batch_size_detect, nc, edge_idx_batch,attr_idx_batch, predictions, features_list)
    detect_labels = (predictions != labels.repeat(args.batch_size_detect)).long()
    data = Dataset_from_sparse(attr_idx_batch, edge_idx_batch, detect_labels, idx, n, d, args.batch_size_detect, 'randomgraph',
                               args.device)
    return data,features


def train_GAD_detector(model,n,d,attr_idx,edge_idx,labels,idx,sample_config,args,savedir):
    # We want to improve the training of the detector that are able to detect misclassified samples
    # Input: random graphs
    # Label: correct classified or not
    # Model: BWGNN
    print('training detector------------')
    model.eval()
    train_config = {'device': args.device, 'epochs': 1, 'patience': 100, 'metric': 'AUROC', 'inductive': False,
                    'seed': 3000}
    model_config = {'model': args.detector, 'lr': 0.0001, 'drop_rate': 0.5, 'lambda':1.0}
    data,_ = generate_conf_dataset(model,n, d, attr_idx, edge_idx, labels, idx, sample_config, args)
    detect_model = model_detector_dict[args.detector](train_config, model_config, data)
    for i in range(args.batch_number): #the model weight is initiated again
        print('number of batch:',i)
        data,_=generate_conf_dataset(model,n, d, attr_idx, edge_idx, labels, idx, sample_config, args)
        detect_model.data_init(data)
        detect_model.train()
    torch.save(detect_model, savedir)
    print('detector model saved to:',savedir)
    return detect_model


def train_Conf_detector(model,n,d,attr_idx,edge_idx,labels,idx,sample_config,args,savedir):
    # We want to train a GNN model along with the confidence as input.
    # Input: random graphs
    # Label: correct classified or not
    # Model: GCN + Conf
    from Detector_Conf.models.detector import BaseGNNDetector
    print('training detector------------')
    model.eval()
    train_config = {'device': args.device, 'epochs': 1, 'patience': 100, 'metric': 'AUROC', 'inductive': False,
                    'seed': 3000}
    model_config = {'model': args.detector.split('_Conf')[0], 'lr': 0.0001, 'drop_rate': 0.5, 'lambda':1.0}
    # idx['train']=idx['val']
    data,Confs=generate_conf_dataset(model,n, d, attr_idx, edge_idx, labels, idx, sample_config, args)
    detect_model = BaseGNNDetector(train_config, model_config, data)
    for i in range(args.batch_number): #the model weight is initiated again
        print('number of batch:',i)
        data,Confs=generate_conf_dataset(model,n, d, attr_idx, edge_idx, labels, idx, sample_config, args)
        detect_model.data_init(data)
        detect_model.confs = Confs
        detect_model.train()
    torch.save(detect_model, savedir)
    print('detector model saved to:',savedir)
    return detect_model


def train_Combine_detector(model,n,d,attr_idx,edge_idx,labels,idx,sample_config,args,savedir):
    # We want to train a GNN model along with the combined features as input.
    # Input: random graphs
    # Label: correct classified or not
    # Model: GCN + Conf + Homo
    from Detector_Combine.models.detector import BaseGNNDetector
    features_list = args.detector.split('_')[-1].split('+')  # 'Conf','Homo'
    print('training detector------------')
    model.eval()
    train_config = {'device': args.device, 'epochs': 1, 'patience': 10, 'metric': 'AUROC', 'inductive': False,
                    'seed': 3000}
    model_config = {'model': args.detector.split('_')[0], 'lr': 0.0001, 'drop_rate': 0.5, 'lambda':0.01, 'add_feat':len(features_list)}
    data, features = generate_combine_dataset(model, n, d, attr_idx, edge_idx, labels, idx, sample_config,
                                              features_list, args)

    if args.detector.split('_')[0]=='MLP':
        model_config['in_feats'] = features.shape[1]
        model_config['h_feats'] = features.shape[1]
        model_config['num_layers'] = 2
        model_config['activation'] = 'Sigmoid'
        model_config['lr'] = 0.01
        model_config['lambda'] = 0.0001
    detect_model = BaseGNNDetector(train_config, model_config, data)
    for i in range(args.batch_number):
        print('number of batch:',i)
        data,features = generate_combine_dataset(model,n, d, attr_idx, edge_idx, labels, idx, sample_config, features_list, args)
        detect_model.data_init(data)
        detect_model.features = features
        detect_model.train()
    torch.save(detect_model, savedir)
    print('detector model saved to:',savedir)
    return detect_model

def GAD_detect(detector, attr_idx_batch, edge_idx_batch, n, batch_size, d, device,det_ratio=0.2):
    data = Dataset_from_sparse(attr_idx_batch, edge_idx_batch, None, None, n, d, batch_size, 'randomgraph',
                               device)
    graph=data.get_graph()
    try:
        logits = detector.model(graph)
        scores = logits.softmax(1)[:, 1]#.cpu().numpy()
    except:
        scores = detector.test(graph)
    thresh = torch.quantile(scores, 1-det_ratio)
    scores = scores.reshape(batch_size, n)
    abnormal_node_idx = torch.where(scores >= thresh)
    anom_preds = torch.zeros((batch_size, n))
    anom_preds[abnormal_node_idx] = 1
    return anom_preds,scores
    # print('auc', roc_auc_score(ano_label, cluster_pred))

def Conf_detect(detector, attr_idx_batch, edge_idx_batch, confs,n, batch_size, d, device,det_ratio=0.2):
    data = Dataset_from_sparse(attr_idx_batch, edge_idx_batch, None, None, n, d, batch_size, 'randomgraph',
                               device)
    graph = data.get_graph()
    try:
        logits = detector.model(graph,confs)
        scores = logits.softmax(1)[:, 1]#.cpu().numpy()
    except:
        scores = detector.test(graph,confs)
    thresh = torch.quantile(scores, 1-det_ratio)
    scores = scores.reshape(batch_size, n)
    #thresh = torch.quantile(scores, 1 - det_ratio, dim=0)
    abnormal_node_idx = torch.where(scores >= thresh)
    anom_preds = torch.zeros((batch_size, n))
    anom_preds[abnormal_node_idx] = 1
    return anom_preds,scores

def Combine_detect(detector, attr_idx_batch, edge_idx_batch, features, n, batch_size, d, device,det_ratio=0.2):
    data = Dataset_from_sparse(attr_idx_batch, edge_idx_batch, None, None, n, d, batch_size, 'randomgraph',
                               device)
    graph = data.get_graph()
    try:
        logits = detector.model(graph,features)
        scores = logits.softmax(1)[:, 1]#.cpu().numpy()
    except:
        scores = detector.test(graph,features)

    # scores = scores.reshape(batch_size, n)
    # thresh = torch.quantile(scores, 1-det_ratio, dim=0)
    thresh = torch.quantile(scores, 1-det_ratio)
    scores = scores.reshape(batch_size, n)
    abnormal_node_idx = torch.where(scores >= thresh)
    anom_preds = torch.zeros((batch_size, n))
    anom_preds[abnormal_node_idx] = 1
    return anom_preds,scores

def test_conf_AUC(model,n,d,attr_idx, edge_idx, labels, idx, sample_config, args):
    model.eval()
    AUC_avr = 0
    for i in range(10): #the model weight is initiated again
        data, Confs = generate_conf_dataset(model, n, d, attr_idx, edge_idx, labels, idx, sample_config, args)
        test=data.graph.ndata['test_mask'].cpu().numpy()
        AUC=roc_auc_score(data.graph.ndata['label'].cpu().numpy()[test], -Confs.cpu().numpy()[test])
        AUC_avr=AUC_avr+AUC
    AUC_avr=AUC_avr/10.0
    print('Average Detector AUC:', AUC_avr)

# def Conf_Filter(embeddings,n,batch_size):
#     probs_vectors = F.log_softmax(embeddings, dim=1)
#     probs = torch.max(probs_vectors,dim=1).values.cpu().numpy()
#     probs = -probs
#     thresh = np.quantile(probs, 0.8)
#     abnormal_node_idx = np.where(probs >= thresh)[0]
#     anom_preds = np.zeros((n * batch_size))
#     anom_preds[abnormal_node_idx] = 1
#     return anom_preds, probs

def Conf_Filter(embeddings,num_graphs,n_samples,det_ratio=0.8):
    probs_vectors = F.log_softmax(embeddings, dim=1)
    probs = torch.max(probs_vectors,dim=2).values
    probs = 1-probs
    thresh = torch.quantile(probs, det_ratio, dim=0)
    abnormal_node_idx = torch.where(probs >= thresh)
    anom_preds = torch.zeros((n_samples,num_graphs))
    anom_preds[abnormal_node_idx] = 1
    return anom_preds, probs

def Conf_Filter2(embeddings,n,batch_size,det_ratio=0.8):
    # probs_vectors = F.log_softmax(embeddings, dim=1)
    probs_vectors = F.softmax(embeddings, dim=1)
    probs = torch.max(probs_vectors,dim=1).values
    anomaly_vector = 1-probs
    anomaly_array=anomaly_vector.reshape(batch_size,n)
    thresh = torch.quantile(anomaly_array, det_ratio, dim=0)# 差异好大哦
    abnormal_node_idx = torch.where(anomaly_array >= thresh)
    anom_preds = torch.zeros((batch_size,n))
    anom_preds[abnormal_node_idx] = 1
    return anom_preds, anomaly_vector


def get_anomaly_scores(embeddings):
    probs_vectors = F.softmax(embeddings, dim=1)
    probs = torch.max(probs_vectors, dim=1).values
    anomaly_vector = 1 - probs
    return anomaly_vector


def Conf_Filter_all(anomaly_array,n,n_subgraphs,min_thre=0.7,det_ratio=0.8):
    thresh = torch.quantile(anomaly_array, det_ratio, dim=0)
    thresh = thresh.clip(min=min_thre)
    abnormal_node_idx = torch.where(anomaly_array >= thresh)
    anom_preds = torch.zeros((n_subgraphs,n))
    anom_preds[abnormal_node_idx] = 1

    # Ensure at most (1-det_ratio)% of subgraphs per node are predicted as anomalies
    max_anomalies_per_node = int((1-det_ratio) * n_subgraphs)
    for node in range(n):
        node_anomalies = torch.where(anom_preds[:, node] == 1)[0]
        if len(node_anomalies) > max_anomalies_per_node:
            # Randomly select indices to set back to 0
            indices_to_reset = node_anomalies[max_anomalies_per_node:]
            anom_preds[indices_to_reset, node] = 0
    return anom_preds

def Conf_Filter_all_1(anomaly_array,n,n_subgraphs,conf_thre=0.7,max_det_ratio=0.4):
    # Ensure at most (max_det_ratio)% of subgraphs per node are predicted as anomalies
    min_thresh = torch.quantile(anomaly_array, 1-max_det_ratio, dim=0)
    thre = torch.maximum(min_thresh,(torch.ones(n)*conf_thre).to(min_thresh.device))
    abnormal_node_idx = torch.where(anomaly_array >= thre)
    anom_preds = torch.zeros((n_subgraphs,n))
    anom_preds[abnormal_node_idx] = 1
    return anom_preds



def Homophiliy_Filter(homophilys,n,batch_size,det_ratio=0.8):
    # homophilys = node_homophily(graph,predictions)
    anomaly_vector = 1-homophilys
    anomaly_array = anomaly_vector.reshape(batch_size,n)
    thresh = torch.quantile(anomaly_array, det_ratio, dim=0)#
    abnormal_node_idx = torch.where(anomaly_array >= thresh)
    anom_preds = torch.zeros((batch_size,n))
    anom_preds[abnormal_node_idx] = 1
    return anom_preds, anomaly_vector


def Thre_Filter(anomaly_vector,n,batch_size,thre=0.8):
    anomaly_array = anomaly_vector.reshape(batch_size,n)
    abnormal_node_idx = torch.where(anomaly_array >= thre)
    anom_preds = torch.zeros((batch_size,n))
    anom_preds[abnormal_node_idx] = 1
    return anom_preds, anomaly_vector

def Thre_Filter_1(anomaly_vector,n,batch_size,thre=0.7,max_det_ratio=0.7):
    # Ensure at most (max_det_ratio)% of subgraphs per node are predicted as anomalies
    min_thresh = torch.quantile(anomaly_vector, 1-max_det_ratio, dim=0)
    thre = torch.maximum(min_thresh,(torch.ones(n)*thre).to(min_thresh.device))
    anomaly_array = anomaly_vector.reshape(batch_size, n)
    abnormal_node_idx = torch.where(anomaly_array >= thre)
    anom_preds = torch.zeros((batch_size,n))
    anom_preds[abnormal_node_idx] = 1
    return anom_preds,anomaly_vector