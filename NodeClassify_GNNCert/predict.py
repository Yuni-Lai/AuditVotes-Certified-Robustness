import torch
import torch.nn.functional as F
# from sparse_smoothing.utils import sample_multiple_graphs, binary_perturb
from tqdm.autonotebook import tqdm
from utils import Conf_Filter,get_batch_idx2
from Detector_DGMM.detector import DGMM_detect
from Detector_Quality.train_test import *

def predict_smooth_gnn(args, subgraph_edges, subgraph_attrs, model, n, d, nc, batch_size=10,detector=None,analysis_data=None):
    """
    Parameters
    ----------
    subgraph_attrs: list of torch.Tensor [2, ?]
        The indices of the non-zero attributes.
    subgraph_edges: list of torch.Tensor [2, ?]
        The indices of the edges.
    model : torch.nn.Module
        The GNN model.
    n : int
        Number of nodes
    d : int
        Number of features
    nc : int
        Number of classes
    batch_size : int
        Number of graphs to sample per batch
    detector : a trained detecor
    Returns
    -------
    votes : array-like [n, nc]
        The votes per class for each node.
    """
    print("smoothing-------------")
    device = subgraph_attrs[0].device
    model.eval()
    n_subgraphs = len(subgraph_attrs)
    votes = torch.zeros((n_subgraphs,n, nc), dtype=torch.long, device=device)
    anomaly_scores = torch.zeros((n_subgraphs,n), dtype=torch.float, device=device)
    assert n_subgraphs % batch_size == 0
    nbatches = n_subgraphs // batch_size

    for batch_id in tqdm(range(nbatches)):
        with torch.no_grad():
            batch_subgraph_edges = subgraph_edges[batch_id*batch_size:(batch_id+1)*batch_size]
            batch_subgraph_attrs = subgraph_attrs[batch_id*batch_size:(batch_id+1)*batch_size]
            edge_idx_batch, attr_idx_batch = get_batch_idx2(
                batch_subgraph_edges, batch_subgraph_attrs, n)

            embeddings = model(attr_idx=attr_idx_batch, edge_idx=edge_idx_batch,
                           n=batch_size * n, d=d)

            predictions = embeddings.argmax(1)
            if detector !=None:
                # attr_sparse = torch.sparse.FloatTensor(attr_idx_batch, torch.ones(attr_idx_batch.size(1), device=device),
                #                                        torch.Size([n*batch_size, d]))
                # edge_sparse = torch.sparse.FloatTensor(edge_idx_batch, torch.ones(edge_idx_batch.size(1), device=device),
                #                                        torch.Size([n*batch_size, n*batch_size]))
                if args.detector == 'DGMM':
                    pred_anomaly,scores_pred = DGMM_detect(embeddings, detector, attr_idx_batch, edge_idx_batch, n*batch_size, d, nc, device)
                elif args.detector == 'Conf':
                    #pred_anomaly,scores_pred = Conf_Filter2(embeddings, n, batch_size, det_ratio=0.99)
                    pred_anomaly, scores_pred = Conf_Filter3(embeddings, n, batch_size, thre=args.conf_thre)
                else:
                    pred_anomaly,scores_pred = MisData_detect(detector, attr_idx_batch, edge_idx_batch, n, batch_size, d, nc, device)

                anomaly_scores[batch_id*batch_size:(batch_id+1)*batch_size,:] = scores_pred.reshape(batch_size,n)
                one_hots=F.one_hot(predictions, nc).reshape(batch_size,n, nc)
                preds_onehot = (one_hots* (1 - pred_anomaly).type(torch.LongTensor).unsqueeze(2).to(device))
                # preds_onehot = one_hots

                if analysis_data !=None:
                    analysis_data['anomaly_score'].extend(scores_pred.view(-1).tolist())# [], 'anomaly_pred': [], 'class_label': []
                    analysis_data['anomaly_pred'].extend(pred_anomaly.view(-1).tolist())
                    analysis_data['class_pred'].extend(predictions.tolist())
            else:
                one_hots = F.one_hot(predictions, nc)
                preds_onehot = one_hots.reshape(batch_size,n, nc)
            votes[batch_id*batch_size:(batch_id+1)*batch_size,:,:] = preds_onehot

    # anom_preds=Conf_Filter_all(anomaly_scores, n, n_subgraphs,min_thre=0.0, det_ratio=0.6)
    # rows,colums=torch.where(anom_preds==1)
    # votes[rows,colums,:]=0
    return votes.cpu().numpy(),analysis_data

