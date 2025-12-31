import torch
import torch.nn.functional as F
from sparse_auditvotes.utils import sample_multiple_graphs, binary_perturb
from tqdm.autonotebook import tqdm
from sparse_auditvotes.filter import get_conf,get_entropy,get_homo,get_prox1,Thre_Filter

# attr_sparse = torch.sparse.FloatTensor(attr_idx_batch, torch.ones(attr_idx_batch.size(1), device=device),
#                                        torch.Size([n*batch_size, d]))
# edge_sparse = torch.sparse.FloatTensor(edge_idx_batch, torch.ones(edge_idx_batch.size(1), device=device),
#                                        torch.Size([n*batch_size, n*batch_size]))

def predict_smooth_gnn(args, attr_idx, edge_idx, sample_config, model, n, d, nc, batch_size=10,analysis_data=None):
    """
    Parameters
    ----------
    attr_idx: torch.Tensor [2, ?]
        The indices of the non-zero attributes.
    edge_idx: torch.Tensor [2, ?]
        The indices of the edges.
    sample_config: dict
        Configuration specifying the sampling probabilities
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
    filter : a trained detecor
    Returns
    -------
    votes : array-like [n, nc]
        The votes per class for each node.
    """
    print("smoothing-------------")
    n_samples = sample_config.get('n_samples', 1)
    device = attr_idx.device
    model.eval()
    votes = torch.zeros((n, nc), dtype=torch.long, device=device)
    with torch.no_grad():
        assert n_samples % batch_size == 0
        nbatches = n_samples // batch_size
        for i in tqdm(range(nbatches)):
            attr_idx_batch, edge_idx_batch = sample_multiple_graphs(
                    attr_idx=attr_idx, edge_idx=edge_idx,
                    sample_config=sample_config, n=n, d=d, nsamples=batch_size)
            # if i<=5:
            #     if args.model in ['JacGCN','FAugGCN','SAugGCN']:
            #         adj_aug = model.get_aug_edge_idx(edge_idx_batch, n*batch_size)
            #         sparsity=(adj_aug.size(1)+(n*batch_size))/((n**2)*batch_size)
            #     else:
            #         sparsity=edge_idx_batch.size(1)/((n**2)*batch_size)
            #     print('sparsity:',sparsity)

            embeddings = model(attr_idx=attr_idx_batch, edge_idx=edge_idx_batch,n=batch_size * n, d=d)
            predictions = embeddings.argmax(1)
            if  args.filter not in ['Conf'] and args.model in ['JacGCN','FAugGCN','SAugGCN']:
                edge_idx_batch=model.get_aug_edge_idx(edge_idx_batch, n * batch_size)
            if args.filter != '': #smoothing with filter
                if args.filter == 'Conf':
                    conf = get_conf(embeddings)
                    pred_anomaly,scores_pred = Thre_Filter(1-conf,n,batch_size,thre=args.conf_thre)
                elif args.filter == 'Entr':
                    entropy = get_entropy(embeddings)
                    pred_anomaly, scores_pred = Thre_Filter(entropy, n, batch_size, thre=args.etr_thre)
                elif args.filter == 'Homo':
                    homophilys = get_homo(edge_idx_batch, n, batch_size, predictions)
                    # pred_anomaly, scores_pred = Homophiliy_Filter(graph,predictions,n,batch_size,det_ratio=0.95)???
                    pred_anomaly, scores_pred = Thre_Filter(1-homophilys, n, batch_size,
                                                                   thre=args.homo_thre)
                elif args.filter == 'Prox1':
                    prox1 = get_prox1(embeddings, edge_idx_batch, n, batch_size)
                    pred_anomaly,scores_pred = Thre_Filter(prox1,n,batch_size,thre=args.prox_thre)
                elif args.filter == 'Prox2':
                    prox2 = get_prox2(embeddings, edge_idx_batch, n, batch_size)
                    pred_anomaly,scores_pred = Thre_Filter(prox2,n,batch_size,thre=args.prox_thre)
                elif args.filter == 'JSD':
                    JSD = get_JSD(embeddings, edge_idx_batch, n,nc, batch_size)
                    pred_anomaly, scores_pred = Thre_Filter(JSD, n, batch_size, thre=args.jsd_thre)
                elif args.filter == 'NSP':
                    NSP = get_NSP(edge_idx_batch,attr_idx_batch,n,d,batch_size)
                    pred_anomaly, scores_pred = Thre_Filter(1-NSP, n, batch_size, thre=args.nsp_thre)

                one_hots=F.one_hot(predictions, nc).reshape(batch_size, n, nc)
                preds_onehot = (one_hots* (1 - pred_anomaly).type(torch.LongTensor).unsqueeze(2).to(device)).sum(0)
                if analysis_data !=None:
                    if args.dataset=='pubmed':
                        if i==0:
                            analysis_data['anomaly_score'].extend(scores_pred.view(-1).tolist())
                            analysis_data['anomaly_pred'].extend(pred_anomaly.view(-1).tolist())
                            analysis_data['class_pred'].extend(predictions.tolist())
                    else:
                        analysis_data['anomaly_score'].extend(scores_pred.view(-1).tolist())
                        analysis_data['anomaly_pred'].extend(pred_anomaly.view(-1).tolist())
                        analysis_data['class_pred'].extend(predictions.tolist())
            else:
                one_hots = F.one_hot(predictions, nc)
                preds_onehot = one_hots.reshape(batch_size, n, nc).sum(0)
            votes += preds_onehot
    return votes.cpu().numpy(),analysis_data

