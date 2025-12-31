import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch_geometric.nn import GCNConv, GATConv, APPNP, global_mean_pool, JumpingKnowledge
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_remaining_self_loops,negative_sampling, remove_self_loops,add_self_loops
from torch_sparse import spmm
from numba import njit
import scipy.sparse as sp
import numpy as np
import pyro
import sys
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve

class SparseGCNConv(GCNConv):
    def __init__(self, in_channel, out_channel, **kwargs):
        super().__init__(in_channels=in_channel, out_channels=out_channel, **kwargs)

    def forward(self, x, edge_idx, n, d):
        x = spmm(x, torch.ones_like(x[0]), n, d, self.lin.weight.t())
        edge_idx, norm = gcn_norm(edge_idx, None, x.size(0), self.improved, self.add_self_loops, x.dtype)
        return self.propagate(edge_idx, x=x, edge_weight=norm, size=None)

class SparseGATConv(GATConv):
    def __init__(self, in_channel, out_channel, **kwargs):
        super().__init__(in_channels=in_channel, out_channels=out_channel, **kwargs)
        self.weight = self.lin.weight.data.T

    def forward(self, x, edge_idx, n, d):
        edge_idx, _ = add_remaining_self_loops(edge_idx)
        x = spmm(x, torch.ones_like(x[0]), n, d, self.weight)
        return self.propagate(edge_idx, x=x)

class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Our fan_in is interpreted by PyTorch as fan_out (swapped dimensions)
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, index, value, n):
        res = spmm(index, value, n, self.in_features, self.weight)
        if self.bias is not None:
            res += self.bias[None, :]
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
                self.in_features, self.out_features, self.bias is not None)

class GCN(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden, p_dropout=0.5):
        super().__init__()
        self.conv1 = SparseGCNConv(n_features, n_hidden)
        self.conv2 = GCNConv(n_hidden, n_classes)
        self.p_dropout = p_dropout

    def forward(self, attr_idx, edge_idx, n, d):
        hidden = F.relu(self.conv1(attr_idx, edge_idx, n, d))
        hidden = F.dropout(hidden, p=self.p_dropout, training=self.training)
        hidden = self.conv2(hidden, edge_idx)
        return hidden
        # return F.log_softmax(hidden, dim=1)

class GAT(torch.nn.Module):
    def __init__(self, n_features, n_classes, n_hidden, k_heads, p_dropout=0.6):
        super().__init__()
        self.p_dropout = p_dropout
        self.conv1 = SparseGATConv(n_features, n_hidden, heads=k_heads, dropout=self.p_dropout)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(
            n_hidden * k_heads, n_classes, heads=1, concat=True, dropout=self.p_dropout)

    def forward(self, attr_idx, edge_idx, n, d):
        # Regular GAT uses dropout on attributes and adjacency matrix
        x = F.elu(self.conv1(attr_idx, edge_idx, n, d))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.conv2(x, edge_idx)
        return F.log_softmax(x, dim=1)

class APPNPNet(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden, k_hops, alpha, p_dropout=0.5):
        super().__init__()
        self.lin1 = SparseLinear(n_features, n_hidden, bias=False)
        self.lin2 = nn.Linear(n_hidden, n_classes, bias=False)
        self.prop = APPNP(k_hops, alpha)
        self.p_dropout = p_dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, attr_idx, edge_idx, n, d):
        # Regular PPNP uses dropout on attributes and adjacency matrix
        x = F.relu(self.lin1(attr_idx, torch.ones_like(attr_idx[0]), n))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_idx)
        return F.log_softmax(x, dim=1)

class MLP(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, dropout_rate=0.5, activation='ReLU', **kwargs):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        if num_layers == 1:
            self.layers.append(nn.Linear(in_feats, num_classes))
        else:
            self.layers.append(nn.Linear(in_feats, in_feats/2))
            for i in range(1, num_layers-1):
                self.layers.append(nn.Linear(in_feats/2, in_feats/2))
            self.layers.append(nn.Linear(h_feats, num_classes))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, h):
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h)
            if i != len(self.layers)-1:
                h = self.act(h)
        return h

class JacAug(nn.Module):
    def __init__(self,device):
        super(JacAug, self).__init__()
        self.devide=device

    def jaccard_similarity(self, a, b):
        intersection = a.multiply(b).count_nonzero()
        J = intersection * 1.0 / (a.count_nonzero() + b.count_nonzero() - intersection)
        return J

    def forward(self, attr_idx, edge_idx, n, d):
        '''pre calculate the jaccard to avoid repeat'''
        features = torch.sparse.FloatTensor(attr_idx, torch.ones(attr_idx.size(1), device=attr_idx.device),
                                            torch.Size([n, d])).to_dense()
        features=np.matrix(features.cpu())
        nonzeros = np.sum(features != 0, axis=1).transpose()
        Jn = np.zeros((n, n))  # Jaccard similarity of n nodes: n x n shape
        # Calculate one node to all node each time. To avoid OOM in the case of large graph.
        for i in range(n):
            feature_i = features[i]
            intersections_i = feature_i * features.transpose()
            J_i = np.multiply(intersections_i, (1 / (np.count_nonzero(feature_i) + nonzeros - intersections_i)))
            Jn[i, :] = J_i
        self.Jn_all = torch.tensor(Jn).to(self.devide)
        return self.Jn_all

    def test_all(self,edge_idx,attr_idx, n, d, test_idx):
        with torch.no_grad():
            adj_logits = self.Jn_all
            adj_logits = adj_logits[test_idx, :][:, test_idx]
            adj_orig = torch.sparse.FloatTensor(edge_idx, torch.ones(edge_idx.size(1), device=edge_idx.device),
                                            torch.Size([n, n])).to_dense()
            adj_orig = adj_orig[test_idx, :][:, test_idx]
            fpr, tpr, thresholds = roc_curve(adj_orig.view(-1).cpu().numpy(), adj_logits.view(-1).cpu().numpy())
            best_index = np.argmax(tpr - fpr)
            best_threshold = thresholds[best_index]
            adj_recons = (torch.sigmoid(adj_logits) > 0.65).float()
            acc = accuracy_score(adj_orig.view(-1).cpu().numpy(), adj_recons.view(-1).cpu().numpy())
            auc = roc_auc_score(adj_orig.view(-1).cpu().numpy(), adj_logits.view(-1).cpu().numpy())
            k = int(adj_orig.sum().item())
            RecK = sum(adj_orig.view(-1)[adj_logits.view(-1).argsort()[-k:]]) / adj_orig.sum()
            PreK = sum(adj_orig.view(-1)[adj_logits.view(-1).argsort()[-k:]]) / k
            print('acc,auc,reck,preK:',acc,auc,RecK.item(),PreK.item())
        return acc,auc,best_threshold

class GCNJaccard_Aug(nn.Module):
    '''Pre-processing the graph by edge augmentation:
    Remove the edges with low Jaccard Similarity (between two node features) < threshold1
    And add the dgdes with high Jaccard Similarity (between two node features) > threshold2
    '''
    def __init__(self,n_features, n_classes, n_hidden,edge_proportion, p_plus, p_minus, p_dropout=0.5,threshold1=0.10, threshold2=0.5, binary_feature=True,device='cuda'):
        super().__init__()
        self.conv1 = SparseGCNConv(n_features, n_hidden)
        self.conv2 = GCNConv(n_hidden, n_classes)
        self.p_dropout = p_dropout
        self.threshold1 = threshold1 #keep threshold
        self.threshold2 = threshold2 #add threshold
        self.binary_feature = binary_feature
        self.edge_proportion=edge_proportion #A.sum()/(N**2)
        self.p_plus=p_plus
        self.p_minus=p_minus
        self.seted_thresholds_by_topK = False

    def reset_model(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def jaccard_similarity(self, a, b):
        intersection = a.multiply(b).count_nonzero()
        J = intersection * 1.0 / (a.count_nonzero() + b.count_nonzero() - intersection)
        return J

    def set_jaccard_matrix(self, features, idx, device):
        '''pre calculate the jaccard to avoid repeat'''
        n, d = features.shape  # n nodes, d dimension feature matrix
        nonzeros = np.sum(features != 0, axis=1).transpose()
        Jn = np.zeros((n, n))  # Jaccard similarity of n nodes: n x n shape
        # Calculate one node to all node each time. To avoid OOM in the case of large graph.
        for i in range(n):
            feature_i = features[i]
            intersections_i = feature_i * features.transpose()
            J_i = np.multiply(intersections_i, (1 / (np.count_nonzero(feature_i) + nonzeros - intersections_i)))
            Jn[i, :] = J_i
        self.Jn_all = torch.tensor(Jn).to(device)
        for id in idx.keys():
            print(id)
            idx[id] = torch.tensor(idx[id]).to(device)
        self.idx = idx
        self.n0_all = features.shape[0]
        self.devide = device

        # train
        subgraph_idx = torch.concatenate([idx['train'], idx['unlabeled']])
        self.Jn_train = self.Jn_all[subgraph_idx, :][:, subgraph_idx]
        # val
        subgraph_idx = torch.concatenate([idx['train'], idx['val'], idx['unlabeled']])
        # self.Jn_val = self.Jn_all[subgraph_idx, :][:, subgraph_idx]

    def test_all(self, edge_idx, attr_idx, n, d, test_idx):
        with torch.no_grad():
            adj_logits = self.Jn_all
            adj_logits = adj_logits[test_idx, :][:, test_idx]
            adj_orig = torch.sparse.FloatTensor(edge_idx, torch.ones(edge_idx.size(1), device=edge_idx.device),
                                                torch.Size([n, n])).to_dense()
            adj_orig = adj_orig[test_idx, :][:, test_idx]
            fpr, tpr, thresholds = roc_curve(adj_orig.view(-1).cpu().numpy(), adj_logits.view(-1).cpu().numpy())
            best_index = np.argmax(tpr - fpr)
            best_threshold = thresholds[best_index]
            adj_recons = (torch.sigmoid(adj_logits) > 0.65).float()
            acc = accuracy_score(adj_orig.view(-1).cpu().numpy(), adj_recons.view(-1).cpu().numpy())
            auc = roc_auc_score(adj_orig.view(-1).cpu().numpy(), adj_logits.view(-1).cpu().numpy())
            k = int(adj_orig.sum().item())
            RecK = sum(adj_orig.view(-1)[adj_logits.view(-1).argsort()[-k:]]) / adj_orig.sum()
            PreK = sum(adj_orig.view(-1)[adj_logits.view(-1).argsort()[-k:]]) / k
            print('acc,auc,reck,preK:', acc, auc, RecK.item(), PreK.item())
        return acc, auc, best_threshold

    def get_Jaccard_mask(self,threshold1, threshold2):
        Jn_mask = (self.Jn_all > threshold1) # mask for keeping the edge
        Jn_Aug_mask = (self.Jn_all > threshold2) # mask for adding the edge
        print("Jn_mask == True:",(Jn_mask.sum()/(self.n0_all**2)).item(), "%")
        print("Jn_Aug_mask == True:", (Jn_Aug_mask.sum()/(self.n0_all**2)).item(), "%")
        return Jn_mask, Jn_Aug_mask

    def set_jaccard_mask(self):
        J_mask, JAug_mask = self.get_Jaccard_mask(self.threshold1, self.threshold2)
        #testing
        J_mask_sparse = J_mask.to_sparse()
        JAug_mask_sparse = JAug_mask.to_sparse()
        self.JAug_mask_idx_all = JAug_mask_sparse.coalesce().indices()
        self.J_mask_idx_all = J_mask_sparse.coalesce().indices()

        #train
        subgraph_idx=torch.concatenate([self.idx['train'], self.idx['unlabeled']])
        self.n0_train = len(subgraph_idx)
        mask_temp = J_mask[subgraph_idx, :][:, subgraph_idx]
        self.J_mask_idx_train = mask_temp.to_sparse().coalesce().indices()
        mask_temp = JAug_mask[subgraph_idx,:][:,subgraph_idx]
        self.JAug_mask_idx_train = mask_temp.to_sparse().coalesce().indices()

        # val
        subgraph_idx = torch.concatenate([self.idx['train'], self.idx['val'], self.idx['unlabeled']])
        self.n0_val = len(subgraph_idx)
        mask_temp = J_mask[subgraph_idx, :][:, subgraph_idx]
        self.J_mask_idx_val = mask_temp.to_sparse().coalesce().indices()
        mask_temp = JAug_mask[subgraph_idx, :][:, subgraph_idx]
        self.JAug_mask_idx_val = mask_temp.to_sparse().coalesce().indices()

    def subgraph_mode(self,mode):
        if mode == 'train':
            self.J_mask_idx = self.J_mask_idx_train
            self.JAug_mask_idx = self.JAug_mask_idx_train
            self.n0 = self.n0_train
            # self.Jn = self.Jn_train
        elif mode == 'val':
            self.J_mask_idx = self.J_mask_idx_val
            self.JAug_mask_idx = self.JAug_mask_idx_val
            self.n0 = self.n0_val
            # self.Jn = self.Jn_val
        elif mode == 'test':
            self.J_mask_idx = self.J_mask_idx_all
            self.JAug_mask_idx = self.JAug_mask_idx_all
            self.n0 = self.n0_all
            # self.Jn = self.Jn_all
            # del self.Jn_all
            del self.Jn_train
            del self.J_mask_idx_val
            del self.JAug_mask_idx_val
            del self.J_mask_idx_train
            del self.JAug_mask_idx_train

    def batch_J_sparse(self,mask_idx,n,batch_size=1):
        # J_mask_repeat = self.J_mask.unsqueeze(0).repeat(batch_size, 1, 1)
        # J_mask_block_diag=torch.block_diag(*J_mask_repeat)
        # batch_J_mask_sparse = J_mask_block_diag.to_sparse()
        #----------------------------------------------------
        # # Another Version, avoid OOM, similar speed:
        if batch_size>1:
            batch_mask_idx = []
            for i in range(batch_size):
                batch_mask_idx.append(mask_idx + i * self.n0)
            batch_mask_idx = torch.cat(batch_mask_idx, dim=1)
            True_tensor = torch.ones(batch_mask_idx.size(1), dtype=torch.bool, device=batch_mask_idx.device)
            batch_mask_sparse = torch.sparse.FloatTensor(batch_mask_idx, True_tensor,
                                                   torch.Size([n, n]))
        else:
            True_tensor = torch.ones(mask_idx.size(1), dtype=torch.bool, device=mask_idx.device)
            batch_mask_sparse = torch.sparse.FloatTensor(mask_idx, True_tensor,
                                                         torch.Size([n, n]))
        return batch_mask_sparse



    def get_aug_edge_idx(self, edge_idx, n):
        batch_size = int(n / self.n0)
        edge_sparse = torch.sparse.FloatTensor(edge_idx, torch.ones(edge_idx.size(1), device=edge_idx.device),
                                               torch.Size([n, n]))
        #drop edges with Jaccard < threshold1
        batch_J_mask_drop = self.batch_J_sparse(mask_idx=self.J_mask_idx, n=n, batch_size=batch_size)
        edge_sparse = batch_J_mask_drop * edge_sparse
        # add edges with Jaccard > threshold2
        batch_J_mask_aug = self.batch_J_sparse(mask_idx=self.JAug_mask_idx, n=n, batch_size=batch_size)
        edge_mask_sparse = batch_J_mask_aug + edge_sparse
        edge_masked_idx = edge_mask_sparse.coalesce().indices()
        return edge_masked_idx

    def get_threshold(self, adj_logits, adj_orig, n, batch_size):
        N_prime = n
        E_prime = N_prime ** 2 * self.edge_proportion  # expected number of edge in the graph
        ADD = batch_size * int(E_prime * self.p_minus)  # number of edge to add
        DEL = batch_size * int((N_prime ** 2 - E_prime) * self.p_plus)  # number of edge to delete
        # Flatten the matrices
        adj_orig = adj_orig.fill_diagonal_(1)
        adj_logits_flat = adj_logits.view(-1)
        adj_orig_flat = adj_orig.clone().view(-1)

        if DEL == 0:
            lowest_k_value = 0.0
        else:
            # Get the indices of the lowest-k values in adj_logits
            adj_logits_flat_filter = adj_logits_flat.clone()
            adj_logits_flat_filter[adj_orig_flat == 0] = 100.0  # we don't want to delete the edge that is not exist
            lowest_k_values, lowest_k_indices = torch.topk(adj_logits_flat_filter, DEL, largest=False)
            lowest_k_value = lowest_k_values[-1].item()
        if ADD == 0:
            highest_l_value = 1.0
        else:
            # Get the indices of the highest-l values in adj_logits
            adj_logits_flat_filter = adj_logits_flat.clone()
            adj_logits_flat_filter[adj_orig_flat == 1] = -100.0  # we don't want to add the edge that is already exist
            highest_l_values, highest_l_indices = torch.topk(adj_logits_flat_filter, ADD, largest=True)
            highest_l_value = highest_l_values[-1].item()

        self.threshold1 = lowest_k_value
        self.threshold2 = highest_l_value
        self.seted_thresholds_by_topK = True
        print("threshold1:", self.threshold1, "threshold2:", self.threshold2)


    def set_daptive_thresholds(self,attr_idx, edge_idx, n, batch_size, d):
        assert n == self.Jn_train.shape[0]
        S_mask_repeat = self.Jn_train.unsqueeze(0).repeat(batch_size, 1, 1)
        adj_logits=torch.block_diag(*S_mask_repeat)
        adj_orig = torch.sparse.FloatTensor(edge_idx, torch.ones(edge_idx.size(1), device=edge_idx.device),
                                            torch.Size([n*batch_size, n*batch_size])).to_dense()
        self.get_threshold(adj_logits,adj_orig,n,batch_size)
        self.set_jaccard_mask()

    def forward(self, attr_idx, edge_idx, n, d): #default forward
        adj_sampled = self.get_aug_edge_idx(edge_idx, n)
        hidden = F.relu(self.conv1(attr_idx, adj_sampled, n, d))
        hidden = F.dropout(hidden, p=self.p_dropout, training=self.training)
        hidden = self.conv2(hidden, adj_sampled)
        return hidden

class FAEAug(nn.Module):
    """ AE/VAE as edge augmentation model """
    def __init__(self, dim_feats, dim_h, dim_z, ae=False):
        super(FAEAug, self).__init__()
        self.ae = ae  # AE or VAE
        self.fc_base = nn.Linear(dim_feats, dim_h)
        self.fc_mean = nn.Linear(dim_h, dim_z)
        self.fc_logstd = nn.Linear(dim_h, dim_z)
        self.EPS = 1e-15
        self.pos_weight = torch.FloatTensor([500])

    def forward(self, attr_idx, edge_idx, n, d):
        Z = self.encoder(attr_idx, edge_idx, n, d)
        adj_logits = Z @ Z.T  # dense adj matrix
        return adj_logits

    def encoder(self,attr_idx, edge_idx, n, d):
        attr = torch.sparse.FloatTensor(attr_idx, torch.ones(attr_idx.size(1), device=attr_idx.device),
                                        torch.Size([n, d])).to_dense()
        # Fully connected encoder
        hidden = F.relu(self.fc_base(attr))
        self.mean = F.relu(self.fc_mean(hidden))#relu makes the logits stay postive and sparse
        if self.ae:
            Z = self.mean
        else:
            # VAE
            self.logstd = self.fc_logstd(hidden)
            gaussian_noise = torch.randn_like(self.mean)
            sampled_Z = gaussian_noise * torch.exp(self.logstd) + self.mean
            Z = sampled_Z
        return Z

    def decoder(self, z, edge_index, sigmoid):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def kl_loss(self, mu=None, logstd=None):
        mu = self.mean if mu is None else mu
        logstd = self.logstd if logstd is None else logstd.clamp(max=10)
        return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

    def recon_loss(self, z, pos_edge_index, n=None, neg_edge_index=None):
        pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + self.EPS).mean()
        if neg_edge_index is None:
            pos_edge_index, _ = remove_self_loops(pos_edge_index)
            pos_edge_index, _ = add_self_loops(pos_edge_index)
            neg_edge_index = negative_sampling(pos_edge_index, num_nodes=z.size(0), num_neg_samples=10000)
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + self.EPS).mean()
        loss = pos_loss + neg_loss
        if not self.ae:
            loss = loss + (1 / n).to(loss.device) * self.kl_loss(self.mean, self.logstd)
        return loss

    def loss(self, attr_idx, edge_idx, n, d):
        adj_logits = self.forward(attr_idx, edge_idx, n, d)
        adj_ture = torch.sparse.FloatTensor(edge_idx, torch.ones(edge_idx.size(1), device=edge_idx.device),
                                        torch.Size([n, n])).to_dense()
        loss = F.binary_cross_entropy_with_logits(adj_logits, adj_ture, pos_weight=self.pos_weight.to(edge_idx.device))
        if not self.ae:
            loss = loss + (1 / n).to(loss.device) * self.kl_loss(self.mean, self.logstd)
        return loss

    def test(self, z, pos_edge_index, neg_edge_index):
        with torch.no_grad():
            pos_y = z.new_ones(pos_edge_index.size(1))
            neg_y = z.new_zeros(neg_edge_index.size(1))
            y = torch.cat([pos_y, neg_y], dim=0)
            pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
            neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
            pred = torch.cat([pos_pred, neg_pred], dim=0)
            y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
            # fpr, tpr, thresholds=roc_curve(y,pred)
            # best_index = np.argmax(tpr - fpr)
            # best_threshold = thresholds[best_index]
        return roc_auc_score(y, pred)

    def test_all(self,edge_idx,attr_idx, n, d, test_idx):
        with torch.no_grad():
            Z = self.encoder(attr_idx, edge_idx, n, d)
            adj_logits = Z @ Z.T
            adj_orig = torch.sparse.FloatTensor(edge_idx, torch.ones(edge_idx.size(1), device=edge_idx.device),
                                            torch.Size([n, n])).to_dense()
            adj_orig = adj_orig[test_idx, :][:, test_idx]
            adj_logits = adj_logits[test_idx, :][:, test_idx]
            fpr, tpr, thresholds = roc_curve(adj_orig.view(-1).cpu().numpy(), adj_logits.view(-1).cpu().numpy())
            best_index = np.argmax(tpr - fpr)
            best_threshold = thresholds[best_index]
            adj_recons = (torch.sigmoid(adj_logits) > 0.65).float()
            acc = accuracy_score(adj_orig.view(-1).cpu().numpy(), adj_recons.view(-1).cpu().numpy())
            auc = roc_auc_score(adj_orig.view(-1).cpu().numpy(), adj_logits.view(-1).cpu().numpy())
            k = int(adj_orig.sum().item())
            RecK = sum(adj_orig.view(-1)[adj_logits.view(-1).argsort()[-k:]]) / adj_orig.sum()
            PreK = sum(adj_orig.view(-1)[adj_logits.view(-1).argsort()[-k:]]) / k
            print('acc,auc,reck,preK:', acc, auc, RecK.item(), PreK.item())
        return acc,auc,best_threshold

class FAugGCN(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden, dim_z,edge_proportion,p_plus, p_minus, p_dropout=0.5):
        super().__init__()
        self.nc_net = GCN(n_features, n_classes, n_hidden, p_dropout)  # node classification network
        self.ep_net = FAEAug(n_features, n_hidden, dim_z, ae=True) # edge prediction network
        self.threshold1 = 0.1
        self.threshold2 = 0.5
        self.edge_proportion = edge_proportion  # (A-I).sum()/(N**2)
        self.p_plus = p_plus
        self.p_minus = p_minus
        self.seted_thresholds_by_topK = False


    def set_adj_logit(self,attr_idx, edge_idx, n, d, idx):
        '''pre calculate the adj_logit to avoid repeated computation'''
        self.S_all = self.ep_net(attr_idx, edge_idx, n, d)

        self.idx=idx
        self.n0_all = n
        #train
        subgraph_idx=np.concatenate([idx['train'], idx['unlabeled']])
        self.S_train = self.S_all[subgraph_idx, :][:, subgraph_idx]
        # val
        # subgraph_idx = np.concatenate([idx['train'], idx['val'], idx['unlabeled']])
        # self.S_val = self.S_all[subgraph_idx, :][:, subgraph_idx]

    def get_mask(self,threshold1, threshold2):
        S_mask1 = (self.S_all > threshold1) # mask for keeping the edge
        S_mask2 = (self.S_all > threshold2) # mask for adding the edge
        print("S_mask1 == True:",(S_mask1.sum()/(self.n0_all**2)).item(), "%")
        print("S_mask2 == True:", (S_mask2.sum()/(self.n0_all**2)).item(), "%")
        return S_mask1, S_mask2

    def set_mask(self):
        S_mask1, S_mask2 = self.get_mask(self.threshold1, self.threshold2)
        #testing
        S_mask1_sparse = S_mask1.to_sparse()
        S_mask2_sparse = S_mask2.to_sparse()
        self.S_mask2_idx_all = S_mask2_sparse.coalesce().indices()
        self.S_mask1_idx_all = S_mask1_sparse.coalesce().indices()

        #train
        subgraph_idx=np.concatenate([self.idx['train'], self.idx['unlabeled']])
        self.n0_train = len(subgraph_idx)
        mask_temp = S_mask1[subgraph_idx, :][:, subgraph_idx]
        self.S_mask1_idx_train = mask_temp.to_sparse().coalesce().indices()
        mask_temp = S_mask2[subgraph_idx,:][:,subgraph_idx]
        self.S_mask2_idx_train = mask_temp.to_sparse().coalesce().indices()

        # val
        subgraph_idx = np.concatenate([self.idx['train'], self.idx['val'], self.idx['unlabeled']])
        self.n0_val = len(subgraph_idx)
        mask_temp = S_mask1[subgraph_idx, :][:, subgraph_idx]
        self.S_mask1_idx_val = mask_temp.to_sparse().coalesce().indices()
        mask_temp = S_mask2[subgraph_idx, :][:, subgraph_idx]
        self.S_mask2_idx_val = mask_temp.to_sparse().coalesce().indices()

    def batch_mask_sparse(self,mask_idx,n,batch_size=1):
        batch_mask_idx = []
        for i in range(batch_size):
            batch_mask_idx.append(mask_idx + i * self.n0)
        batch_mask_idx = torch.cat(batch_mask_idx, dim=1)
        True_tensor = torch.ones(batch_mask_idx.size(1), dtype=torch.bool, device=batch_mask_idx.device)
        batch_mask_sparse = torch.sparse.FloatTensor(batch_mask_idx, True_tensor,
                                               torch.Size([n, n]))
        return batch_mask_sparse

    def get_aug_edge_idx(self, edge_idx, n):
        batch_size = int(n / self.n0)
        edge_sparse = torch.sparse.FloatTensor(edge_idx, torch.ones(edge_idx.size(1), device=edge_idx.device),
                                               torch.Size([n, n]))
        #drop edges with Jaccard < threshold1
        batch_S_mask_drop = self.batch_mask_sparse(mask_idx=self.S_mask1_idx, n=n,batch_size=batch_size)
        edge_sparse = batch_S_mask_drop * edge_sparse
        # add edges with Jaccard > threshold2
        batch_S_mask_aug = self.batch_mask_sparse(mask_idx=self.S_mask2_idx, n=n, batch_size=batch_size)
        edge_mask_sparse = batch_S_mask_aug + edge_sparse
        edge_masked_idx = edge_mask_sparse.coalesce().indices()
        return edge_masked_idx

    def aug_adj_threshold(self,adj_logits,adj_orig):
        keep_adj = adj_logits > self.threshold1
        aug_adj = adj_orig * keep_adj
        add_adj = adj_logits > self.threshold2
        aug_adj = aug_adj + add_adj
        aug_adj = aug_adj.clamp(0, 1)
        # print("add_adj == True:", add_adj.sum()/(add_adj.shape[0]**2), "%")
        return aug_adj

    def get_threshold(self, adj_logits, adj_orig, n, batch_size):
        N_prime = n
        E_prime = N_prime ** 2 * self.edge_proportion  # expected number of edge in the graph
        ADD = batch_size * int(E_prime * self.p_minus)  # number of edge to add
        DEL = batch_size * int((N_prime ** 2 - E_prime) * self.p_plus)  # number of edge to delete
        # Flatten the matrices
        adj_orig = adj_orig.fill_diagonal_(1)
        adj_logits_flat = adj_logits.view(-1)
        adj_orig_flat = adj_orig.clone().view(-1)

        if DEL == 0:
            lowest_k_value = 0.0
        else:
            # Get the indices of the lowest-k values in adj_logits
            adj_logits_flat_filter = adj_logits_flat.clone()
            adj_logits_flat_filter[adj_orig_flat == 0] = 100.0  # we don't want to delete the edge that is not exist
            lowest_k_values, lowest_k_indices = torch.topk(adj_logits_flat_filter, DEL, largest=False)
            lowest_k_value = lowest_k_values[-1].item()
        if ADD == 0:
            highest_l_value = 1.0
        else:
            # Get the indices of the highest-l values in adj_logits
            adj_logits_flat_filter = adj_logits_flat.clone()
            adj_logits_flat_filter[adj_orig_flat == 1] = -100.0  # we don't want to add the edge that is already exist
            highest_l_values, highest_l_indices = torch.topk(adj_logits_flat_filter, ADD, largest=True)
            highest_l_value = highest_l_values[-1].item()

        self.threshold1 = lowest_k_value
        self.threshold2 = highest_l_value
        self.seted_thresholds_by_topK = True
        print("threshold1:", self.threshold1, "threshold2:", self.threshold2)


    def set_daptive_thresholds(self,attr_idx, edge_idx, n, batch_size, d):
        assert n == self.S_train.shape[0]
        S_mask_repeat = self.S_train.unsqueeze(0).repeat(batch_size, 1, 1)
        adj_logits=torch.block_diag(*S_mask_repeat)
        adj_orig = torch.sparse.FloatTensor(edge_idx, torch.ones(edge_idx.size(1), device=edge_idx.device),
                                            torch.Size([n*batch_size, n*batch_size])).to_dense()
        self.get_threshold(adj_logits,adj_orig,n,batch_size)
        self.set_mask()

    def subgraph_mode(self,mode):
        if mode == 'train':
            self.S_mask1_idx = self.S_mask1_idx_train
            self.S_mask2_idx = self.S_mask2_idx_train
            self.n0 = self.n0_train
            # self.S = self.S_train
        elif mode == 'val':
            self.S_mask1_idx = self.S_mask1_idx_val
            self.S_mask2_idx = self.S_mask2_idx_val
            self.n0 = self.n0_val
            # self.S = self.S_val
        elif mode == 'test':
            self.S_mask1_idx = self.S_mask1_idx_all
            self.S_mask2_idx = self.S_mask2_idx_all
            self.n0 = self.n0_all
            # self.S = self.S_all

            del self.S_mask1_idx_train
            del self.S_mask2_idx_train
            del self.S_mask1_idx_val
            del self.S_mask2_idx_val
            del self.S_train


    def forward(self, attr_idx, edge_idx, n, d): #default forward
        # adj_aug = self.aug_adj(adj_logits, adj_orig)
        # adj_aug = self.aug_adj_threshold(adj_logits, adj_orig)
        # adj_aug = adj_aug.to_sparse().indices()
        adj_aug = self.get_aug_edge_idx(edge_idx, n)
        hidden = self.nc_net(attr_idx, adj_aug, n, d)
        return hidden


class SimAug(nn.Module):
    def __init__(self, input_size, num_wegihts):
        super(SimAug, self).__init__()
        self.weight_tensor = torch.Tensor(num_wegihts, input_size)
        self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))
        self.EPS = 1e-15
        self.pos_weight = torch.FloatTensor([500])

    def build_epsilon_neighbourhood(self, adj_logit, threshold1, threshold2):
        keep = (adj_logit > epsilon).detach().float()
        weighted_adjacency_matrix = adj_logit * keep + threshold2 * (1 - keep)
        return weighted_adjacency_matrix

    def encoder(self, attr_idx, edge_idx, n, d):
        attr = torch.sparse.FloatTensor(attr_idx, torch.ones(attr_idx.size(1), device=attr_idx.device),
                                        torch.Size([n, d])).to_dense()
        expand_weight_tensor = self.weight_tensor.unsqueeze(1)
        attr_weighted = attr.unsqueeze(0) * expand_weight_tensor
        Z = F.normalize(attr_weighted, p=2, dim=-1)
        return Z

    def forward(self, attr_idx, edge_idx, n, d):
        Z = self.encoder(attr_idx, edge_idx, n, d)
        adj_logits = torch.matmul(Z, Z.transpose(-1, -2)).mean(0)  # dense adj matrix
        return adj_logits

    def decoder(self, z, edge_index, sigmoid):
        value = (z[:,edge_index[0],:] * z[:,edge_index[1],:]).mean(0).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def recon_loss(self, z, pos_edge_index, n=None, neg_edge_index=None):
        pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=False) + self.EPS).mean()
        if neg_edge_index is None:
            pos_edge_index, _ = remove_self_loops(pos_edge_index)
            pos_edge_index, _ = add_self_loops(pos_edge_index)
            neg_edge_index = negative_sampling(pos_edge_index, num_nodes=z.size(1), num_neg_samples=10000)
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=False) + self.EPS).mean()
        loss = pos_loss + neg_loss
        return loss

    def loss(self, attr_idx, edge_idx, n, d):
        adj_logits = self.forward(attr_idx, edge_idx, n, d)
        adj_ture = torch.sparse.FloatTensor(edge_idx, torch.ones(edge_idx.size(1), device=edge_idx.device),
                                            torch.Size([n, n])).to_dense()
        loss = F.binary_cross_entropy_with_logits(adj_logits, adj_ture, pos_weight=self.pos_weight.to(adj.device))
        return loss

    def test(self, z, pos_edge_index, neg_edge_index):
        with torch.no_grad():
            pos_y = z.new_ones(pos_edge_index.size(1))
            neg_y = z.new_zeros(neg_edge_index.size(1))
            y = torch.cat([pos_y, neg_y], dim=0)
            pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
            neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
            pred = torch.cat([pos_pred, neg_pred], dim=0)
            y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
            # fpr, tpr, thresholds=roc_curve(y,pred)
            # best_index = np.argmax(tpr - fpr)
            # best_threshold = thresholds[best_index]
        return roc_auc_score(y, pred)

    def test_all(self,edge_idx,attr_idx, n, d, test_idx):
        with torch.no_grad():
            adj_logits = self.forward(attr_idx, edge_idx, n, d)
            adj_orig = torch.sparse.FloatTensor(edge_idx, torch.ones(edge_idx.size(1), device=edge_idx.device),torch.Size([n, n])).to_dense()
            adj_orig = adj_orig[test_idx, :][:, test_idx]
            adj_logits = adj_logits[test_idx, :][:, test_idx]
            fpr, tpr, thresholds = roc_curve(adj_orig.view(-1).cpu().numpy(), adj_logits.view(-1).cpu().numpy())
            best_index = np.argmax(tpr - fpr)
            best_threshold = thresholds[best_index]
            adj_recons = (torch.sigmoid(adj_logits) > 0.65).float()
            acc = accuracy_score(adj_orig.view(-1).cpu().numpy(), adj_recons.view(-1).cpu().numpy())
            auc = roc_auc_score(adj_orig.view(-1).cpu().numpy(), adj_logits.view(-1).cpu().numpy())
            k = int(adj_orig.sum().item())
            RecK = sum(adj_orig.view(-1)[adj_logits.view(-1).argsort()[-k:]]) / adj_orig.sum()
            PreK = sum(adj_orig.view(-1)[adj_logits.view(-1).argsort()[-k:]]) / k
            print('acc,auc,reck,preK:', acc, auc, RecK.item(), PreK.item())
        return acc,auc,best_threshold

class SAugGCN(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden, num_wegihts,edge_proportion,p_plus, p_minus,p_dropout=0.5):
        super().__init__()
        self.nc_net = GCN(n_features, n_classes, n_hidden, p_dropout)  # node classification network
        self.ep_net = SimAug(n_features, num_wegihts) # edge prediction network
        self.threshold1 = 0.6
        self.threshold2 = 0.995
        self.edge_proportion=edge_proportion #(A-I).sum()/(N**2)
        self.p_plus=p_plus
        self.p_minus=p_minus

    def set_adj_logit(self,attr_idx, edge_idx, n, d, idx):
        '''pre calculate the adj_logit to avoid repeated computation'''
        self.S_all = self.ep_net(attr_idx, edge_idx, n, d)

        self.idx=idx
        self.n0_all = n
        #train
        subgraph_idx=np.concatenate([idx['train'], idx['unlabeled']])
        self.S_train = self.S_all[subgraph_idx, :][:, subgraph_idx]
        # val
        subgraph_idx = np.concatenate([idx['train'], idx['val'], idx['unlabeled']])
        # self.S_val = self.S_all[subgraph_idx, :][:, subgraph_idx]

    def get_mask(self,threshold1, threshold2):
        S_mask1 = (self.S_all > threshold1) # mask for keeping the edge
        S_mask2 = (self.S_all > threshold2) # mask for adding the edge
        print("S_mask1 == True:",(S_mask1.sum()/(self.n0_all**2)).item(), "%")
        print("S_mask2 == True:", (S_mask2.sum()/(self.n0_all**2)).item(), "%")
        return S_mask1, S_mask2

    def set_mask(self):
        S_mask1, S_mask2 = self.get_mask(self.threshold1, self.threshold2)
        #testing
        self.S_mask1_sparse = S_mask1.to_sparse()
        self.S_mask2_sparse = S_mask2.to_sparse()
        self.S_mask2_idx_all = self.S_mask2_sparse.coalesce().indices()
        self.S_mask1_idx_all = self.S_mask1_sparse.coalesce().indices()

        #train
        subgraph_idx=np.concatenate([self.idx['train'], self.idx['unlabeled']])
        self.n0_train = len(subgraph_idx)
        mask_temp = S_mask1[subgraph_idx, :][:, subgraph_idx]
        self.S_mask1_idx_train = mask_temp.to_sparse().coalesce().indices()
        mask_temp = S_mask2[subgraph_idx,:][:,subgraph_idx]
        self.S_mask2_idx_train = mask_temp.to_sparse().coalesce().indices()

        # val
        subgraph_idx = np.concatenate([self.idx['train'], self.idx['val'], self.idx['unlabeled']])
        self.n0_val = len(subgraph_idx)
        mask_temp = S_mask1[subgraph_idx, :][:, subgraph_idx]
        self.S_mask1_idx_val = mask_temp.to_sparse().coalesce().indices()
        mask_temp = S_mask2[subgraph_idx, :][:, subgraph_idx]
        self.S_mask2_idx_val = mask_temp.to_sparse().coalesce().indices()

    def batch_mask_sparse(self,mask_idx,n,batch_size=1):
        batch_mask_idx = []
        for i in range(batch_size):
            batch_mask_idx.append(mask_idx + i * self.n0)
        batch_mask_idx = torch.cat(batch_mask_idx, dim=1)
        True_tensor = torch.ones(batch_mask_idx.size(1), dtype=torch.bool, device=batch_mask_idx.device)
        batch_mask_sparse = torch.sparse.FloatTensor(batch_mask_idx, True_tensor,
                                               torch.Size([n, n]))
        return batch_mask_sparse

    def get_aug_edge_idx(self, edge_idx, n):
        batch_size = int(n / self.n0)
        edge_sparse = torch.sparse.FloatTensor(edge_idx, torch.ones(edge_idx.size(1), device=edge_idx.device),
                                               torch.Size([n, n]))
        #drop edges with Jaccard < threshold1
        batch_S_mask_drop = self.batch_mask_sparse(mask_idx=self.S_mask1_idx, n=n,batch_size=batch_size)
        edge_sparse = batch_S_mask_drop * edge_sparse
        # add edges with Jaccard > threshold2
        batch_S_mask_aug = self.batch_mask_sparse(mask_idx=self.S_mask2_idx, n=n, batch_size=batch_size)
        edge_mask_sparse = batch_S_mask_aug + edge_sparse
        edge_masked_idx = edge_mask_sparse.coalesce().indices()
        return edge_masked_idx

    def aug_adj_threshold(self,adj_logits,adj_orig):
        keep_adj = adj_logits > self.threshold1
        aug_adj = adj_orig * keep_adj
        add_adj = adj_logits > self.threshold2
        aug_adj = aug_adj + add_adj
        aug_adj = aug_adj.clamp(0, 1)
        # print("add_adj == True:", add_adj.sum()/(add_adj.shape[0]**2), "%")
        return aug_adj

    def get_threshold(self, adj_logits, adj_orig, n, batch_size):
        N_prime = n
        E_prime = N_prime ** 2 * self.edge_proportion  # expected number of edge in the graph
        ADD = batch_size * int(E_prime * self.p_minus)  # number of edge to add
        DEL = batch_size * int((N_prime ** 2 - E_prime) * self.p_plus)  # number of edge to delete
        # Flatten the matrices
        adj_orig = adj_orig.fill_diagonal_(1)
        adj_logits_flat = adj_logits.view(-1)
        adj_orig_flat = adj_orig.clone().view(-1)

        if DEL == 0:
            lowest_k_value = 0.0
        else:
            # Get the indices of the lowest-k values in adj_logits
            adj_logits_flat_filter = adj_logits_flat.clone()
            adj_logits_flat_filter[adj_orig_flat == 0] = 100.0  # we don't want to delete the edge that is not exist
            lowest_k_values, lowest_k_indices = torch.topk(adj_logits_flat_filter, DEL, largest=False)
            lowest_k_value = lowest_k_values[-1].item()
        if ADD == 0:
            highest_l_value = 1.0
        else:
            # Get the indices of the highest-l values in adj_logits
            adj_logits_flat_filter = adj_logits_flat.clone()
            adj_logits_flat_filter[adj_orig_flat == 1] = -100.0  # we don't want to add the edge that is already exist
            highest_l_values, highest_l_indices = torch.topk(adj_logits_flat_filter, ADD, largest=True)
            highest_l_value = highest_l_values[-1].item()

        self.threshold1 = lowest_k_value
        self.threshold2 = highest_l_value
        self.seted_thresholds_by_topK = True
        print("threshold1:", self.threshold1, "threshold2:", self.threshold2)


    def set_daptive_thresholds(self,attr_idx, edge_idx, n, batch_size, d):
        assert n == self.S_train.shape[0]
        S_mask_repeat = self.S_train.unsqueeze(0).repeat(batch_size, 1, 1)
        adj_logits=torch.block_diag(*S_mask_repeat)
        adj_orig = torch.sparse.FloatTensor(edge_idx, torch.ones(edge_idx.size(1), device=edge_idx.device),
                                            torch.Size([n*batch_size, n*batch_size])).to_dense()
        self.get_threshold(adj_logits,adj_orig,n,batch_size)
        self.set_mask()

    def subgraph_mode(self,mode):
        if mode == 'train':
            self.S_mask1_idx = self.S_mask1_idx_train
            self.S_mask2_idx = self.S_mask2_idx_train
            self.n0 = self.n0_train
            # self.S = self.S_train
        elif mode == 'val':
            self.S_mask1_idx = self.S_mask1_idx_val
            self.S_mask2_idx = self.S_mask2_idx_val
            self.n0 = self.n0_val
            # self.S = self.S_val
        elif mode == 'test':
            self.S_mask1_idx = self.S_mask1_idx_all
            self.S_mask2_idx = self.S_mask2_idx_all
            self.n0 = self.n0_all
            # self.S = self.S_all

            del self.S_mask1_idx_train
            del self.S_mask2_idx_train
            del self.S_mask1_idx_val
            del self.S_mask2_idx_val
            del self.S_train


    def forward(self, attr_idx, edge_idx, n, d): #default forward
        adj_aug = self.get_aug_edge_idx(edge_idx, n)
        hidden = self.nc_net(attr_idx, adj_aug, n, d)
        return hidden


class AuditVotes(nn.Module):
    def __init__(self, nc_net,ep_net,edge_proportion,p_plus, p_minus):
        super().__init__()
        self.nc_net = nc_net  # node classification network
        self.ep_net = ep_net # edge prediction network
        self.edge_proportion=edge_proportion #(A-I).sum()/(N**2)
        self.p_plus=p_plus
        self.p_minus=p_minus
        self.threshold1 = 0.6
        self.threshold2 = 0.995
        self.seted_thresholds_by_topK = False

    def set_adj_logit(self,attr_idx, edge_idx, n, d, idx):
        '''pre calculate the adj_logit to avoid repeated computation'''
        self.S_all = self.ep_net(attr_idx, edge_idx, n, d)

        self.idx=idx
        self.n0_all = n
        #train
        subgraph_idx=np.concatenate([idx['train'], idx['unlabeled']])
        self.S_train = self.S_all[subgraph_idx, :][:, subgraph_idx]
        # val
        subgraph_idx = np.concatenate([idx['train'], idx['val'], idx['unlabeled']])
        # self.S_val = self.S_all[subgraph_idx, :][:, subgraph_idx]

    def get_mask(self,threshold1, threshold2):
        S_mask1 = (self.S_all > threshold1) # mask for keeping the edge
        S_mask2 = (self.S_all > threshold2) # mask for adding the edge
        print("S_mask1 == True:",(S_mask1.sum()/(self.n0_all**2)).item(), "%")
        print("S_mask2 == True:", (S_mask2.sum()/(self.n0_all**2)).item(), "%")
        return S_mask1, S_mask2

    def set_mask(self):
        S_mask1, S_mask2 = self.get_mask(self.threshold1, self.threshold2)
        #testing
        self.S_mask1_sparse = S_mask1.to_sparse()
        self.S_mask2_sparse = S_mask2.to_sparse()
        self.S_mask2_idx_all = self.S_mask2_sparse.coalesce().indices()
        self.S_mask1_idx_all = self.S_mask1_sparse.coalesce().indices()

        #train
        subgraph_idx=np.concatenate([self.idx['train'], self.idx['unlabeled']])
        self.n0_train = len(subgraph_idx)
        mask_temp = S_mask1[subgraph_idx, :][:, subgraph_idx]
        self.S_mask1_idx_train = mask_temp.to_sparse().coalesce().indices()
        mask_temp = S_mask2[subgraph_idx,:][:,subgraph_idx]
        self.S_mask2_idx_train = mask_temp.to_sparse().coalesce().indices()

        # val
        subgraph_idx = np.concatenate([self.idx['train'], self.idx['val'], self.idx['unlabeled']])
        self.n0_val = len(subgraph_idx)
        mask_temp = S_mask1[subgraph_idx, :][:, subgraph_idx]
        self.S_mask1_idx_val = mask_temp.to_sparse().coalesce().indices()
        mask_temp = S_mask2[subgraph_idx, :][:, subgraph_idx]
        self.S_mask2_idx_val = mask_temp.to_sparse().coalesce().indices()

    def batch_mask_sparse(self,mask_idx,n,batch_size=1):
        batch_mask_idx = []
        for i in range(batch_size):
            batch_mask_idx.append(mask_idx + i * self.n0)
        batch_mask_idx = torch.cat(batch_mask_idx, dim=1)
        True_tensor = torch.ones(batch_mask_idx.size(1), dtype=torch.bool, device=batch_mask_idx.device)
        batch_mask_sparse = torch.sparse.FloatTensor(batch_mask_idx, True_tensor,
                                               torch.Size([n, n]))
        return batch_mask_sparse

    def get_aug_edge_idx(self, edge_idx, n):
        batch_size = int(n / self.n0)
        edge_sparse = torch.sparse.FloatTensor(edge_idx, torch.ones(edge_idx.size(1), device=edge_idx.device),
                                               torch.Size([n, n]))
        #drop edges with Jaccard < threshold1
        batch_S_mask_drop = self.batch_mask_sparse(mask_idx=self.S_mask1_idx, n=n,batch_size=batch_size)
        edge_sparse = batch_S_mask_drop * edge_sparse
        # add edges with Jaccard > threshold2
        batch_S_mask_aug = self.batch_mask_sparse(mask_idx=self.S_mask2_idx, n=n, batch_size=batch_size)
        edge_mask_sparse = batch_S_mask_aug + edge_sparse
        edge_masked_idx = edge_mask_sparse.coalesce().indices()
        return edge_masked_idx

    def aug_adj_threshold(self,adj_logits,adj_orig):
        keep_adj = adj_logits > self.threshold1
        aug_adj = adj_orig * keep_adj
        add_adj = adj_logits > self.threshold2
        aug_adj = aug_adj + add_adj
        aug_adj = aug_adj.clamp(0, 1)
        # print("add_adj == True:", add_adj.sum()/(add_adj.shape[0]**2), "%")
        return aug_adj

    def get_threshold(self, adj_logits, adj_orig, n, batch_size):
        N_prime = n
        E_prime = N_prime ** 2 * self.edge_proportion  # expected number of edge in the graph
        ADD = batch_size * int(E_prime * self.p_minus)  # number of edge to add
        DEL = batch_size * int((N_prime ** 2 - E_prime) * self.p_plus)  # number of edge to delete
        # Flatten the matrices
        adj_orig = adj_orig.fill_diagonal_(1)
        adj_logits_flat = adj_logits.view(-1)
        adj_orig_flat = adj_orig.clone().view(-1)

        if DEL == 0:
            lowest_k_value = 0.0
        else:
            # Get the indices of the lowest-k values in adj_logits
            adj_logits_flat_filter = adj_logits_flat.clone()
            adj_logits_flat_filter[adj_orig_flat == 0] = 100.0  # we don't want to delete the edge that is not exist
            lowest_k_values, lowest_k_indices = torch.topk(adj_logits_flat_filter, DEL, largest=False)
            lowest_k_value = lowest_k_values[-1].item()
        if ADD == 0:
            highest_l_value = 1.0
        else:
            # Get the indices of the highest-l values in adj_logits
            adj_logits_flat_filter = adj_logits_flat.clone()
            adj_logits_flat_filter[adj_orig_flat == 1] = -100.0  # we don't want to add the edge that is already exist
            highest_l_values, highest_l_indices = torch.topk(adj_logits_flat_filter, ADD, largest=True)
            highest_l_value = highest_l_values[-1].item()

        self.threshold1 = lowest_k_value
        self.threshold2 = highest_l_value
        self.seted_thresholds_by_topK = True
        print("threshold1:", self.threshold1, "threshold2:", self.threshold2)


    def set_daptive_thresholds(self,attr_idx, edge_idx, n, batch_size, d):
        assert n == self.S_train.shape[0]
        S_mask_repeat = self.S_train.unsqueeze(0).repeat(batch_size, 1, 1)
        adj_logits=torch.block_diag(*S_mask_repeat)
        adj_orig = torch.sparse.FloatTensor(edge_idx, torch.ones(edge_idx.size(1), device=edge_idx.device),
                                            torch.Size([n*batch_size, n*batch_size])).to_dense()
        self.get_threshold(adj_logits,adj_orig,n,batch_size)
        self.set_mask()

    def subgraph_mode(self,mode):
        if mode == 'train':
            self.S_mask1_idx = self.S_mask1_idx_train
            self.S_mask2_idx = self.S_mask2_idx_train
            self.n0 = self.n0_train
            # self.S = self.S_train
        elif mode == 'val':
            self.S_mask1_idx = self.S_mask1_idx_val
            self.S_mask2_idx = self.S_mask2_idx_val
            self.n0 = self.n0_val
            # self.S = self.S_val
        elif mode == 'test':
            self.S_mask1_idx = self.S_mask1_idx_all
            self.S_mask2_idx = self.S_mask2_idx_all
            self.n0 = self.n0_all
            # self.S = self.S_all
            del self.S_mask1_idx_train
            del self.S_mask2_idx_train
            del self.S_mask1_idx_val
            del self.S_mask2_idx_val
            del self.S_train

    def forward(self, attr_idx, edge_idx, n, d): #default forward
        adj_aug = self.get_aug_edge_idx(edge_idx, n)
        hidden = self.nc_net(attr_idx, adj_aug, n, d)
        return hidden







