import random
import os
import numpy as np
import math
from collections import Counter

import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from deeprobust.graph.utils import sparse_mx_to_torch_sparse_tensor
from torch_sparse import SparseTensor
from torch_geometric.utils import get_laplacian
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.nn.conv.gcn_conv import gcn_norm

def sparse_eye(n):
    eye = sp.eye(n).tocoo()
    eye = sparse_mx_to_torch_sparse_tensor(eye).float()
    return eye

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def aug_normalized_adjacency(adj, aug=True):
    if aug:
        adj = adj + torch.eye(adj.size(0), device=adj.device)



    adj_sparse = adj

    row_sum = adj_sparse.sum(dim=1)

    d_inv_sqrt = torch.pow(row_sum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0

    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)

    normalized_adj = d_mat_inv_sqrt @ adj_sparse @ d_mat_inv_sqrt
    return normalized_adj






def init_params(module):
    if isinstance(module, nn.Linear):
        stdv = 1.0 / math.sqrt(module.weight.size(1))
        module.weight.data.uniform_(-stdv, stdv)
        if module.bias is not None:
            module.bias.data.uniform_(-stdv, stdv)
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


def normalize_features(mx):
     rowsum = mx.sum(1)
     r_inv = torch.pow(rowsum, -1)
     r_inv[torch.isinf(r_inv)] = 0.
     r_mat_inv = torch.diag(r_inv)
     mx = r_mat_inv @ mx
     return mx


def normalize_adj(mx):
    """Normalize sparse adjacency matrix,
    A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    """
    if type(mx) is not sp.lil.lil_matrix:
        mx = mx.tolil()
    mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    
    return mx



def mmd_rbf(x, y, sigma=0.2):

    # x = torch.tensor(x, dtype=torch.float)
    # y = torch.tensor(y, dtype=torch.float)

    xx = torch.cdist(x, x, p=2)**2
    yy = torch.cdist(y, y, p=2)**2
    xy = torch.cdist(x, y, p=2)**2

    k_xx = torch.exp(-xx / (2 * sigma**2))
    k_yy = torch.exp(-yy / (2 * sigma**2))
    k_xy = torch.exp(-xy / (2 * sigma**2))

    mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    return torch.sqrt(mmd).item()

def normalize_adj_to_sparse_tensor(mx):
    mx = normalize_adj(mx)
    mx = sparse_mx_to_torch_sparse_tensor(mx)
    sparsetensor = SparseTensor(row=mx._indices()[0], col=mx._indices()[1], value=mx._values(), sparse_sizes=mx.size()).cuda()
    return sparsetensor


def get_syn_eigen(real_eigenvals, real_eigenvecs, eigen_k, ratio, step=1):
    k1 = math.ceil(eigen_k * ratio)
    k2 = eigen_k - k1
    print("k1:", k1, ",", "k2:", k2)
    k1_end = (k1 - 1) * step + 1
    eigen_sum = real_eigenvals.shape[0]
    k2_end = eigen_sum - (k2 - 1) * step - 1
    k1_list = range(0, k1_end, step)
    k2_list = range(k2_end, eigen_sum, step)
    eigenvals = torch.cat(
        [real_eigenvals[k1_list], real_eigenvals[k2_list]]
    )
    eigenvecs = torch.cat(
        [real_eigenvecs[:, k1_list], real_eigenvecs[:, k2_list]], dim=1,
    )
    
    return eigenvals, eigenvecs


def get_subspace_embed(eigenvecs, x):
    x_trans = eigenvecs.T @ x  # kd
    u_unsqueeze = (eigenvecs.T).unsqueeze(2) # kn1
    x_trans_unsqueeze = x_trans.unsqueeze(1) # k1d
    sub_embed = torch.bmm(u_unsqueeze, x_trans_unsqueeze)  # kn1 @ k1d = knd
    return x_trans, sub_embed


def get_subspace_covariance_matrix(eigenvecs, x):
    x_trans = eigenvecs.T @ x  # kd
    x_trans = F.normalize(input=x_trans, p=2, dim=1)
    x_trans_unsqueeze = x_trans.unsqueeze(1)  # k1d
    co_matrix = torch.bmm(x_trans_unsqueeze.permute(0, 2, 1), x_trans_unsqueeze)  # kd1 @ k1d = kdd
    return co_matrix

  
def get_embed_sum(eigenvals, eigenvecs, x):
    x_trans = eigenvecs.T @ x  # kd
    x_trans = torch.diag(1 - eigenvals) @ x_trans # kd
    embed_sum = eigenvecs @ x_trans # nk @ kd = nd
    return embed_sum


def get_embed_mean(embed_sum, label):
    class_matrix = F.one_hot(label).float()  # nc
    class_matrix = class_matrix.T  # cn
    embed_sum = class_matrix @ embed_sum  # cd
    mean_weight = (1 / class_matrix.sum(1)).unsqueeze(-1)  # c1
    embed_mean = mean_weight * embed_sum
    embed_mean = F.normalize(input=embed_mean, p=2, dim=1)
    return embed_mean


def get_train_lcc(idx_lcc, idx_train, y_full, num_nodes, num_classes):
    idx_train_lcc = list(set(idx_train).intersection(set(idx_lcc)))
    y_full = y_full.cpu().numpy()
    if len(idx_lcc) == num_nodes:
        idx_map = idx_train
    else:
        y_train = y_full[idx_train]
        y_train_lcc = y_full[idx_train_lcc]

        y_lcc_idx = list((set(range(num_nodes)) - set(idx_train)).intersection(set(idx_lcc)))
        y_lcc_ = y_full[y_lcc_idx]
        counter_train = Counter(y_train)
        counter_train_lcc = Counter(y_train_lcc)
        idx = np.arange(len(y_lcc_))
        for c in range(num_classes):
            num_c = counter_train[c] - counter_train_lcc[c]
            if num_c > 0:
                idx_c = list(idx[y_lcc_ == c])
                idx_c = np.array(y_lcc_idx)[idx_c]
                idx_train_lcc += list(np.random.permutation(idx_c)[:num_c])
        idx_map = [idx_lcc.index(i) for i in idx_train_lcc]
                        
    return idx_train_lcc, idx_map



def sys_data_init(args, ori_data, map):

    edge_i, edge_w = gcn_norm(ori_data.edge_index)
    ori_adj = sparse_mx_to_torch_sparse_tensor(to_scipy_sparse_matrix(edge_i, edge_w, num_nodes=ori_data.num_nodes).tocoo()).cuda()

    map_inverse = torch.linalg.pinv(map)
    map_inverse = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(map_inverse.cpu().numpy()).tocoo()).cuda()

    map = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(map.cpu().numpy())).cuda()
    map_transpose = map.t()

    sys_adj = torch.spmm(map_transpose, ori_adj)
    sys_adj = torch.spmm(sys_adj, map)
    sys_adj = sys_adj.to_dense()

    # if 'hm' in args.dataset:
    #     mask = (sys_adj < 1e-1) & ~torch.eye(sys_adj.size(0), dtype=torch.bool).cuda()
    #     sys_adj[mask] = 0
    row_max, _ = torch.max(sys_adj, dim=1, keepdim=True)
    normalized_matrix = sys_adj / row_max

    sys_x_init = torch.spmm(map_inverse, ori_data.x)

    return normalized_matrix, sys_x_init