from time import perf_counter
from utils import sparse_eye
import torch


def graph_diffusion_dense(adj, features, K, T):
    if K == 0 or T == 0.:
        return features, 0.
    delta = T / K
    eye = torch.eye(adj.shape[0]).to(adj.device)
    op = (1 - delta) * eye + delta * adj
    t = perf_counter()
    for i in range(K):
        features = torch.spmm(op, features)
    precompute_time = perf_counter() - t
    return features, precompute_time


def graph_diffusion_sparse(adj, features, K, T):
    if K == 0 or T == 0.:
        return features, 0.
    delta = T / K
    eye = sparse_eye(adj.shape[0]).to(adj.device)
    op = (1 - delta) * eye + delta * adj
    t = perf_counter()
    for i in range(K):
        features = torch.spmm(op, features)
    precompute_time = perf_counter() - t
    return features, precompute_time