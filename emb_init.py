
import json
import argparse
import torch
from time import perf_counter
from utils import sparse_eye, seed_everything, sys_data_init
from model.message_passing_ax import graph_diffusion_sparse, graph_diffusion_dense
from dataset import get_dataset, get_dataset2, get_dataset3
from torch_geometric.utils import to_scipy_sparse_matrix, remove_self_loops
from deeprobust.graph.utils import sparse_mx_to_torch_sparse_tensor
from graph_opt import Graph_opt
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from utils2 import generate_condensed_z_y, results_test
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import OneHotEncoder
from graph_select import condensed_graph_selection




def kmeans_cluster(num_sys, z, type = False):
    print("Initialize sys node embedding...based on K-means+++")
    if type:
        kmeans = MiniBatchKMeans(n_clusters=num_sys, max_iter=1000, batch_size=50000, max_no_improvement=100, init_size=30000, n_init=20, reassignment_ratio=0.01)
    else:
        kmeans = MiniBatchKMeans(n_clusters=num_sys, max_iter=1000, batch_size=50000)
    t = perf_counter()
    kmeans.fit(z)
    precompute_time = perf_counter() - t
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    encoder = OneHotEncoder(sparse_output=False)
    labels_matrix = encoder.fit_transform(labels.reshape(-1, 1))
    labels_matrix = torch.from_numpy(labels_matrix).to(torch.float).cuda()
    centers = torch.from_numpy(centers).cuda()
    print("Inintialize Times: {:.2f}s".format(precompute_time))
    return centers, labels_matrix


parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
parser.add_argument("--seed", type=int, default=2025)
parser.add_argument("--config", type=str, default='./config/config_init.json')
parser.add_argument("--runs", type=int, default=5)
parser.add_argument("--rounds", type=int, default=50)
parser.add_argument("--dataset", type=str, default="pubmed")
parser.add_argument("--reduction_rate", type=float, default=0.25)
parser.add_argument("--hidden_dim", type=int, default=64)
parser.add_argument("--original", type=bool, default=False)
parser.add_argument("--test_sgc", type=bool, default=True)
#If set to True, it operates in a task-oriented manner; otherwise, it does not.
parser.add_argument("--val_on_nc", type=bool, default=True)

args = parser.parse_args([])

with open(args.config, "r") as config_file:
    config = json.load(config_file)
if args.dataset+'_'+str(args.reduction_rate) in config:
    config = config[args.dataset+'_'+str(args.reduction_rate)]
for key, value in config.items():
    setattr(args, key, value)

torch.cuda.set_device(args.gpu_id)
seed_everything(args.seed)


if args.dataset in ['arxiv_topic_s', 'arxiv_year_s', 'hm_class_s', 'hm_regre_s']:
    data = get_dataset2(args.dataset)
elif args.dataset in ['yelp', 'reddit', 'flickr']:
    data = get_dataset3(args.dataset)
elif args.dataset in ['cora', 'pubmed', 'citeseer']:
    data = get_dataset(args.dataset)


if args.dataset in ['yelp', 'reddit', 'flickr']:
    data_train, data_val, data_test, data_all = data
    edge_i, edge_w = gcn_norm(data_train.edge_index, data_train.edge_weight, data_train.x.shape[0])
    adj_train = sparse_mx_to_torch_sparse_tensor(to_scipy_sparse_matrix(edge_i, edge_w).tocoo())

    edge_i, edge_w = gcn_norm(data_val.edge_index, data_val.edge_weight, data_val.x.shape[0])
    adj_val = sparse_mx_to_torch_sparse_tensor(to_scipy_sparse_matrix(edge_i, edge_w).tocoo())

    edge_i, edge_w = gcn_norm(data_test.edge_index, data_test.edge_weight, data_test.x.shape[0])
    adj_test = sparse_mx_to_torch_sparse_tensor(to_scipy_sparse_matrix(edge_i, edge_w).tocoo())

    edge_i, edge_w = gcn_norm(data_all.edge_index, data_all.edge_weight, data_all.x.shape[0])
    adj_all = sparse_mx_to_torch_sparse_tensor(to_scipy_sparse_matrix(edge_i, edge_w).tocoo())

    adj=[adj_train,adj_val,adj_test, adj_all]

else:
    edge_i, edge_w = gcn_norm(data.edge_index, data.edge_weight, data.x.shape[0])
    adj = sparse_mx_to_torch_sparse_tensor(to_scipy_sparse_matrix(edge_i, edge_w).tocoo())




if 'hm' in args.dataset:
    print('Graph Diffusion on Original Graph...')
    global_emb, precompute_time = graph_diffusion_sparse(adj, data.x, 1, 0.6)
    print('Graph Diffusion Times: {:.2f}s'.format(precompute_time))
    data = data.cuda()
    global_emb = global_emb.cuda()
else:
    if args.dataset in ['yelp', 'reddit', 'flickr']:
        global_emb = []
        for i in range(len(adj)):
            dataset = data[i].cuda()
            g_emb, precompute_time = graph_diffusion_sparse(adj[i].cuda(), dataset.x, args.K, args.T)
            global_emb.append(g_emb)
    else:
        data = data.cuda()
        print('Graph Diffusion on Original Graph...')
        global_emb, precompute_time = graph_diffusion_sparse(adj.cuda(), data.x, args.K, args.T)
        print('Graph Diffusion Times: {:.2f}s'.format(precompute_time))

def init_sys_z(args, data, global_emb):

    if args.dataset in ['yelp', 'reddit', 'flickr']:
        data_train = data[0]
        num_sys = round(data_train.x.shape[0] * args.reduction_rate * 0.1)+1
        sys_emb_init, mapping_init = kmeans_cluster(num_sys, global_emb[3].cpu().numpy(), type=False)

        sys_adj_init, sys_x_init = sys_data_init(args, data[3], mapping_init)  ####

        if args.dataset == 'flickr':
            sys_adj_init = torch.eye(len(sys_x_init))
            nonzero_indices = torch.nonzero(sys_adj_init).t()
            nonzero_values = sys_adj_init[nonzero_indices[0], nonzero_indices[1]]
            edge_i, edge_w = gcn_norm(nonzero_indices, nonzero_values, sys_x_init.shape[0])
            sys_adj = sparse_mx_to_torch_sparse_tensor(to_scipy_sparse_matrix(edge_i, edge_w).tocoo()).cuda()
            init_emb, _ = graph_diffusion_sparse(sys_adj, sys_x_init, args.K, args.T)
        else:

            nonzero_indices = torch.nonzero(sys_adj_init).t()
            nonzero_values = sys_adj_init[nonzero_indices[0], nonzero_indices[1]]

            nonzero_indices, nonzero_values = remove_self_loops(nonzero_indices, nonzero_values)
            edge_i, edge_w = gcn_norm(nonzero_indices, nonzero_values, sys_x_init.shape[0])

            sys_adj = sparse_mx_to_torch_sparse_tensor(to_scipy_sparse_matrix(edge_i, edge_w).tocoo()).cuda()
            init_emb, _ = graph_diffusion_sparse(sys_adj, sys_x_init, args.K, args.T)

        return sys_adj.to_dense(), sys_x_init, mapping_init

    else:
        if 'arxiv' in args.dataset and "s" in args.dataset:
            num_sys = round(data.num_nodes * 0.05 * args.reduction_rate)
        elif 'hm' in args.dataset:
            num_sys = round(data.num_nodes * 0.1 * args.reduction_rate)
        else:
            num_sys = round(torch.sum(data.train_mask).item() * args.reduction_rate)
        print(f'Number of condensed nodes: {num_sys}. Condensed ratio: {str(round(num_sys * 100 / data.num_nodes, 2))}%')

        # init sys graph
        if args.dataset == 'pubmed':
            sys_emb_init, mapping_init = kmeans_cluster(num_sys, global_emb.cpu().numpy(), type = True)
        else:
            sys_emb_init, mapping_init = kmeans_cluster(num_sys, global_emb.cpu().numpy(),type = False)
        sys_adj_init, sys_x_init = sys_data_init(args, data, mapping_init)

        nonzero_indices = torch.nonzero(sys_adj_init).t()
        nonzero_values = sys_adj_init[nonzero_indices[0], nonzero_indices[1]]

        nonzero_indices, nonzero_values = remove_self_loops(nonzero_indices, nonzero_values)
        edge_i, edge_w = gcn_norm(nonzero_indices, nonzero_values, sys_x_init.shape[0])

        sys_adj = sparse_mx_to_torch_sparse_tensor(to_scipy_sparse_matrix(edge_i, edge_w).tocoo()).cuda()
        init_emb, _ = graph_diffusion_sparse(sys_adj, sys_x_init, args.K, args.T)

        return sys_adj.to_dense(), sys_x_init, mapping_init

sys_adj, sys_x, mapping = init_sys_z(args, data, global_emb)

for i in range(args.rounds):
    print('\n Round: ', i)
    agent = Graph_opt(args=args, data=data, x=global_emb, x2=global_emb, sys_x_init=sys_x, sys_adj_init=sys_adj)
    agent.model_train()


print('Selecting Condensed Graph...')
condensed_graph_selection(args, data, global_emb)


