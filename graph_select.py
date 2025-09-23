
import os
import torch
from utils2 import Embed_test2
from torch_geometric.data import Data
import numpy as np

def condensed_graph_selection(args, data, global_emb):
    path_list = []
    path = f"./results/{args.dataset}/{args.reduction_rate}/"
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        path_list.append(item_path)

    best_result = 0
    print(f'Find {len(path_list)} condensed graphs')
    num = 1
    for path in path_list:
        print(f'Process {num}/{len(path_list)}')
        adj = torch.load(path+'/sys_adj.pt', weights_only=True).cuda()
        x = torch.load(path+'/sys_x.pt', weights_only=True).cuda()
        y = torch.load(path+'/sys_y.pt', weights_only=True).cuda()
        map = torch.load(path + '/sys_plan.pt', weights_only=True)


        c_edge_index = torch.nonzero(adj, as_tuple=False).t()
        c_edge_weight = adj[c_edge_index[0], c_edge_index[1]]
        c_graph = Data(x=x, edge_index=c_edge_index, edge_weight=c_edge_weight, y=y, num_nodes=x.shape[0], num_calsses=data.num_classes)

        args.runs = 3
        args.epoch = 400
        args.hard = False
        a = []
        for i in range(args.runs):
            aaa= Embed_test2(args, data, c_graph,False)
            a.append(aaa.model_train())
        mean_result = np.mean(a)
        if mean_result > best_result:
            best_result= mean_result
            best_adj = adj.detach().cpu()
            best_x = x.detach().cpu()
            best_y = y.detach().cpu()
            best_map = map
        num += 1
    edge_index = torch.nonzero(best_adj, as_tuple=False).t()
    edge_weight = best_adj[edge_index[0], edge_index[1]]
    graph = Data(x=best_x, edge_index=edge_index, edge_weight=edge_weight, y=best_y, map=best_map, num_nodes=best_x.shape[0], num_calsses=data.num_classes)
    #torch.save(graph, f'./output/pgc_{args.dataset}_{args.reduction_rate}.pt')