import torch.nn as nn
import random
import torch
import torch.nn.functional as F
import ot
from ot.gromov import semirelaxed_fused_gromov_wasserstein
import geomloss
from utils2 import generate_condensed_z_y, results_test
from utils import aug_normalized_adjacency
from model.message_passing_ax import graph_diffusion_dense, graph_diffusion_sparse
import time
class OptimalTransportModel(nn.Module):
    def __init__(self, sys_x, sys_adj):
        super(OptimalTransportModel, self).__init__()

        self.sys_x = nn.Parameter(sys_x.clone().detach().requires_grad_(True))
        self.sys_adj = nn.Parameter(sys_adj.clone().detach().requires_grad_(True))

    def forward(self, K, T):
        self.K = K
        self.T = T
        sys_sym_adj = F.relu((self.sys_adj+self.sys_adj.t())/2)
        #sys_sym_adj = F.relu(self.sys_adj)
        self.sys_norm_adj = aug_normalized_adjacency(sys_sym_adj)
        self.sys_emb, _ = graph_diffusion_dense(self.sys_norm_adj, self.sys_x, K, T)

        return sys_sym_adj, self.sys_x, self.sys_emb

    def compute_sinkhorn_loss(self, ori_x, ori_adj, args):

        h1 = ot.unif(ori_x.shape[0], type_as=ori_x)
        h2 = ot.unif(self.sys_x.shape[0], type_as=self.sys_x)

        if '__' in args.dataset:
            ori_emb, _ = graph_diffusion_dense(ori_adj.to('cpu'), ori_x.to('cpu'), self.K, self.T)
            ori_emb = ori_emb.cuda()
        else:
            ori_emb, _ = graph_diffusion_sparse(ori_adj.to_sparse(), ori_x, self.K, self.T)

        cost_emb = ot.dist(ori_emb, self.sys_emb, metric='euclidean')
        emb_p = ot.emd(h1, h2, cost_emb, numItermax=500000)

        cost_x = ot.dist(ori_x, self.sys_x, metric='euclidean')
        if args.xi == 0:
            graph_p = torch.zeros_like(emb_p).cuda()
        else:
            if 'hm' in args.dataset:
                graph_p = semirelaxed_fused_gromov_wasserstein(cost_x.to('cpu'), ori_adj.to('cpu'), self.sys_norm_adj.to('cpu'), h1.to('cpu'), symmetric=True, alpha=args.rho, log=False, G0=None)
                graph_p = graph_p.cuda()
            else:
                graph_p = semirelaxed_fused_gromov_wasserstein(cost_x, ori_adj, self.sys_norm_adj, h1, symmetric=True, alpha=args.rho, log=False, G0=None)

        if args.xi == 0:
            plan_matching = 0
        else:
            plan_matching = torch.linalg.matrix_norm(graph_p - emb_p, ord='fro')

        sinkhorn_loss = geomloss.SamplesLoss("sinkhorn", p=args.p_class, blur=args.blur ** (1 / 2), scaling=args.scaling, backend=args.backend)
        plan_cost_loss = sinkhorn_loss(ori_emb, self.sys_emb)

        return plan_cost_loss + args.xi * plan_matching, emb_p




class Graph_opt:
    def __init__(self, args, data, x, x2, sys_x_init, sys_adj_init):
        self.args = args
        self.data = data
        self.x = x
        self.x2 = x2
        self.sys_x_init = sys_x_init
        self.sys_adj_init = sys_adj_init

    def model_train(self):
        print("Calculate target embedding \mathbf{\hat{Z}} and transmission plan \mathbf{Plan} and transmission cost \mathbf{Cost}")
        args = self.args
        model = OptimalTransportModel(self.sys_x_init, self.sys_adj_init).cuda()

        optimizer_feat = torch.optim.Adam([model.sys_x], lr=args.lr_feat, weight_decay=args.wd_feat)
        optimizer_adj = torch.optim.Adam([model.sys_adj], lr=args.lr_adj, weight_decay=args.wd_adj)


        model.train()
        loss_old = 10000

        if args.val_on_nc:
            best_result = 0
        else:
            best_result = 10000

        from torch_geometric.utils import to_scipy_sparse_matrix
        from torch_geometric.nn.conv.gcn_conv import gcn_norm
        from deeprobust.graph.utils import sparse_mx_to_torch_sparse_tensor
        if args.dataset in ['yelp', 'reddit', 'flickr']:
            data_train = self.data[0]
            edge_i, edge_w = gcn_norm(data_train.edge_index, data_train.edge_weight, data_train.x.shape[0])
            ori_adj = sparse_mx_to_torch_sparse_tensor(to_scipy_sparse_matrix(edge_i, edge_w, num_nodes=data_train.num_nodes).tocoo()).to_dense().cuda()
        else:
            edge_i, edge_w = gcn_norm(self.data.edge_index, self.data.edge_weight, self.data.x.shape[0])
            ori_adj = sparse_mx_to_torch_sparse_tensor(to_scipy_sparse_matrix(edge_i, edge_w).tocoo()).to_dense().cuda()

        start_time = time.time()
        for epoch in range(args.epoch_init_opt):
            epoch_time = time.time()- start_time


            random_T = random.uniform(2, 6)
            random_K = random.randint(10, 15)

            if 'hm' in args.dataset:
                random_T = random.uniform(0.5, 1)
                random_K = random.randint(1, 2)
            sys_adj, sys_x, sys_emb = model(K=random_K, T=random_T)

            if args.dataset in ['yelp', 'reddit', 'flickr']:
                loss, emb_p = model.compute_sinkhorn_loss(self.data[0].x, ori_adj, args)
            else:
                loss, emb_p = model.compute_sinkhorn_loss(self.data.x, ori_adj, args)

            optimizer_feat.zero_grad()
            optimizer_adj.zero_grad()

            loss.backward()

            if epoch % (args.e1 + args.e2) < args.e1:
                optimizer_adj.step()
            else:
                optimizer_feat.step()


            with torch.no_grad():
                model.eval()

                sys_adj, sys_x, sys_emb = model(K=args.K, T=args.T)

                if args.dataset in ['yelp', 'reddit', 'flickr']:
                    loss, emb_p = model.compute_sinkhorn_loss(self.data[0].x, ori_adj, args)
                else:
                    loss, emb_p = model.compute_sinkhorn_loss(self.data.x, ori_adj, args)


                if epoch % 5 == 0 and epoch >= args.numItermax:
                #if epoch_time >=10:
                    row_sums = torch.sum(emb_p, dim=1, keepdim=True)
                    P_norm = emb_p / row_sums

                    if args.dataset in ['yelp', 'reddit', 'flickr']:
                        mmd_dist = ot.max_sliced_wasserstein_distance(self.x[0], sys_emb)
                        s_emb_init, s_emb_label = generate_condensed_z_y(self.data[0], sys_emb, P_norm)
                    else:
                        mmd_dist = ot.max_sliced_wasserstein_distance(self.x, sys_emb)

                        print(f"Epoch:{epoch}, Loss: {loss}, Embedding_mmd_dist: {mmd_dist}")
                        s_emb_init, s_emb_label = generate_condensed_z_y(self.data, sys_emb, P_norm)
                    current_result, current_std = results_test(self.args, self.data, self.x,self.x2, s_emb_init=s_emb_init, s_emb_label=s_emb_label)

                    if args.val_on_nc:
                        if current_result > best_result:
                            best_result = current_result
                            best_result_val = current_result
                            best_std = current_std
                            best_emb = s_emb_init.detach().cpu()
                            best_label = s_emb_label.detach().cpu()
                            best_plan = emb_p.detach().cpu()
                            best_adj = sys_adj.detach().cpu()
                            best_x = sys_x.detach().cpu()
                    else:
                        if mmd_dist < best_result:
                            best_result = mmd_dist
                            best_result_val = current_result
                            best_std = current_std
                            best_emb = s_emb_init.detach().cpu()
                            best_label = s_emb_label.detach().cpu()
                            best_plan = emb_p.detach().cpu()
                            best_adj = sys_adj.detach().cpu()
                            best_x = sys_x.detach().cpu()
                    start_time = time.time()
        import arrow
        import json
        timestamp = arrow.now().timestamp()
        from pathlib import Path
        folder_path = Path(f"./results/{args.dataset}/{args.reduction_rate}/{timestamp}/")
        folder_path.mkdir(parents=True, exist_ok=True)
        torch.save(best_emb, f"./results/{args.dataset}/{args.reduction_rate}/{timestamp}/sys_emb.pt")
        torch.save(best_label, f"./results/{args.dataset}/{args.reduction_rate}/{timestamp}/sys_y.pt")
        torch.save(best_plan, f"./results/{args.dataset}/{args.reduction_rate}/{timestamp}/sys_plan.pt")
        torch.save(best_adj,f"./results/{args.dataset}/{args.reduction_rate}/{timestamp}/sys_adj.pt")
        torch.save(best_x,f"./results/{args.dataset}/{args.reduction_rate}/{timestamp}/sys_x.pt")

        print(f'Best result: {best_result_val:.2f}±{best_std:.2f}')