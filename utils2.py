import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score, r2_score, f1_score
from torch_kmeans import KMeans
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SGConv
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score,silhouette_score,calinski_harabasz_score,davies_bouldin_score, mean_absolute_percentage_error, explained_variance_score

from emb_test import Embed_test
class SGC(nn.Module):
    def __init__(self, args, input_dim, output_dim, hid_dim):
        super(SGC, self).__init__()
        self.num_layers = 1
        self.dropout = 0.5

        self.activation = 'relu'
        if self.activation == 'relu':
            self.activation_fn = F.relu
        elif self.activation == 'leaky_relu':
            self.activation_fn = F.leaky_relu
        elif self.activation == 'tanh':
            self.activation_fn = F.tanh
        elif self.activation == 'sigmoid':
            self.activation_fn = F.sigmoid
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        self.layers = SGConv(input_dim, hid_dim, K=self.num_layers)
        self.output = nn.Linear(hid_dim, output_dim)

        self.reset_parameter()

    def reset_parameter(self):

        nn.init.xavier_uniform_(self.layers.lin.weight.data)
        if self.layers.lin.bias is not None:
            self.layers.lin.bias.data.zero_()

        nn.init.xavier_uniform_(self.output.weight.data)
        if self.output.bias is not None:
            self.output.bias.data.zero_()


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if data.edge_weight is not None:
            edge_weight = data.edge_weight
        else:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

        x = self.layers(x, edge_index, edge_weight)
        x = self.activation_fn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        logits = self.output(x)

        return logits, x

def evaluate_clustering(embeddings, true_labels, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    ss = silhouette_score(embeddings, cluster_labels)
    chi = calinski_harabasz_score(embeddings, cluster_labels)
    dbi = davies_bouldin_score(embeddings, cluster_labels)
    return nmi, ari, ss, chi, dbi


def generate_condensed_z_y(data, B, P2):
    s_emb_init = B.detach()
    P = P2.detach()

    P_one_hot = torch.zeros_like(P)
    P_one_hot[torch.arange(P.shape[0]), P.argmax(dim=1)] = 1.0


    train_labels = data.y[data.train_mask]
    one_hot_train_labels = F.one_hot(train_labels, num_classes=data.num_classes).float().cuda()

    one_hot_labels = torch.zeros(P.shape[0], data.num_classes).cuda()

    one_hot_labels[data.train_mask] = one_hot_train_labels


    s_emb_label = torch.mm(P_one_hot.t(), one_hot_labels)

    s_emb_label = F.normalize(s_emb_label.clamp(min=0), p=1, dim=1)


    return s_emb_init, s_emb_label


def results_test(args,data,g_emb, test_emb,s_emb_init=None,s_emb_label=None):
    results_v = []
    for ep in range(args.runs):
        agent = Embed_test(args=args, data=data, x=g_emb, x2=test_emb, original=args.original, s_emb_init=s_emb_init, s_emb_label=s_emb_label)
        results_val, results_test = agent.model_train()
        results_v.append(results_val)
    mean_result = np.mean(results_v)*100
    std_result = np.std(results_v)*100
    return mean_result, std_result




class Embed_test2:

    def __init__(self, args, ori_data, con_data, original):
        self.args = args
        self.ori_data = ori_data
        self.con_data = con_data
        self.dim = ori_data.x.shape[1]
        self.num_classes = ori_data.num_classes
        self.original = original

    def model_train(self):
        ori_data =self.ori_data
        args = self.args


        model = SGC(args, self.dim, self.num_classes, self.args.hidden_dim).cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_result_val = 0
        epochs = 600
        lr = args.lr
        if self.original:
            input_graph = ori_data
            if ori_data.num_classes == 1:
                self.y_train = ori_data.y_regre_std[ori_data.train_mask]
            else:
                self.y_train = ori_data.y[ori_data.train_mask]
        else:
            input_graph = self.con_data
            self.y_train = self.con_data.y


        y_val = ori_data.y[ori_data.val_mask]
        y_test = ori_data.y[ori_data.test_mask]


        for i in range(epochs):
            if i == epochs // 2 and i > 0:
                lr = lr * 0.1
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            model.train()

            optimizer.zero_grad()

            output, _ = model.forward(input_graph)

            if ori_data.num_classes != 1:  # NC
                if self.original:
                    loss_train = F.nll_loss(F.log_softmax(output[ori_data.train_mask], dim=1), self.y_train)
                else:
                    if args.hard:
                        loss_train = F.nll_loss(F.log_softmax(output, dim=1), self.y_train)
                    else:
                        loss_train = torch.mean(torch.sum(self.y_train * -F.log_softmax(output, dim=1), dim=1))
            else:  # NR
                if self.original:
                    loss_train = F.mse_loss(output[ori_data.train_mask].squeeze(1), self.y_train)
                else:
                    loss_train = F.mse_loss(output.squeeze(1), self.y_train)

            loss_train.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                output, ori_emb = model.forward(ori_data)

                if ori_data.num_classes != 1:  # NC
                    pred = output.max(1)[1]
                    pred = pred[ori_data.val_mask].cpu().numpy()
                    result_val = accuracy_score(y_val.cpu().numpy(), pred)
                else:  # NR
                    pred = output[ori_data.val_mask].squeeze().cpu().numpy() * ori_data.y_std.cpu().numpy() + ori_data.y_mean.cpu().numpy()
                    result_val = r2_score(y_val.cpu().numpy(), pred)

                if result_val > best_result_val:
                    best_result_val = result_val
                    best_epoch = i
                    weights = deepcopy(model.state_dict())

        model.load_state_dict(weights)
        model.eval()
        output, embed = model.forward(ori_data)

        all_results = {}

        if ori_data.num_classes != 1:
            pred = output[ori_data.test_mask].max(1)[1].cpu().numpy()
            result_test = accuracy_score(y_test.cpu().numpy(), pred)
            macro_f1 = f1_score(y_test.cpu().numpy(), pred, average='macro')
            nmi, ari, ss, chi, dbi = evaluate_clustering(embed[ori_data.test_mask].detach().cpu().numpy(), y_test.cpu().numpy(), ori_data.num_classes)
            all_results['nmi'] = nmi
            all_results['ari'] = ari
            all_results['acc'] = result_test
            all_results['mac_f1'] = macro_f1
        else:
            pred = output[ori_data.test_mask].squeeze().detach().cpu().numpy() * ori_data.y_std.cpu().numpy() + ori_data.y_mean.cpu().numpy()
            y_test = y_test.cpu().numpy()
            result_test = r2_score(y_test, pred)
            mape = mean_absolute_percentage_error(y_test, pred)
            all_results['mape'] = mape
            explained_variance = explained_variance_score(y_test, pred)
            all_results['explained_variance'] = explained_variance
            all_results['r2'] = result_test

        return all_results['acc']