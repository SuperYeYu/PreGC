
from copy import deepcopy
from sklearn.metrics import accuracy_score, r2_score
from torch_kmeans import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score,silhouette_score,calinski_harabasz_score,davies_bouldin_score

def evaluate_clustering(embeddings, true_labels, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    ss = silhouette_score(embeddings, cluster_labels)
    chi = calinski_harabasz_score(embeddings, cluster_labels)
    dbi = davies_bouldin_score(embeddings, cluster_labels)
    return nmi, ari, ss, chi, dbi

class MLP(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim, dropout):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList([nn.Linear(num_features, hidden_dim), nn.Linear(hidden_dim, num_classes)])
        self.reset_parameter()

    def reset_parameter(self):
        for lin in self.layers:
            nn.init.xavier_uniform_(lin.weight.data)
            if lin.bias is not None:
                lin.bias.data.zero_()

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)


        for ix, layer in enumerate(self.layers):
            x = layer(x)
            if ix != len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
                embedding = x
        return x, embedding


class Embed_test:
    def __init__(self, args, data, x, x2, original, s_emb_init, s_emb_label):
        self.args = args
        self.data = data
        self.x = x
        self.x2 = x2

        self.original = original
        self.s_emb_init = s_emb_init
        self.s_emb_label = s_emb_label

    def model_train(self):
        data =self.data
        args = self.args

        if args.dataset in ['yelp','reddit', 'flickr']:
            num_classes = data[0].num_classes
            self.d = self.x[0].shape[1]
            self.num_classes = data[0].num_classes
        else:
            num_classes = data.num_classes
            self.d = self.x.shape[1]
            self.num_classes = data.num_classes
        model2 = MLP(self.d, num_classes=num_classes, hidden_dim=args.hidden_dim, dropout=args.dropout).cuda()
        optimizer = torch.optim.Adam(model2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_result_val = 0
        epochs = 400
        lr = args.lr

        if args.dataset in ['yelp', 'reddit', 'flickr']:
            if self.original:
                x_train = self.x[0]
                y_train = data[0].y
            else:
                x_train = self.s_emb_init
                y_train = self.s_emb_label
            if args.test_sgc:
                x_val = self.x2[1]
                y_val = data[1].y
                x_test = self.x2[2]
                y_test = data[2].y
            else:
                x_val = self.x[1]
                y_val = data[1].y
                x_test = self.x[2]
                y_test = data[2].y

        else:
            if self.original:
                x_train = self.x[data.train_mask]
                if data.num_classes == 1:
                    y_train = data.y_regre_std[data.train_mask]
                else:
                    y_train = data.y[data.train_mask]
            else:
                x_train = self.s_emb_init
                y_train = self.s_emb_label
            if args.test_sgc:
                x_val = self.x2[data.val_mask]
                y_val = data.y[data.val_mask]
                x_test = self.x2[data.test_mask]
                y_test = data.y[data.test_mask]
            else:
                x_val = self.x[data.val_mask]
                y_val = data.y[data.val_mask]
                x_test = self.x[data.test_mask]
                y_test = data.y[data.test_mask]

        for i in range(epochs):
            if i == epochs // 2 and i > 0:
                lr = lr * 0.1
                optimizer = torch.optim.Adam(model2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            model2.train()
            optimizer.zero_grad()

            output,_ = model2.forward(x_train)

            if num_classes != 1:  #NC
                if self.original:
                    loss_train = F.nll_loss(F.log_softmax(output, dim=1), y_train)  # Hard label
                else:
                    loss_train = torch.mean(torch.sum(y_train * -F.log_softmax(output, dim=1), dim=1))  # Soft label
            else:  #LP
                loss_train = F.mse_loss(output.squeeze(1), y_train)


            loss_train.backward()
            optimizer.step()

            with torch.no_grad():
                model2.eval()
                output,_ = model2.forward(x_val)

                if num_classes != 1:  # NC
                    pred = output.max(1)[1]
                    pred = pred.cpu().numpy()
                    result_val = accuracy_score(y_val.cpu().numpy(), pred)
                else:
                    pred = output.squeeze().cpu().numpy() * data.y_std.cpu().numpy() + data.y_mean.cpu().numpy()
                    result_val = r2_score(y_val.cpu().numpy(), pred)

                if result_val > best_result_val:
                    best_result_val = result_val
                    best_epoch = i
                    weights = deepcopy(model2.state_dict())

        model2.load_state_dict(weights)
        model2.eval()
        output,embed = model2.forward(x_test)

        if num_classes != 1:  # NC
            pred = output.max(1)[1].cpu().numpy()
            result_test = accuracy_score(y_test.cpu().numpy(), pred)
            if args.dataset in ['yelp', 'reddit', 'flickr']:
                nmi, ari, ss, chi, dbi = evaluate_clustering(embed.detach().cpu().numpy(), y_test.cpu().numpy(),num_classes)
            else:
                nmi, ari,ss, chi, dbi = evaluate_clustering(embed.detach().cpu().numpy(), y_test.cpu().numpy(), data.num_classes)
        else:
            pred = output.squeeze().detach().cpu().numpy() * data.y_std.cpu().numpy() + data.y_mean.cpu().numpy()
            result_test = r2_score(y_test.cpu().numpy(), pred)
        return best_result_val, result_test
