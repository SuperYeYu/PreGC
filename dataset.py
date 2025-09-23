
import torch
from torch_geometric.datasets import Planetoid
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import to_undirected


def get_dataset(name, normalize_features=True, transform=None):
    path = f"./data/{name}"

    if name in ["cora", "citeseer", "pubmed"]:
        dataset = Planetoid(path, name)
    else:
        raise NotImplementedError
    data = dataset[0]
    data.num_classes = dataset.num_classes
    data.num_nodes = data.x.shape[0]
    return data


def get_dataset2(name):
    path = f"./data/{name}/{name}/{name}.pt"
    dataset = torch.load(path)
    if name in ["arxiv_topic", 'arxiv_year',"arxiv_topic_s", 'arxiv_year_s',"arxiv_s",'arxiv_topic_0.15','arxiv_topic_0.3','arxiv_topic_0.45','arxiv_topic_0.6','arxiv_topic_0.75']:
        feat = dataset.x
        scaler = StandardScaler()
        scaler.fit(feat)
        dataset.x = torch.tensor(scaler.transform(feat), dtype=torch.float32)
        edge_index_with_self_loops, _ = add_self_loops(dataset.edge_index)
        undirected_edge_index = to_undirected(edge_index_with_self_loops)
        dataset.edge_index = undirected_edge_index


    elif name in ["hm_class", 'hm_regre',"hm_class_s", 'hm_regre_s']:

        if name in ["hm_regre", "hm_regre_s"]:
            dataset.y_mean = torch.mean(dataset.y)
            dataset.y_std = torch.std(dataset.y)
            dataset.y_regre_std = (dataset.y - dataset.y_mean) / dataset.y_std

    return dataset

def get_dataset3(name):
    path1 = f"./data/{name}/{name}/{name}_sub.pt"
    path2 = f"./data/{name}/{name}/{name}_all.pt"
    d1,d2,d3 = torch.load(path1)
    d4= torch.load(path2)
    return [d1,d2,d3,d4]
