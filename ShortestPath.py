import numpy as np
import networkx as nx
from torch_geometric.datasets import TUDataset
import torch
from torch_geometric.utils import degree

def to_networkx(data):
    G = nx.Graph()
    edge_index = data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0, i], edge_index[1, i])
    return G

def get_feature(dataset, node_label):
    Gs = []
    min_sp = float('inf')
    max_sp = 0
    graph_num = len(dataset)
    
    for i in range(graph_num):
        G = nx.floyd_warshall_numpy(to_networkx(dataset[i]))
        G = np.where(np.isinf(G), 0, G)  # Correct replacement of inf values
        G = np.where(np.isnan(G), 0, G)  # Correct replacement of nan values
        max_sp = max(max_sp, np.max(G[G > 0]))
        min_sp = min(min_sp, np.min(G[G > 0]))
        np.fill_diagonal(G, -1)
        Gs.append(G)
    
    feature_len = int(max_sp) + 1
    if node_label == True:
        labels = []
        for i in range(graph_num):
            mat = dataset[i].x
                
            label_i = []
            for j in range(dataset[i].num_nodes):
                label_i.append(torch.nonzero(mat[j]).item())
                
            labels.append(label_i)
    else: # for unlabeled graph
        labels = []
        for i in range(graph_num):
            edge_index = dataset[i].edge_index
            node_degrees = degree(edge_index[0]).tolist()
            for j in range(len(node_degrees)):
                node_degrees[j] = int(node_degrees[j])
           
            labels.append(node_degrees)

    
    unique_labels = np.unique([label for i in range(len(labels)) for label in labels[i]])
    
    label_pairs = [(l1, l2) for l1 in unique_labels for l2 in unique_labels]
    label_pair_to_index = {pair: idx for idx, pair in enumerate(label_pairs)}
    
    feature = np.zeros((graph_num, feature_len, len(label_pairs)))
    
    for i in range(graph_num):
        graph = to_networkx(dataset[i])
        # mat = dataset[i].x  # Assuming the node labels are stored in 'x'
        # label_i = []
        # for j in range(dataset[i].num_nodes):
        #     label_i.append(torch.nonzero(mat[j]).item())
        label_i = labels[i]
            
        for j in range(len(Gs[i])):
            for k in range(len(Gs[i])):
                path_len = int(Gs[i][j][k])
                if path_len > 0:
                    start_label = label_i[j]  # Convert to scalar
                    end_label = label_i[k]    # Convert to scalar
                    label_index = label_pair_to_index[(start_label, end_label)]
                    feature[i][path_len][label_index] += 1
    
    #feature = feature / 2
    
    # Flatten the feature matrix to a 2D matrix (graph_num, feature_len * len(label_pairs))
    feature = feature.reshape(graph_num, -1)
    feature = feature[:, [not np.all(feature[:, i] == 0) for i in range(feature.shape[1])]]
    
    return feature

if __name__ == '__main__':
    dataset = TUDataset(root='./data/MUTAG', name='MUTAG')
    phi = get_feature(dataset, node_label=True)
    print(phi.shape)
    print(phi)
    print(dataset.y)





