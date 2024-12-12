import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import copy
from collections import defaultdict
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import numpy as np


def get_WL_feature(dataset, h, node_label =True):
    graph_num = len(dataset)
    #label_num = dataset.num_node_features

    label = []

    n_nodes = 0
    max_num_nodes = 0
    for i in range(graph_num):
        n_nodes = n_nodes + dataset[i].num_nodes
        if dataset[i].num_nodes > max_num_nodes:
            max_num_nodes = dataset[i].num_nodes
    print("the maximum number of nodes: %d"%max_num_nodes)

    '''
    Initialize the labels 
    '''

    if node_label == True:
        for i in range(graph_num):
            mat = dataset[i].x
            
            label_i = []
            for j in range(dataset[i].num_nodes):
                 label_i.append(torch.nonzero(mat[j]).item())
            
            label.append(label_i)

    else: # for unlabeled graph
        for i in range(graph_num):
            edge_index = dataset[i].edge_index
            #print(edge_index)
            degree = torch.zeros(dataset[i].num_nodes, dtype=torch.long)
            row, col = edge_index
            for node in row:
                degree[node] += 1

            label.append(degree)

    '''
    kernel 0-th iteration
    '''
    phi = np.zeros((graph_num, n_nodes))
    for i in range(graph_num):
        aux = np.bincount(label[i])
        phi[i, label[i]] = aux[label[i]]

    '''
    remove zero columns/features
    '''
    non_zero_columns = np.any(phi, axis=0)
    feature = []
    phi = phi[:, non_zero_columns]
    #print(phi)
    feature.append(phi)

    new_labels = copy.deepcopy(label)


    '''
    adjacency list for TUDataset
    '''
    lists = []
    for i in range(graph_num):
        adj_list = defaultdict(list)
        edge_index = dataset[i].edge_index
        adj = edge_index.t().tolist()
        for src, dst in edge_index.t().tolist():
            if dst not in adj_list[src]:
                adj_list[src].append(dst)
            if src not in adj_list[dst]:
                adj_list[dst].append(src)
        lists.append(adj_list)

    it = 0
    while it < h:
        label_lookup = {}
        label_counter = 0
        phi_tmp = np.zeros((graph_num, n_nodes))
        for i in range(graph_num):
            #print(dataset[i].num_nodes)
            for j in range(dataset[i].num_nodes):
                neighbor_label = [label[i][j] for j in lists[i][j]]
                if neighbor_label:
                    long_label = np.concatenate((np.atleast_1d(label[i][j]), np.sort(neighbor_label)))
                else:
                    #print(label[i][j])
                    long_label = label[i][j]
                    
                long_label_string = str(long_label)
                if long_label_string not in label_lookup:
                    label_lookup[long_label_string] = label_counter
                    if label_counter in [4, 5, 6, 15]:
                        print(long_label_string)
                    new_labels[i][j] = label_counter
                    label_counter += 1
                else:
                    new_labels[i][j] = label_lookup[long_label_string]

            aux = np.bincount(new_labels[i])
            #print(aux)
            phi_tmp[i, new_labels[i]] = aux[new_labels[i]]
        label = copy.deepcopy(new_labels)
        #print(label)
        it = it + 1
        
        column_sums = np.sum(phi_tmp, axis=0)

        top_k_indices = np.argsort(column_sums)[-4:]
        print(top_k_indices)


        non_zero_columns = np.any(phi_tmp, axis=0)

        phi_t = phi_tmp[:, non_zero_columns]
        #print(phi_t.shape)

        feature.append(phi_t)

        phi = np.concatenate((phi, phi_t), axis=1)
    print(phi)

    return feature, phi

if __name__ == '__main__':
    dataset = TUDataset(root='./data/PROTEINS', name='PROTEINS')
    # for i in range(10):
    get_WL_feature(dataset, h=1, node_label=True)
    print(dataset.y)



















