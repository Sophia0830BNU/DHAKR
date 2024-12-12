
import numpy as np
import os

from math import sqrt

from WeiL_multi import get_WL_feature
#from test import get_WL_feature
#from test import read_to_list
from torch_geometric.datasets import TUDataset

#from ShortestPath import get_feature
from ShortestPath import get_feature
#from SPT import get_feature
#from Graphlet import get_feature_GCGK
import torch
from ModelAtt import Model
#from Model import  Model
from train import train

from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import random

import argparse





'''
hyperparameters: lr, batchsize, weight_decay
'''


parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: MUTAG)')
parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
parser.add_argument('--epochs', type=int, default=350,
                        help='number of epochs to train (default: 350)')
parser.add_argument('--lr', type=float, default=0.004,
                        help='learning rate (default: 0.01)')

parser.add_argument('--feature_hid', type=int, default=50,
                        help='number of hidden units (default: 50)')

parser.add_argument('--layers', type=int, default=1,
                        help='number of cluster layers')
parser.add_argument('--num_mlp_layers', type=int, default=1,
                        help='number of mlp layers')
parser.add_argument('--method', type=str, default="WL",
                        help='The kernel based method')
parser.add_argument('--iteration_num', type=int, default=1,
                        help='The WL iteration_num 1-10')
parser.add_argument('--loss_alpha', type=float, default=0.001,
                        help='loss coefficient (default: 0.01)')
parser.add_argument('--seed', type=int, default=0,
                        help='random seed for data splits')



args = parser.parse_args()




weight_decay = 5e-8
hidden_num1 = 50
hidden_num2 = 300


device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")



def pre_process(labels, dim):
    y = labels
    for i in range(len(y)):
        if y[i] == -1:
            y[i] = 0
    if dim == max(labels):
        for i in range(len(y)):
            y[i] = y[i]-1

    return y

def set_random_seed(seed, deterministic=False):
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seeds = [0, 2, 16, 42, 123, 456, 789, 101112, 222324, 252627]





dataset = TUDataset(root='./data/'+str(args.dataset), name=str(args.dataset))
if args.method == "WL":
    #feature_list = [torch.tensor(feature).float() for feature in get_WL_feature(dataset, h=args.iteration_num, node_label = True)[0]]
    feature_vector = torch.tensor(get_WL_feature(dataset, h= args.iteration_num , node_label = True)[1]).float().to(device)
    feature_dim = feature_vector.shape[1]
elif args.method == "SP":
    #feature_list = [torch.tensor(get_feature(dataset)).float()]
    feature_vector = torch.tensor(get_feature(dataset, node_label = True)).float().to(device)
    feature_dim = feature_vector.shape[1]

y = dataset.y
# dataset_name = 'Shock' 

# folder_path = './data'
# filepath = os.path.join(folder_path, dataset_name)

# y_path = os.path.join(filepath, 'graph_label.txt')   
# y = read_to_list(y_path, l=False)

# feature_vector = get_WL_feature(dataset_name, h=1)
# feature_dim = feature_vector.shape[1]


# dataset_name = 'Shock' 

# folder_path = './data'
# filepath = os.path.join(folder_path, dataset_name)

# y_path = os.path.join(filepath, 'graph_label.txt')   
# y = read_to_list(y_path, l=False)
# if args.method == "SP":
#     feature_vector = torch.tensor(get_feature(dataset_name)).float().to(device)
# else:
#     feature_vector = torch.tensor(get_WL_feature(dataset_name, h= args.iteration_num , node_label = True)[1]).float().to(device)
# feature_dim = feature_vector.shape[1]

# #feature_vector = feature_prepro(feature_vector)

#feature_vector = torch.cat((feature_vector_WL, feature_vector_SP), dim=1)

# #feature_vector = normalize(feature_vector)

label_dim = len(set(y))
y = pre_process(y, label_dim)

#feature_dims = []


graph_num = feature_vector.shape[0]

# for i in range(len(feature_list)):
#     feature_dims.append(feature_list[i].shape[1])
#     feature_list[i] = feature_list[i].float().to(device)



set_random_seed(0, deterministic=True)


test_acc_t = 0 
loss_alpha = args.loss_alpha


Acc_ten_times = []

n_splits = 10
#kfold = KFold(n_splits=n_splits, shuffle=True, random_state= i)
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
splits = kfold.split(np.zeros(len(y)), y)

    
ave_train_fold = []
ave_test_fold = []
k = 0
for train_index, test_index in splits:
    k  = k+1
    print('currnt fold: %d'% (k))
                    
    net = Model(graph_num=graph_num, feature_hid=args.feature_hid,  feature_dim1 = feature_dim, layers = args.layers, num_mlp_layers = args.num_mlp_layers, hidden_dim1=hidden_num1, hidden_dim2 = hidden_num2, n_labels= label_dim)

    train_acc_all, test_acc_all= train(net, feature_vector, y, args.epochs, train_index, test_index, device,  args.lr, weight_decay, alpha = loss_alpha)
    ave_train_fold.append(train_acc_all)
    ave_test_fold.append(test_acc_all)

    
ave_train = np.mean(ave_train_fold, axis=0)
ave_test = np.mean(ave_test_fold, axis=0)

max_mean_acc = np.max(ave_test)
max_index = np.argmax(ave_test)
    
print('current time ten-fold best mean acc: %f' % max_mean_acc)









