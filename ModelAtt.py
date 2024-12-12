import torch
import torch.nn as nn
import torch.nn.functional as F
from Attention import Attention
from GAttention import GAttention
from MLP import MLP

class Model(nn.Module):
    """
    A graph neural network model that utilizes hierarchical assignment layers, attention mechanisms,
    and multi-layer perceptron (MLP) layers for graph classification.

    Args:
        graph_num (int): The number of graphs.
        feature_hid (int): Hidden dimensions of the attention layers.
        feature_dim1 (int): Hidden dimensions of the first hierarchical assignment layer.
        layers (int): The number of hierarchical assignment layers.
        num_mlp_layers (int): The number of layers in the MLP used for classification.
        hidden_dim1 (int): The hidden dimension in the first MLP layer.
        hidden_dim2 (int): The hidden dimension in the second MLP layer.
        n_labels (int): The number of graph labels.
    """
    def __init__(self, graph_num, feature_hid, feature_dim1, layers, num_mlp_layers, hidden_dim1, hidden_dim2, n_labels):
        super(Model, self).__init__()

        self.alpha = 0.2
        self.layers = layers

        feature_dims = [feature_dim1]
        for layer in range(layers - 1):
            feature_dim1 = int(feature_dim1 / 2)
            feature_dims.append(feature_dim1)

        # Define assignment layers
        self.Assign_Linear = nn.ModuleList()
        for layer in range(layers - 1):
            self.Assign_Linear.append(nn.Linear(graph_num, feature_dims[layer + 1]))

        # Define attention layers
        self.Attention_list = nn.ModuleList()
        for layer in range(layers):
            self.Attention_list.append(Attention(feature_dims[layer], feature_hid, feature_dims[layer]))

        # Define global attention layers
        self.global_attention = nn.ModuleList()
        for layer in range(layers - 1):
            self.global_attention.append(GAttention(layer + 2))

        # Define classification layers
        self.classify = nn.ModuleList()
        input_dim = graph_num
        for layer in range(layers):
            self.classify.append(MLP(num_mlp_layers, input_dim, hidden_dim1, hidden_dim2, n_labels))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): The input graph features of shape (batch_size, graph_num).

        Returns:
            torch.Tensor: The classification scores for each graph.
            list: The list of assignment matrices (S) for each layer.
        """
        features = [x]
        S_list = []

        # Apply hierarchical assignment layers
        for layer in range(self.layers - 1):
            S = self.Assign_Linear[layer](x.T)
            S = self.softmax(S)
            S_list.append(S)
            x = torch.matmul(x, S)
            features.append(x)

        # Calculate kernel matrices using attention and global attention
        K_list = []
        Kernel_list = []
        for layer in range(self.layers):
            features[layer] = self.Attention_list[layer](features[layer])
            K = torch.matmul(features[layer], features[layer].T)
            Kernel_list.append(K)
            if layer >= 1:
                K = torch.stack(Kernel_list, dim=0)
                K = self.global_attention[layer - 1](K)
            K_list.append(K)

        # Sum the scores from the classification layers
        score = 0
        for layer in range(self.layers):
            score += self.classify[layer](K_list[layer])

        return score, S_list


if __name__ == '__main__':
    input = torch.randn(2, 20).float()
    net = Model(graph_num=2, feature_hid=4, feature_dim1=20, layers=4, num_mlp_layers=4, hidden_dim1=4, hidden_dim2=4, n_labels=3)
    print(input)
    output = net(input)
    print(output)
    print(output[0].shape)
