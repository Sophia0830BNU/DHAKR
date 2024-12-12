import numpy as np
import torch
from torch import nn
from torch.nn import init
import math


# class GAttention(nn.Module):

#     def __init__(self, input_dim, hidden_dim=16, output_dim=1):

#         super(GAttention, self).__init__()

#         self.project = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_dim, bias=False)
    #     )
    #     self.init_weights()

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             init.kaiming_normal_(m.weight, mode='fan_out')
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             init.constant_(m.weight, 1)
    #             init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             init.normal_(m.weight, std=0.1)
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)

#     def forward(self, z):
#         w = self.project(z)
#         beta = torch.softmax(w, dim=1)

#         return (beta * z).sum(1)

# 暂时稍好的结果

class GAttention(nn.Module):

    def __init__(self, input_dim, hidden_dim=100):

        super(GAttention, self).__init__()
        self.alpha = 0.2 
        #self.Linear = nn.Linear(kernel_dim, kernel_dim)
        self.Linear0 = nn.Linear(input_dim, hidden_dim)
        self.Linear1 = nn.Linear(hidden_dim, input_dim)

        self.leakrelu = nn.LeakyReLU(self.alpha)
        self.softmax = nn.Softmax(dim=0)

    #     self.init_weights()

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             init.kaiming_normal_(m.weight, mode='fan_out')
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             init.constant_(m.weight, 1)
    #             init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             init.normal_(m.weight, std=0.1)
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)

        
    def forward(self, z):

        w = z.sum(dim=(1, 2))/(z.shape[1]*z.shape[2])
        w = w.view(1, 1, z.shape[0])
        w = self.Linear0(w)
        w = torch.relu(w)
        w = self.Linear1(w).view(z.shape[0], 1, 1)
        #print(w.shape)

        w= self.softmax(self.leakrelu(w))
        z = w * z
        #z= z.view(z.shape[1], z.shape[0]*z.shape[2])
        return z.sum(dim=0)



# class GAttention(nn.Module):

#     def __init__(self, input_dims, hidden_dim=300, hidden_dim2= 100):

#         super(GAttention, self).__init__()
#         #self.alpha = 0.2 
#         #self.Linear = nn.Linear(kernel_dim, kernel_dim)

#         self.Q = torch.nn.ModuleList()
#         for input_dim in input_dims:
#             self.Q.append(nn.Linear(input_dim, hidden_dim))

        
#         self.Linear0 = nn.Linear(len(input_dims), hidden_dim2)
#         self.Linear1 = nn.Linear(hidden_dim2, len(input_dims))


#         #self.leakrelu = nn.LeakyReLU(self.alpha)
#         self.softmax = nn.Softmax(dim=1)

#     #     self.init_weights()

#     # def init_weights(self):
#     #     for m in self.modules():
#     #         if isinstance(m, nn.Conv2d):
#     #             init.kaiming_normal_(m.weight, mode='fan_out')
#     #             if m.bias is not None:
#     #                 init.constant_(m.bias, 0)
#     #         elif isinstance(m, nn.BatchNorm2d):
#     #             init.constant_(m.weight, 1)
#     #             init.constant_(m.bias, 0)
#     #         elif isinstance(m, nn.Linear):
#     #             init.normal_(m.weight, std=0.1)
#     #             if m.bias is not None:
#     #                 init.constant_(m.bias, 0)

        
#     def forward(self, feature_list, K_WL):
#         new_feature_list = []

#         for i in range(len(feature_list)):
#             Q = self.Q[i](feature_list[i])
#             new_feature_list.append(Q)
        
#         z = torch.stack(new_feature_list, dim=0)
#         w = z.sum(dim=(1, 2))/(z.shape[1]*z.shape[2])
#         w = w.view(1, 1, z.shape[0])
#         w = self.Linear0(w)
#         w = torch.relu(w)
#         w = self.Linear1(w)
#         w= self.softmax(w).view(z.shape[0], 1, 1)
#         #print(K_WL.shape)

#         K = w * K_WL
        


#         K = torch.sum(K, dim=0)
#         return K


if __name__ == '__main__':
    input = torch.randn( 2,3, 3)
    print(input)
    sa = GAttention(2)
    output = sa(input)


    print(output.shape)
    print(output)



