import numpy as np
import torch
from torch import nn
from torch.nn import init
import math


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



if __name__ == '__main__':
    input = torch.randn( 2,3, 3)
    print(input)
    sa = GAttention(2)
    output = sa(input)


    print(output.shape)
    print(output)



