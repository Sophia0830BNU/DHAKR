import numpy as np
import torch
from torch import nn
from torch.nn import init


class Attention(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):

        super(Attention, self).__init__()

        self.alpha = 0.2 
        self.Linear00_WL = nn.Linear(input_dim, hidden_dim)
        self.Linear01_WL = nn.Linear(hidden_dim, output_dim)

        self.leakrelu = nn.LeakyReLU(self.alpha)
        self.softmax = nn.Softmax(dim=1)
        #self.dropout = nn.Dropout(0.5)


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
    #             torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):

        x1 = torch.sum(x, dim=0, keepdim=True)/x.shape[0]
        x2 = self.Linear00_WL(x1)
        x3 = torch.relu(x2)
        #x3 = self.dropout(x3)
        x4_WL = self.Linear01_WL(x3)
        attention = self.softmax(self.leakrelu(x4_WL))
        #print(attention)
        #attention = self.dropout(attention)
        x5 = attention * x
        #out = torch.relu(x5)

        
        return x5


if __name__ == '__main__':
    input = torch.randn(2, 20)
    sa =Attention(input_dim = 20, hidden_dim=15, output_dim=20)
    output = sa(input)


    print(output.shape)
    print(output)



