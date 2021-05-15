import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SGConv

class SimpleGCN(torch.nn.Module):
    def __init__(self, input_shape, output_shape, 
                    num_hops, dropout, ):
        super(SimpleGCN, self).__init__()

        self.dropout = dropout

        self.conv = SGConv(input_shape, output_shape, K=num_hops)

    def forward(self, x, edge_index, edge_weights):
        x = F.dropout(x, training=self.training)

        x = self.conv(x, edge_index)

        return x