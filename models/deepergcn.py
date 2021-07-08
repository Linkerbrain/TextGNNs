import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GENConv

class DeeperGCN(torch.nn.Module):
    def __init__(self, input_shape, output_shape, 
                    layer_dims, dropout, aggr, t, learn_t, norm):
        super(DeeperGCN, self).__init__()

        all_layer_dims = [input_shape] + layer_dims + [output_shape]
        self.num_layers = len(all_layer_dims) - 1

        self.dropout = dropout

        self.convs = torch.nn.ModuleList()

        for i in range(self.num_layers):
            self.convs.append(GENConv(all_layer_dims[i], all_layer_dims[i+1], aggr=aggr, t=t, learn_t=learn_t, norm=norm))
        

    def forward(self, x, edge_index, edge_weights):
        for i in range(self.num_layers):
            if i > 0:
                x = F.dropout(x, training=self.training)

            x = self.convs[i](x, edge_index)

            if i != self.num_layers-1:
                x = nn.ReLU()(x)

        return x