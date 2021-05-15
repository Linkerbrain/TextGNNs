import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, input_shape, output_shape, 
                    layer_dims, dropout, num_heads, concat):
        super(GAT, self).__init__()

        all_layer_dims = [input_shape] + layer_dims + [output_shape]
        self.num_layers = len(all_layer_dims) - 1

        self.dropout = dropout

        self.convs = torch.nn.ModuleList()

        for i in range(self.num_layers):
            output_per_head = all_layer_dims[i+1] // num_heads if concat else all_layer_dims[i+1]
            self.convs.append(GATConv(all_layer_dims[i], output_per_head, heads=num_heads, dropout=dropout, concat=concat))
        

    def forward(self, x, edge_index, edge_weights):
        for i in range(self.num_layers):
            if i > 0:
                x = F.dropout(x, training=self.training)

            x = self.convs[i](x, edge_index)

            if i != self.num_layers-1:
                x = nn.ReLU()(x)

        return x