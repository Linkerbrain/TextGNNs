import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCN2Conv

class GCN2(torch.nn.Module):
    def __init__(self, input_shape, output_shape, 
                    hidden_channels, num_layers, dropout, alpha,
                     shared_weights, use_edge_weights):
        super(GCN2, self).__init__()

        self.lins = torch.nn.ModuleList()
        
        self.start_with_lin = input_shape != hidden_channels

        if self.start_with_lin:
            self.lins.append(Linear(input_shape, hidden_channels))
        self.lins.append(Linear(hidden_channels, output_shape))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(channels=hidden_channels, alpha=alpha, theta=0.5, layer=layer + 1,
                         shared_weights=shared_weights, normalize=True))

        self.dropout = dropout
        self.use_edge_weights = use_edge_weights
        
    def forward(self, x, edge_index, edge_weights):
        if torch.isnan(x).any():
            print("Received a NAN value at the start...")
            raise AssertionError("NAN given to network...?!")

        x = F.dropout(x, self.dropout, training=self.training)

        if self.start_with_lin:
            x = x_0 = self.lins[0](x).relu()
        else:
            x_0 = x

        if torch.isnan(x).any():
            print("Received a NAN value after first linear layer")
            raise AssertionError("NAN?!")
        
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, self.dropout, training=self.training)
            if self.use_edge_weights:
                x = conv(x, x_0, edge_index, edge_weight=edge_weights)

                if torch.isnan(x).any():
                    print("Received a NAN value after conv layer %i" % i)
                    raise AssertionError("NAN?!")
        
            else:
                x = conv(x, x_0, edge_index)
                # print("F:", x)
            x = x.relu()
            # print("G:", x)

        x = F.dropout(x, self.dropout, training=self.training)
        # print("H:", x)
        x = self.lins[-1](x)
        # print("I:", x)
        if torch.isnan(x).any():
            print("Received a NAN value after final linear layer")
            raise AssertionError("NAN?!")

        return x
