import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv

from config import config

class SAGE(torch.nn.Module):
    def __init__(self, input_shape, output_shape, 
                    layer_dims, dropout):
        super(SAGE, self).__init__()

        all_layer_dims = [input_shape] + layer_dims + [output_shape]
        self.num_layers = len(all_layer_dims) - 1

        self.dropout = dropout

        self.convs = torch.nn.ModuleList()

        for i in range(self.num_layers):
            self.convs.append(SAGEConv(all_layer_dims[i], all_layer_dims[i+1]))

    def forward(self, x, adjs):
        """
        Use a neighboursampler to make adjs which are samples from big graph
        """
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)

            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def inference(self, x_all, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all