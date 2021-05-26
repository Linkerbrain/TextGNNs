import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool

from config import config

from .gcn import GCN
from .simple_gcn import SimpleGCN
from .gat import GAT
from .sage import SAGE


available_models = {
    "gcn" : {
        "class" : GCN,
    },
    "simplegcn" : {
        "class" : SimpleGCN,
    },
    "gat" : {
        "class" : GAT,
    },
    "sage" : {
        "class" : SAGE,
    }
}

def create_model(dataset):
    input_shape = dataset.vocab_size
    output_shape = dataset.num_labels

    embed_layer = config["embedding_layer"]
    if embed_layer and config["repr_type"] != "index":
        raise AssertionError("[model] An embeddings layer needs the input as indices, make sure config['repr_type'] == index")

    classify_graph = config["ductive"] == "in"
    head_settings = config["inductive_head"]

    if not config["model"]["name"] in available_models:
        raise NotImplementedError("[create_model] The model %s is not implemented!" % (config["model"]["name"]))

    core_model = available_models[config["model"]["name"]]["class"]
    core_model_kwargs = config["model"]["kwargs"]

    sampled_training = config["sampled_training"]

    model = ModelWrapper(input_shape, embed_layer, core_model, core_model_kwargs, output_shape, classify_graph, head_settings, sampled_training)

    return model

class ModelWrapper(nn.Module):
    def __init__(self, input_shape, embed_layer, 
                    core_model, core_model_kwargs, 
                    output_shape,
                    classify_graph, head_settings,
                    sampled_training):
        super(ModelWrapper, self).__init__()

        # Embedding layer
        if embed_layer:
            self.embedding_layer = nn.Embedding(input_shape, embed_layer)
            core_model_input = embed_layer
        else:
            self.embedding_layer = None
            core_model_input = input_shape

        # Graph classification head for inductive training
        if classify_graph:
            MLPlayers = head_settings["layer_dims"] + [output_shape]

            core_model_output = head_settings["layer_dims"][0]
            self.graph_classifier = MLPHead(layer_dims=MLPlayers, **head_settings["kwargs"])
        else:
            self.graph_classifier = None
            core_model_output = output_shape

        # core model
        self.core = core_model(core_model_input, core_model_output, **core_model_kwargs)

        self.sampled_training = sampled_training

        print("[model] Succesfully created model")

    def forward(self, data):
        x, edge_index, edge_weights = data.x, data.edge_index, data.edge_attr

        if self.embedding_layer:
            x = self.embedding_layer(x).squeeze()

        x = self.core(x, edge_index, edge_weights)

        if self.graph_classifier:
            return self.graph_classifier(x, data.batch)
        
        return x

    # def forward(self, x, adjs):
    #     # handle sampled graph
        
    #     if self.embedding_layer:
    #         x = self.embedding_layer(x).squeeze()

    #     x = self.core(x, adjs)

    #     return x


    def inference(self, x_all, subgraph_loader, device):
        if not self.sampled_training:
            raise NotImplementedError("[model] Inference is for sampled training with for example graphsage")
        
        if self.embedding_layer:
            x_all = self.embedding_layer(x_all).squeeze()

        x = self.core.inference(x_all, subgraph_loader, device)

        return x


class MLPHead(nn.Module):
    def __init__(self, layer_dims, pooling):
        super(MLPHead, self).__init__()
        self.pooling = pooling

        self.num_layers = len(layer_dims)-1

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0 and self.pooling == "both":
                linear_layer = torch.nn.Linear(layer_dims[i]*2, layer_dims[i+1])
            else:
                linear_layer = torch.nn.Linear(layer_dims[i], layer_dims[i+1])
            self.layers.append(linear_layer)


    def forward(self, x, batch):
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'both':
            x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        else:
            raise NotImplementedError("[model inductive head] The pooling method %s has not been implemented!" % self.pooling)

        for i in range(self.num_layers):
            if i > 0:
                x = F.dropout(x, training=self.training)
            x = self.layers[i](x)
            if i < self.num_layers-1:
                x = nn.ReLU()(x)

        return x