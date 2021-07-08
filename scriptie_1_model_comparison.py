"""
Experiment #1: The value of classifying transductively
 1 layer GCN 

 2 layer,  10 channel GCN  
 2 layer,  30 channel GCN  
 2 layer, 100 channel GCN

 3 layer,  10 channel GCN  
 3 layer,  30 channel GCN  
 3 layer, 100 channel GCN  

 8 layer,  10 channel GCNII
16 layer,  20 channel GCNII
16 layer, 100 channel GCNII
64 layer, 100 channel GCNII

 2 layer, 20 channel, 2 Headed GAT
 2 layer, 20 channel, 4 Headed GAT
 2 layer, 80 channel, 4 Headed GAT

 2 layer, 20 channel, 2 Headed GATv2
 2 layer, 20 channel, 4 Headed GATv2
 2 layer, 80 channel, 4 Headed GATv2


 1 hop SimpleGCN
 2 hop SimpleGCN
 4 hop SimpleGCN
10 hop,  10 channel  SimpleGCN
"""

from pipeline.load_data import load_data
from pipeline.vocab import make_vocab
from graph_methods.create_graph import create_graphs
from pipeline.dataset import DocumentGraphDataset
from models.create_model import create_model
from pipeline.trainer import Trainer

from other_models.fasttext import evaluate_fasttext

from config import config, update_config

import gc
import torch
import sys
import numpy as np
import pandas as pd

"""
Trains and Tests according to the config specified in config.py
"""
train_amounts = [80, 1600]
unlab_amounts = [0]
test_amounts = ["OFFICIAL"]

seeds = range(5, 10)
repeats_per_seed = 1

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_current_config(docs, labels, tvt_idx, verbose=True):
    dataset = DocumentGraphDataset(docs, labels, tvt_idx)

    model = create_model(dataset)

    # paras = count_parameters(model)
    # print("\t\"", config["test_name"], "\" : ", paras, sep="")

    # return 0

    trainer = Trainer(dataset, model)

    best_val_loss = float('inf')
    best_model_test = 0
    time_since_best = 0
    for i in range(config["epochs"]):
        train_loss, val_loss = trainer.train_epoch()
        test_acc = trainer.test()

        if verbose:
            print(config["indices"] + " [epoch %02d] Train loss %.4f, Val loss %.4f, Test Acc %.4f" % (i, train_loss, val_loss, test_acc))

        # Early stopping
        time_since_best += 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_test = test_acc
            time_since_best = 0

        if config["terminate_early"] and time_since_best >= config["terminate_patience"]:
            print("\n ## [RESULT!] %s achieved final test score at epoch %i: %.4f ## \n" % (config["test_name"], i-time_since_best, best_model_test))
            break

    del dataset
    del model
    del trainer

    gc.collect()
    torch.cuda.empty_cache()

    return best_model_test

base_config = {
    "dataset" : "reuters_english_FULL_min5", # "dataset" : "reuters_english_FULL_min5", "reutersENmin5"
    "experiment_name" : "scriptie_1_BUNCH_OF_2_LAYERGCNS",
    "graph_method" : {
        "name" : "pmi_tfidf",
        "kwargs" : {
            "window_size" : 10,
            "wordword_edges" : True,
            "worddoc_edges" : True,
            "docdoc_edges" : False,
            "average_docdoc_conns" : None, # or None for pmi > 0
            "average_wordword_conns" : None, # or None for pmi > 0
        }
    },
    "initial_repr" : "onehot",
    "repr_type" : "index",
    "embedding_layer" : 100,

    "sampled_training" : False,
    "lr" : 0.02,
    "terminate_early" : False,
    "terminate_patience" : 15,
    "epochs" : 140,
}

models = [
    # {
    #     "test_name" : "1 layer,  100 channel GCNII",
    #     "model" : {
    #         "name" : "gcn2",
    #         "kwargs" : {
    #             "hidden_channels" : 100,
    #             "num_layers" : 1,
    #             "dropout" : 0.6,
    #             "alpha" : 0.1,
    #             "shared_weights" : False,
    #             "use_edge_weights" : True,
    #         }
    #     },
    # },
    # {
    #     "test_name" : "2 layer,  100 channel GCNII",
    #     "model" : {
    #         "name" : "gcn2",
    #         "kwargs" : {
    #             "hidden_channels" : 100,
    #             "num_layers" : 2,
    #             "dropout" : 0.6,
    #             "alpha" : 0.1,
    #             "shared_weights" : False,
    #             "use_edge_weights" : True,
    #         }
    #     },
    # },
    # {
    #     "test_name" : "3 layer,  100 channel GCNII",
    #     "model" : {
    #         "name" : "gcn2",
    #         "kwargs" : {
    #             "hidden_channels" : 100,
    #             "num_layers" : 3,
    #             "dropout" : 0.6,
    #             "alpha" : 0.1,
    #             "shared_weights" : False,
    #             "use_edge_weights" : True,
    #         }
    #     },
    # },
    # {
    #     "test_name" : "4 layer,  100 channel GCNII",
    #     "model" : {
    #         "name" : "gcn2",
    #         "kwargs" : {
    #             "hidden_channels" : 100,
    #             "num_layers" : 4,
    #             "dropout" : 0.6,
    #             "alpha" : 0.1,
    #             "shared_weights" : False,
    #             "use_edge_weights" : True,
    #         }
    #     },
    # },
    # {
    #     "test_name" : " 1 layer, 100 channel GCN ",
    #     "model" : {
    #         "name" : "gcn",
    #         "kwargs" : {
    #             "layer_dims" : [],
    #             "dropout" : 0.5,
    #             "use_edge_weights" : True,
    #             "custom_impl" : True
    #         }
    #     },
    # },
    # {
    #     "test_name" : " 2 layer,  10 channel GCN  ",
    #     "model" : {
    #         "name" : "gcn",
    #         "kwargs" : {
    #             "layer_dims" : [10],
    #             "dropout" : 0.5,
    #             "use_edge_weights" : True,
    #             "custom_impl" : True
    #         }
    #     },
    # },
    # {
    #     "test_name" : " 2 layer,  30 channel GCN  ",
    #     "model" : {
    #         "name" : "gcn",
    #         "kwargs" : {
    #             "layer_dims" : [30],
    #             "dropout" : 0.5,
    #             "use_edge_weights" : True,
    #             "custom_impl" : True
    #         }
    #     },
    # },
    # {
    #     "test_name" : " 2 layer, 100 channel GCN  ",
    #     "model" : {
    #         "name" : "gcn",
    #         "kwargs" : {
    #             "layer_dims" : [100],
    #             "dropout" : 0.5,
    #             "use_edge_weights" : True,
    #             "custom_impl" : True
    #         }
    #     },
    # },
    # {
    #     "test_name" : " 2 layer, 300 channel GCN  ",
    #     "model" : {
    #         "name" : "gcn",
    #         "kwargs" : {
    #             "layer_dims" : [300],
    #             "dropout" : 0.5,
    #             "use_edge_weights" : True,
    #             "custom_impl" : True
    #         }
    #     },
    # },
    # {
    #     "test_name" : " 3 layer,  10 channel GCN  ",
    #     "model" : {
    #         "name" : "gcn",
    #         "kwargs" : {
    #             "layer_dims" : [10, 10],
    #             "dropout" : 0.5,
    #             "use_edge_weights" : True,
    #             "custom_impl" : True
    #         }
    #     },
    # },
    # {
    #     "test_name" : " 3 layer,  30 channel GCN  ",
    #     "model" : {
    #         "name" : "gcn",
    #         "kwargs" : {
    #             "layer_dims" : [30, 30],
    #             "dropout" : 0.5,
    #             "use_edge_weights" : True,
    #             "custom_impl" : True
    #         }
    #     },
    # },
    # {
    #     "test_name" : " 3 layer, 100 channel GCN  ",
    #     "model" : {
    #         "name" : "gcn",
    #         "kwargs" : {
    #             "layer_dims" : [100, 100],
    #             "dropout" : 0.5,
    #             "use_edge_weights" : True,
    #             "custom_impl" : True
    #         }
    #     },
    # },
    # {
    #     "test_name" : " 8 layer,  10 channel GCNII",
    #     "model" : {
    #         "name" : "gcn2",
    #         "kwargs" : {
    #             "hidden_channels" : 10,
    #             "num_layers" : 8,
    #             "dropout" : 0.6,
    #             "alpha" : 0.1,
    #             "shared_weights" : False,
    #             "use_edge_weights" : True,
    #         }
    #     },
    # },
    # {
    #     "test_name" : "16 layer,  20 channel GCNII",
    #     "model" : {
    #         "name" : "gcn2",
    #         "kwargs" : {
    #             "hidden_channels" : 20,
    #             "num_layers" : 16,
    #             "dropout" : 0.6,
    #             "alpha" : 0.1,
    #             "shared_weights" : False,
    #             "use_edge_weights" : True,
    #         }
    #     },
    # },
    # {
    #     "test_name" : "16 layer, 100 channel GCNII",
    #     "model" : {
    #         "name" : "gcn2",
    #         "kwargs" : {
    #             "hidden_channels" : 100,
    #             "num_layers" : 16,
    #             "dropout" : 0.6,
    #             "alpha" : 0.1,
    #             "shared_weights" : False,
    #             "use_edge_weights" : True,
    #         }
    #     },
    # },
    # {
    #     "test_name" : "64 layer, 100 channel GCNII",
    #     "model" : {
    #         "name" : "gcn2",
    #         "kwargs" : {
    #             "hidden_channels" : 100,
    #             "num_layers" : 64,
    #             "dropout" : 0.6,
    #             "alpha" : 0.1,
    #             "shared_weights" : False,
    #             "use_edge_weights" : True,
    #         }
    #     },
    # },
    # {
    #     "test_name" : " 8 layer,  10 channel shared weight GCNII",
    #     "model" : {
    #         "name" : "gcn2",
    #         "kwargs" : {
    #             "hidden_channels" : 10,
    #             "num_layers" : 8,
    #             "dropout" : 0.6,
    #             "alpha" : 0.1,
    #             "shared_weights" : True,
    #             "use_edge_weights" : True,
    #         }
    #     },
    # },
    # {
    #     "test_name" : "16 layer,  20 channel shared weight GCNII",
    #     "model" : {
    #         "name" : "gcn2",
    #         "kwargs" : {
    #             "hidden_channels" : 20,
    #             "num_layers" : 16,
    #             "dropout" : 0.6,
    #             "alpha" : 0.1,
    #             "shared_weights" : True,
    #             "use_edge_weights" : True,
    #         }
    #     },
    # },
    # {
    #     "test_name" : "16 layer, 100 channel shared weight GCNII",
    #     "model" : {
    #         "name" : "gcn2",
    #         "kwargs" : {
    #             "hidden_channels" : 100,
    #             "num_layers" : 16,
    #             "dropout" : 0.6,
    #             "alpha" : 0.1,
    #             "shared_weights" : True,
    #             "use_edge_weights" : True,
    #         }
    #     },
    # },
    # {
    #     "test_name" : "64 layer, 100 channel shared weight GCNII",
    #     "model" : {
    #         "name" : "gcn2",
    #         "kwargs" : {
    #             "hidden_channels" : 100,
    #             "num_layers" : 64,
    #             "dropout" : 0.6,
    #             "alpha" : 0.1,
    #             "shared_weights" : True,
    #             "use_edge_weights" : True,
    #         }
    #     },
    # },
    # {
    #     "test_name" : " 2 layer, 20 channel, 2 Headed GAT",
    #     "model" : {
    #         "name" : "gat",
    #         "kwargs" : {
    #             "layer_dims" : [20],
    #             "num_heads" : 2,
    #             "concat" : True,
    #             "dropout" : 0.6
    #         }
    #     },
    # },
    # {
    #     "test_name" : " 2 layer, 20 channel, 4 Headed GAT",
    #     "model" : {
    #         "name" : "gat",
    #         "kwargs" : {
    #             "layer_dims" : [20],
    #             "num_heads" : 4,
    #             "concat" : True,
    #             "dropout" : 0.6
    #         }
    #     },
    # },
    # {
    #     "test_name" : " 2 layer, 80 channel, 4 Headed GAT",
    #     "model" : {
    #         "name" : "gat",
    #         "kwargs" : {
    #             "layer_dims" : [80],
    #             "num_heads" : 4,
    #             "concat" : True,
    #             "dropout" : 0.6
    #         }
    #     },
    # },
    # {
    #     "test_name" : " 2 layer, 20 channel, 2 Headed GATv2",
    #     "model" : {
    #         "name" : "gatv2",
    #         "kwargs" : {
    #             "layer_dims" : [20],
    #             "num_heads" : 2,
    #             "concat" : True,
    #             "dropout" : 0.6
    #         }
    #     },
    # },
    # {
    #     "test_name" : " 2 layer, 20 channel, 4 Headed GATv2",
    #     "model" : {
    #         "name" : "gatv2",
    #         "kwargs" : {
    #             "layer_dims" : [20],
    #             "num_heads" : 4,
    #             "concat" : True,
    #             "dropout" : 0.6
    #         }
    #     },
    # },
    # {
    #     "test_name" : " 2 layer, 80 channel, 4 Headed GATv2",
    #     "model" : {
    #         "name" : "gatv2",
    #         "kwargs" : {
    #             "layer_dims" : [80],
    #             "num_heads" : 4,
    #             "concat" : True,
    #             "dropout" : 0.6
    #         }
    #     },
    # },
    # {
    #     "test_name" : " 1 hop SimpleGCN",
    #     "model" : {
    #         "name" : "simplegcn",
    #         "kwargs" : {
    #             "num_hops" : 1,
    #             "dropout" : 0.5
    #         }
    #     },
    # },
    # #### --- CHAOS ---
    # {
    #     "embedding_layer" : 20,
    #     "linear_layer" : None,
    #     "repr_type" : "index",
    #     "test_name" : "20 emb 2 hop SimpleGCN",
    #     "model" : {
    #         "name" : "simplegcn",
    #         "kwargs" : {
    #             "num_hops" : 2,
    #             "dropout" : 0.5
    #         }
    #     },
    # },
    # {
    #     "embedding_layer" : 40,
    #     "linear_layer" : None,
    #     "repr_type" : "index",
    #     "test_name" : "40 emb 2 hop SimpleGCN",
    #     "model" : {
    #         "name" : "simplegcn",
    #         "kwargs" : {
    #             "num_hops" : 2,
    #             "dropout" : 0.5
    #         }
    #     },
    # },
    # {
    #     "embedding_layer" : 100,
    #     "linear_layer" : None,
    #     "repr_type" : "index",
    #     "test_name" : "100 emb 2 hop SimpleGCN",
    #     "model" : {
    #         "name" : "simplegcn",
    #         "kwargs" : {
    #             "num_hops" : 2,
    #             "dropout" : 0.5
    #         }
    #     },
    # },
    # {
    #     "embedding_layer" : 200,
    #     "linear_layer" : None,
    #     "repr_type" : "index",
    #     "test_name" : "200 emb 2 hop SimpleGCN",
    #     "model" : {
    #         "name" : "simplegcn",
    #         "kwargs" : {
    #             "num_hops" : 2,
    #             "dropout" : 0.5
    #         }
    #     },
    # },
    # {
    #     "embedding_layer" : None,
    #     "linear_layer" : 20,
    #     "repr_type" : "sparse_tensor",
    #     "test_name" : "20 LIN 2 hop SimpleGCN",
    #     "model" : {
    #         "name" : "simplegcn",
    #         "kwargs" : {
    #             "num_hops" : 2,
    #             "dropout" : 0.5
    #         }
    #     },
    # },
    # {
    #     "embedding_layer" : None,
    #     "linear_layer" : 40,
    #     "repr_type" : "sparse_tensor",
    #     "test_name" : "40 LIN 2 hop SimpleGCN",
    #     "model" : {
    #         "name" : "simplegcn",
    #         "kwargs" : {
    #             "num_hops" : 2,
    #             "dropout" : 0.5
    #         }
    #     },
    # },
    # {
    #     "embedding_layer" : None,
    #     "linear_layer" : 100,
    #     "repr_type" : "sparse_tensor",
    #     "test_name" : "100 LIN 2 hop SimpleGCN",
    #     "model" : {
    #         "name" : "simplegcn",
    #         "kwargs" : {
    #             "num_hops" : 2,
    #             "dropout" : 0.5
    #         }
    #     },
    # },
    # {
    #     "embedding_layer" : None,
    #     "linear_layer" : 200,
    #     "repr_type" : "sparse_tensor",
    #     "test_name" : "200 LIN 2 hop SimpleGCN",
    #     "model" : {
    #         "name" : "simplegcn",
    #         "kwargs" : {
    #             "num_hops" : 2,
    #             "dropout" : 0.5
    #         }
    #     },
    # },
    # {
    #     "test_name" : " 2 hop SimpleGCN",
    #     "model" : {
    #         "name" : "simplegcn",
    #         "kwargs" : {
    #             "num_hops" : 2,
    #             "dropout" : 0.5
    #         }
    #     },
    # },
    # {
    #     "test_name" : " 3 hop SimpleGCN",
    #     "model" : {
    #         "name" : "simplegcn",
    #         "kwargs" : {
    #             "num_hops" : 3,
    #             "dropout" : 0.5
    #         }
    #     },
    # },
    # {
    #     "test_name" : " 4 hop SimpleGCN",
    #     "model" : {
    #         "name" : "simplegcn",
    #         "kwargs" : {
    #             "num_hops" : 4,
    #             "dropout" : 0.5
    #         }
    #     },
    # },
    # {
    #     "test_name" : " 5 hop SimpleGCN",
    #     "model" : {
    #         "name" : "simplegcn",
    #         "kwargs" : {
    #             "num_hops" : 5,
    #             "dropout" : 0.5
    #         }
    #     },
    # },
    # {
    #     "test_name" : "10 hop SimpleGCN",
    #     "model" : {
    #         "name" : "simplegcn",
    #         "kwargs" : {
    #             "num_hops" : 10,
    #             "dropout" : 0.5
    #         }
    #     },
    # },
    # #### GCNS ####
    {
        "embedding_layer" : None,
        "linear_layer" : 100,
        "repr_type" : "sparse_tensor",
        "test_name" : "2 layer, 10 channel GCN + LINEAR",
        "model" : {
            "name" : "gcn",
            "kwargs" : {
                "layer_dims" : [10],
                "dropout" : 0.5,
                "use_edge_weights" : True,
                "custom_impl" : True
            }
        },
    },
    {
        "embedding_layer" : None,
        "linear_layer" : 100,
        "repr_type" : "sparse_tensor",
        "test_name" : "2 layer,  30 channel GCN + LINEAR",
        "model" : {
            "name" : "gcn",
            "kwargs" : {
                "layer_dims" : [30],
                "dropout" : 0.5,
                "use_edge_weights" : True,
                "custom_impl" : True
            }
        },
    },
    {
        "embedding_layer" : None,
        "linear_layer" : 100,
        "repr_type" : "sparse_tensor",
        "test_name" : "2 layer, 100 channel GCN + LINEAR",
        "model" : {
            "name" : "gcn",
            "kwargs" : {
                "layer_dims" : [100],
                "dropout" : 0.5,
                "use_edge_weights" : True,
                "custom_impl" : True
            }
        },
    },
    {
        "embedding_layer" : 100,
        "linear_layer" : None,
        "repr_type" : "index",
        "test_name" : "2 layer, 10 channel GCN + EMBEDDING",
        "model" : {
            "name" : "gcn",
            "kwargs" : {
                "layer_dims" : [10],
                "dropout" : 0.5,
                "use_edge_weights" : True,
                "custom_impl" : True
            }
        },
    },
    {
        "embedding_layer" : 100,
        "linear_layer" : None,
        "repr_type" : "index",
        "test_name" : "2 layer,  30 channel GCN + EMBEDDING",
        "model" : {
            "name" : "gcn",
            "kwargs" : {
                "layer_dims" : [30],
                "dropout" : 0.5,
                "use_edge_weights" : True,
                "custom_impl" : True
            }
        },
    },
    {
        "embedding_layer" : 100,
        "linear_layer" : None,
        "repr_type" : "index",
        "test_name" : "2 layer, 100 channel GCN + EMBEDDING",
        "model" : {
            "name" : "gcn",
            "kwargs" : {
                "layer_dims" : [100],
                "dropout" : 0.5,
                "use_edge_weights" : True,
                "custom_impl" : True
            }
        },
    },
    {
        "embedding_layer" : None,
        "linear_layer" : None,
        "repr_type" : "sparse_tensor",
        "test_name" : "2 layer, 10 channel GCN + RAW",
        "model" : {
            "name" : "gcn",
            "kwargs" : {
                "layer_dims" : [10],
                "dropout" : 0.5,
                "use_edge_weights" : True,
                "custom_impl" : True
            }
        },
    },
    {
        "embedding_layer" : None,
        "linear_layer" : None,
        "repr_type" : "sparse_tensor",
        "test_name" : "2 layer,  30 channel GCN + RAW",
        "model" : {
            "name" : "gcn",
            "kwargs" : {
                "layer_dims" : [30],
                "dropout" : 0.5,
                "use_edge_weights" : True,
                "custom_impl" : True
            }
        },
    },
    {
        "embedding_layer" : None,
        "linear_layer" : None,
        "repr_type" : "sparse_tensor",
        "test_name" : "2 layer, 100 channel GCN + RAW",
        "model" : {
            "name" : "gcn",
            "kwargs" : {
                "layer_dims" : [100],
                "dropout" : 0.5,
                "use_edge_weights" : True,
                "custom_impl" : True
            }
        },
    },
]

def test_all_models(results, docs, labels, tvt_idx, save_info):
    update_config(base_config)

    for model in models:
        update_config(model)

        print("Working on " + config["test_name"] + ", " + config["ductive"] + "ductively!")

        try:
            if config['test_name'] == 'fasttext':
                result = evaluate_fasttext(docs, labels, tvt_idx, verbose=True)
            else:
                result = evaluate_current_config(docs, labels, tvt_idx, verbose=True)
            results["test_name"].append(config["test_name"])
            results["acc"].append(result)
            for key, value in save_info.items():
                if key in results:
                    results[key].append(value)
                else:
                    results[key] = [value]
            save_dic(results)
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print(config["test_name"], "gone wrong! Error:", e)
    
    return results

def save_dic(results):
    pd.DataFrame(results).to_csv("./results/"+config["experiment_name"]+".csv")

def main():
    update_config(base_config)

    results = {
        "test_name" : [],
        "acc" : []
    }

    for seed in seeds:
        for train_amount in train_amounts:
            for unlab_amount in unlab_amounts:
                for test_amount in test_amounts:
                    info_to_save = {
                        "seed" : seed,
                        "train_amount" : train_amount,
                        "unlab_amount" : unlab_amount,
                        "test_amount" : test_amount
                    }

                    indices_name = str(seed) + "_" + str(train_amount) + "train" + str(unlab_amount) + "unlab" + str(test_amount) + "test"
                    update_config({"indices" : indices_name})
                    docs, labels, tvt_idx = load_data()

                    for i in range(repeats_per_seed):
                        results = test_all_models(results, docs, labels, tvt_idx, info_to_save)

if __name__ == "__main__":
    main()
