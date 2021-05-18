from pipeline.load_data import load_data
from pipeline.vocab import make_vocab
from graph_methods.create_graph import create_graphs
from pipeline.dataset import DocumentGraphDataset
from models.create_model import create_model
from pipeline.trainer import Trainer
from config import config, update_config

import gc
import torch
import sys
import numpy as np
import pandas as pd

"""
Trains and Tests according to the config specified in config.py
"""
labeled_amounts = [400] # 80
unlabeled_amounts = [1000] # [40, 100, 200, 300, 400, 600, 1000] # 40
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
repeats_per_seed = 1

def evaluate_current_config(docs, labels, tvt_idx, verbose=True):
    dataset = DocumentGraphDataset(docs, labels, tvt_idx)

    model = create_model(dataset)

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

transductive_settings = {
    "ductive" : "trans",
    "graph_method" : {
        "name" : "pmi_tfidf",
        "kwargs" : {
            "window_size" : 10
        }
    },
    "lr" : 0.02,
    "terminate_patience" : 8,
}

inductive_settings = {
    "ductive" : "in",
    "graph_method" : {
        "name" : "co_occurence",
        "kwargs" : {
            "window_size" : 4
        }
    },
    "lr" : 0.001,
    "terminate_patience" : 8,
}

models = [
    {
        "test_name" : "2 layer GCN with edge weights",
        "embedding_layer" : 300,
        "features_as_onehot" : False, 
        "model" : {
            "name" : "gcn",
            "kwargs" : {
                "layer_dims" : [80],
                "dropout" : 0.4,
                "use_edge_weights" : True,
                "custom_impl" : False
            }
        },
    },
    {
        "test_name" : "2 layer GCN without edge weights",
        "embedding_layer" : 300,
        "features_as_onehot" : False, 
        "model" : {
            "name" : "gcn",
            "kwargs" : {
                "layer_dims" : [80],
                "dropout" : 0.4,
                "use_edge_weights" : False,
                "custom_impl" : False
            }
        },
    },
    # {
    #     "test_name" : "300 20 GCN",
    #     "embedding_layer" : 300,
    #     "features_as_onehot" : False,
    #     "model" : {
    #         "name" : "gcn",
    #         "kwargs" : {
    #             "layer_dims" : [20],
    #             "dropout" : 0.25,
    #             "use_edge_weights" : False,
    #             "custom_impl" : False
    #         }
    #     },
    # },
    # {
    #     "test_name" : "200 80 GCN",
    #     "embedding_layer" : 200,
    #     "features_as_onehot" : False,
    #     "model" : {
    #         "name" : "gcn",
    #         "kwargs" : {
    #             "layer_dims" : [80],
    #             "dropout" : 0.25,
    #             "use_edge_weights" : False,
    #             "custom_impl" : False
    #         }
    #     },
    # },
    # {
    #     "test_name" : "100 80 GCN",
    #     "embedding_layer" : 100,
    #     "features_as_onehot" : False,
    #     "model" : {
    #         "name" : "gcn",
    #         "kwargs" : {
    #             "layer_dims" : [80],
    #             "dropout" : 0.25,
    #             "use_edge_weights" : False,
    #             "custom_impl" : False
    #         }
    #     },
    # },
    # {
    #     "test_name" : "40 80 GCN",
    #     "embedding_layer" : 40,
    #     "features_as_onehot" : False,
    #     "model" : {
    #         "name" : "gcn",
    #         "kwargs" : {
    #             "layer_dims" : [80],
    #             "dropout" : 0.25,
    #             "use_edge_weights" : False,
    #             "custom_impl" : False
    #         }
    #     },
    # },
    # {
    #     "test_name" : "40 40 GCN",
    #     "embedding_layer" : 40,
    #     "features_as_onehot" : False,
    #     "model" : {
    #         "name" : "gcn",
    #         "kwargs" : {
    #             "layer_dims" : [40],
    #             "dropout" : 0.25,
    #             "use_edge_weights" : False,
    #             "custom_impl" : False
    #         }
    #     },
    # },
    # {
    #     "test_name" : "40 10 GCN",
    #     "embedding_layer" : 40,
    #     "features_as_onehot" : False,
    #     "model" : {
    #         "name" : "gcn",
    #         "kwargs" : {
    #             "layer_dims" : [10],
    #             "dropout" : 0.25,
    #             "use_edge_weights" : False,
    #             "custom_impl" : False
    #         }
    #     },
    # },
    # {
    #     "test_name" : "200 40 GCN",
    #     "embedding_layer" : 200,
    #     "features_as_onehot" : False,
    #     "model" : {
    #         "name" : "gcn",
    #         "kwargs" : {
    #             "layer_dims" : [40],
    #             "dropout" : 0.25,
    #             "use_edge_weights" : False,
    #             "custom_impl" : False
    #         }
    #     },
    # },
    # {
    #     "test_name" : "200 20 GCN",
    #     "embedding_layer" : 200,
    #     "features_as_onehot" : False,
    #     "model" : {
    #         "name" : "gcn",
    #         "kwargs" : {
    #             "layer_dims" : [20],
    #             "dropout" : 0.25,
    #             "use_edge_weights" : False,
    #             "custom_impl" : False
    #         }
    #     },
    # },
    # {
    #     "test_name" : "200 200 GCN",
    #     "embedding_layer" : 200,
    #     "features_as_onehot" : False,
    #     "model" : {
    #         "name" : "gcn",
    #         "kwargs" : {
    #             "layer_dims" : [200],
    #             "dropout" : 0.25,
    #             "use_edge_weights" : False,
    #             "custom_impl" : False
    #         }
    #     },
    # },
    # {
    #     "test_name" : "None 80 GCN",
    #     "embedding_layer" : None,
    #     "features_as_onehot" : True,
    #     "model" : {
    #         "name" : "gcn",
    #         "kwargs" : {
    #             "layer_dims" : [80],
    #             "dropout" : 0.25,
    #             "use_edge_weights" : False,
    #             "custom_impl" : False
    #         }
    #     },
    # },
    # {
    #     "test_name" : "None 200 GCN",
    #     "embedding_layer" : None,
    #     "features_as_onehot" : True,
    #     "model" : {
    #         "name" : "gcn",
    #         "kwargs" : {
    #             "layer_dims" : [200],
    #             "dropout" : 0.25,
    #             "use_edge_weights" : False,
    #             "custom_impl" : False
    #         }
    #     },
    # },
    # {
    #     "test_name" : "None 100 GCN",
    #     "embedding_layer" : None,
    #     "features_as_onehot" : True,
    #     "model" : {
    #         "name" : "gcn",
    #         "kwargs" : {
    #             "layer_dims" : [100],
    #             "dropout" : 0.25,
    #             "use_edge_weights" : False,
    #             "custom_impl" : False
    #         }
    #     },
    # },
    # {
    #     "test_name" : "None 20 GCN",
    #     "embedding_layer" : None,
    #     "features_as_onehot" : True,
    #     "model" : {
    #         "name" : "gcn",
    #         "kwargs" : {
    #             "layer_dims" : [20],
    #             "dropout" : 0.25,
    #             "use_edge_weights" : False,
    #             "custom_impl" : False
    #         }
    #     },
    # },
    # {
    #     "test_name" : "None 10 10 GCN",
    #     "embedding_layer" : None,
    #     "features_as_onehot" : True,
    #     "model" : {
    #         "name" : "gcn",
    #         "kwargs" : {
    #             "layer_dims" : [10, 10],
    #             "dropout" : 0.25,
    #             "use_edge_weights" : False,
    #             "custom_impl" : False
    #         }
    #     },
    # },
    # {
    #     "test_name" : "2 layer GCN unweighted",
    #     "model" : {
    #         "name" : "gcn",
    #         "kwargs" : {
    #             "layer_dims" : [80],
    #             "dropout" : 0.25,
    #             "use_edge_weights" : False,
    #             "custom_impl" : False
    #         }
    #     },
    # },
    # {
    #     "test_name" : "2 layer GCN no EMBED",
    #     "embedding_layer" : None,
    #     "features_as_onehot" : True,
    #     "model" : {
    #         "name" : "gcn",
    #         "kwargs" : {
    #             "layer_dims" : [80],
    #             "dropout" : 0.25,
    #             "use_edge_weights" : True,
    #             "custom_impl" : False
    #         }
    #     },
    # },
    # {
    #     "test_name" : "2 layer GCN unweighted no EMBED",
    #     "model" : {
    #         "name" : "gcn",
    #         "kwargs" : {
    #             "layer_dims" : [80],
    #             "dropout" : 0.25,
    #             "use_edge_weights" : False,
    #             "custom_impl" : False
    #         }
    #     },
    # },
    # {
    #     "test_name" : "1 layer GCN no EMBED",
    #     "model" : {
    #         "name" : "gcn",
    #         "kwargs" : {
    #             "layer_dims" : [80],
    #             "dropout" : 0.25,
    #             "use_edge_weights" : True,
    #             "custom_impl" : False
    #         }
    #     },
    # },
    # {
    #     "test_name" : "3 layer GCN no EMBED",
    #     "model" : {
    #         "name" : "gcn",
    #         "kwargs" : {
    #             "layer_dims" : [80],
    #             "dropout" : 0.25,
    #             "use_edge_weights" : True,
    #             "custom_impl" : False
    #         }
    #     },
    # },
    # {
    #     "test_name" : "3 layer GCN",
    #     "embedding_layer" : 200,
    #     "features_as_onehot" : False,
    #     "model" : {
    #         "name" : "gcn",
    #         "kwargs" : {
    #             "layer_dims" : [40, 10],
    #             "dropout" : 0.25,
    #             "use_edge_weights" : True,
    #             "custom_impl" : False
    #         }
    #     },
    # },
    # {
    #     "test_name" : "4 layer GCN",
    #     "embedding_layer" : 200,
    #     "features_as_onehot" : False,
    #     "model" : {
    #         "name" : "gcn",
    #         "kwargs" : {
    #             "layer_dims" : [80, 40, 10],
    #             "dropout" : 0.25,
    #             "use_edge_weights" : True,
    #             "custom_impl" : False
    #         }
    #     },
    # },
    # {
    #     "test_name" : "1 hop SimpleGCN",
    #     "model" : {
    #         "name" : "simplegcn",
    #         "kwargs" : {
    #             "num_hops" : 1,
    #             "dropout" : 0.25
    #         }
    #     },
    # },
    # {
    #     "test_name" : "3 hop SimpleGCN",
    #     "model" : {
    #         "name" : "simplegcn",
    #         "kwargs" : {
    #             "num_hops" : 3,
    #             "dropout" : 0.25
    #         }
    #     },
    # },
    # {
    #     "test_name" : "5 hop SimpleGCN",
    #     "model" : {
    #         "name" : "simplegcn",
    #         "kwargs" : {
    #             "num_hops" : 5,
    #             "dropout" : 0.25
    #         }
    #     },
    # },
    # {
    #     "test_name" : "4 head GAT",
    #     "model" : {
    #         "name" : "gat",
    #         "kwargs" : {
    #             "layer_dims" : [40],
    #             "num_heads" : 4,
    #             "concat" : True,
    #             "dropout" : 0.1
    #         }
    #     },
    # },
    # {
    #     "test_name" : "2 head GAT",
    #     "model" : {
    #         "name" : "gat",
    #         "kwargs" : {
    #             "layer_dims" : [40],
    #             "num_heads" : 2,
    #             "concat" : True,
    #             "dropout" : 0.1
    #         }
    #     },
    # },
    # {
    #     "test_name" : "4 head no concat GAT",
    #     "model" : {
    #         "name" : "gat",
    #         "kwargs" : {
    #             "layer_dims" : [40],
    #             "num_heads" : 4,
    #             "concat" : False,
    #             "dropout" : 0.1
    #         }
    #     },
    # },
    # {
    #     "test_name" : "8 head no concat GAT",
    #     "model" : {
    #         "name" : "gat",
    #         "kwargs" : {
    #             "layer_dims" : [80],
    #             "num_heads" : 8,
    #             "concat" : False,
    #             "dropout" : 0.1
    #         }
    #     },
    # },
    # {
    #     "test_name" : "8 head no concat GAT",
    #     "model" : {
    #         "name" : "gat",
    #         "kwargs" : {
    #             "layer_dims" : [160],
    #             "num_heads" : 8,
    #             "concat" : False,
    #             "dropout" : 0.1
    #         }
    #     },
    # }
]

def test_all_models(results, docs, labels, tvt_idx, save_info):
    for model in models:
        update_config(model)

        print("Working on " + config["test_name"] + ", " + config["ductive"] + "ductively!")

        try:
            result = evaluate_current_config(docs, labels, tvt_idx, verbose=True)
            results["test_name"].append(config["test_name"])
            results["acc"].append(result)
            results["ductive"].append(config["ductive"])
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

    results = {
        "test_name" : [],
        "acc" : [],
        "ductive" : []
    }

    for seed in seeds:
        for unlabeled_amount in unlabeled_amounts:
            for labeled_amount in labeled_amounts:
                info_to_save = {
                    "seed" : seed,
                    "unlabeled_amount" : unlabeled_amount,
                    "labeled_amount" : labeled_amount
                }

                indices_name = str(seed) + "_" + str(labeled_amount) + "lab" + str(unlabeled_amount) + "unlab"
                update_config({"indices" : indices_name})
                docs, labels, tvt_idx = load_data()

                for i in range(repeats_per_seed):
                    update_config(transductive_settings)
                    results = test_all_models(results, docs, labels, tvt_idx, info_to_save)

                    # update_config(inductive_settings)
                    # results = test_all_models(results, docs, labels, tvt_idx, info_to_save)

if __name__ == "__main__":
    main()