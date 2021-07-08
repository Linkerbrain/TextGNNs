"""
Experiment #1: The value of classifying transductively

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
train_amounts = [80]
unlab_amounts = [320]
test_amounts = [800]

seeds = range(2)
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

base_config = {
    "experiment_name" : "scriptie_1b",
    "graph_method" : {
        "name" : "pmi_tfidf",
        "kwargs" : {
            "window_size" : 10,
            "wordword_edges" : True,
            "worddoc_edges" : True,
            "docdoc_edges" : False,
            "docdoc_min_weight" : 0.3
        }
    },
    "initial_repr" : "bag_of_documents",
    'unique_bag_entries' : False,
    'binary_bag' : True,
    "repr_type" : "sparse_tensor",
    "sampled_training" : False,
    "lr" : 0.02,
    "terminate_early" : True,
    "terminate_patience" : 15,
}

models = [
    {
        "test_name" : "1 layer GCN",
        "embedding_layer" : None,
        "model" : {
            "name" : "gcn",
            "kwargs" : {
                "layer_dims" : [],
                "dropout" : 0.5,
                "use_edge_weights" : True,
                "custom_impl" : True
            }
        },
    },
    {
        "test_name" : "2 layer GCN (10 embedding)",
        "embedding_layer" : None,
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
        "test_name" : "2 layer GCN (20 embedding)",
        "embedding_layer" : None,
        "model" : {
            "name" : "gcn",
            "kwargs" : {
                "layer_dims" : [20],
                "dropout" : 0.5,
                "use_edge_weights" : True,
                "custom_impl" : True
            }
        },
    },
    {
        "test_name" : "2 layer GCN (50 embedding)",
        "embedding_layer" : None,
        "model" : {
            "name" : "gcn",
            "kwargs" : {
                "layer_dims" : [20],
                "dropout" : 0.5,
                "use_edge_weights" : True,
                "custom_impl" : True
            }
        },
    },
    {
        "test_name" : "2 layer GCN (100 embedding)",
        "embedding_layer" : None,
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
        "test_name" : "2 layer GCN (300 embedding)",
        "embedding_layer" : None,
        "model" : {
            "name" : "gcn",
            "kwargs" : {
                "layer_dims" : [300],
                "dropout" : 0.5,
                "use_edge_weights" : True,
                "custom_impl" : True
            }
        },
    },
    {
        "test_name" : "3 layer GCN (20, 20 embedding)",
        "embedding_layer" : None,
        "model" : {
            "name" : "gcn",
            "kwargs" : {
                "layer_dims" : [20, 20],
                "dropout" : 0.5,
                "use_edge_weights" : True,
                "custom_impl" : True
            }
        },
    },
    {
        "test_name" : "3 layer GCN (20, 100 embedding)",
        "embedding_layer" : None,
        "model" : {
            "name" : "gcn",
            "kwargs" : {
                "layer_dims" : [20, 100],
                "dropout" : 0.5,
                "use_edge_weights" : True,
                "custom_impl" : True
            }
        },
    },
    {
        "test_name" : "3 layer GCN (100, 100 embedding)",
        "embedding_layer" : None,
        "model" : {
            "name" : "gcn",
            "kwargs" : {
                "layer_dims" : [100, 100],
                "dropout" : 0.5,
                "use_edge_weights" : True,
                "custom_impl" : True
            }
        },
    },
    {
        "test_name" : "3 layer GCN (5, 20 embedding)",
        "embedding_layer" : None,
        "model" : {
            "name" : "gcn",
            "kwargs" : {
                "layer_dims" : [5, 20],
                "dropout" : 0.5,
                "use_edge_weights" : True,
                "custom_impl" : True
            }
        },
    },
    {
        "test_name" : "3 layer GCN (25, 100 embedding)",
        "embedding_layer" : None,
        "model" : {
            "name" : "gcn",
            "kwargs" : {
                "layer_dims" : [25, 100],
                "dropout" : 0.5,
                "use_edge_weights" : True,
                "custom_impl" : True
            }
        },
    },
    {
        "test_name" : "4 layer GCN (5, 10, 20 embedding)",
        "embedding_layer" : None,
        "model" : {
            "name" : "gcn",
            "kwargs" : {
                "layer_dims" : [5, 10, 20],
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
