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
train_amounts = [80, 160, 320, 640, 1600, 3200]
unlab_amounts = [0]
test_amounts = ["OFFICIAL"]

seeds = range(10)
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
    "dataset" : "reuters_english_FULL_min5",
    "experiment_name" : "scriptie_0a_ONLYTRANSDUCTIVE",
    'unique_bag_entries' : True,

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
}

models = [
    {
        
        "test_name" : "transductive GCN",
        "ductive" : "trans",
        "initial_repr" : "onehot",
        "repr_type" : "sparse_tensor",
        "graph_method" : {
            "name" : "pmi_tfidf",
            "kwargs" : {
                "window_size" : 10,
                "wordword_edges" : True,
                "worddoc_edges" : True,
                "docdoc_edges" : False,
                "average_wordword_conns" : None,
                "average_docdoc_conns" : None,
            }
        },
        "embedding_layer" : None, # Size of feature to embed to, else "None"
        "model" : {
            "name" : "gcn",
            "kwargs" : {
                "layer_dims" : [100],
                "dropout" : 0.5,
                "use_edge_weights" : True,
                "custom_impl" : True
            }
        },
        "sampled_training" : False,
        "lr" : 0.02,
        "terminate_early" : True,
        "terminate_patience" : 10,
    },
    # {
    #     "test_name" : "inductive GCN",
    #     "ductive" : "in",
    #     "initial_repr" : "onehot",
    #     "repr_type" : "index",
        
    #     "graph_method" : {
    #         "name" : "co_occurence",
    #         "kwargs" : {
    #             "window_size" : 2
    #         }
    #     },

    #     "batch_size" : 32,
    #     "inductive_head" : {
    #         "layer_dims" : [100],
    #         "kwargs" : {
    #             "pooling" : "both", # 'max', 'mean' or 'both'
    #         }
    #     },
    #     "embedding_layer" : 300, # Size of feature to embed to, else "None"
    #     "model" : {
    #         "name" : "gcn",
    #         "kwargs" : {
    #             "layer_dims" : [100],
    #             "dropout" : 0.5,
    #             "use_edge_weights" : True,
    #             "custom_impl" : True
    #         }
    #     },

    #     "sampled_training" : False,
    #     "lr" : 0.01,
    #     "epochs" : 75,
    #     "terminate_early" : True,
    #     "terminate_patience" : 10,
    # },
    # {
    #     "test_name" : "inductive GraphSage",
    #     "ductive" : "trans",
    #     "initial_repr" : "onehot",
    #     "repr_type" : "sparse_tensor",
        
    #     "graph_method" : {
    #         "name" : "pmi_tfidf",
    #         "kwargs" : {
    #             "window_size" : 10,
    #             "wordword_edges" : True,
    #             "worddoc_edges" : True,
    #             "docdoc_edges" : False,
    #             "average_wordword_conns" : None, # or None for pmi > 0
    #             "average_docdoc_conns" : None,
    #         }
    #     },

    #     "embedding_layer" : None, # Size of feature to embed to, else "None"
    #     "model" : {
    #         "name" : "sage",
    #         "kwargs" : {
    #             "layer_dims" : [20],
    #             "dropout" : 0.5
    #         }
    #     },

    #     "sampled_training" : True,
    #     "sample_batch_size" : 3, # small batch size due to memory issues
    #     "sample_sizes" : [120, 5],
    #     "sampling_num_workers" : 0,

    #     "unsupervised_loss" : False,
    #     "lr" : 0.005,
    #     "terminate_early" : False,
    #     "terminate_patience" : 10, # stop if validation loss has not reached new high in x episodes
    #     "epochs" : 140,
    # },
    {
        "test_name" : 'fasttext'
    }
]

def test_all_models(results, docs, labels, tvt_idx, save_info):
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
