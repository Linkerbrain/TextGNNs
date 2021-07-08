"""
Experiment #3: The value of classifying transductively

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
train_amount = 80 # 360 * 4 + 40 * 4
test_amount = "OFFICIAL" # 300 * 4
unlab_amount = 0 # 0 * 4

languages = ['english', 'french', 'german', 'japanese', 'chinese', 'italian', 'russian', 'spanish']

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
    "experiment_name" : "scriptie_5_COUNTITALL",
    "ductive" : "trans",
    "initial_repr" : "onehot",
    "repr_type" : "sparse_tensor",
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
    "terminate_early" : False,
    "terminate_patience" : 10,
    "epochs" : 100,
}

models = [
    {
        "test_name" : "SimpleGCN",
        "initial_repr" : "bag_of_words",
        "repr_type" : "sparse_tensor",
        'unique_bag_entries' : True,
        'min_count_for_bag' : 1,
        'binary_bag' : False,

        "graph_method" : {
            "name" : "pmi_tfidf",
            "kwargs" : {
                "window_size" : 10,
                "wordword_edges" : True,
                "worddoc_edges" : True,
                "docdoc_edges" : False,
                "average_docdoc_conns" : None,
                "average_wordword_conns" : 70,
            }
        },

        "embedding_layer" : None,
        "linear_layer" : 100,
        "model" : {
            "name" : "simplegcn",
            "kwargs" : {
                "num_hops" : 2,
                "dropout" : 0.5
            }
        },
    },
    # {
    #     "test_name" : "Bag of Words",
    #     "initial_repr" : "bag_of_words",
    #     "repr_type" : "sparse_tensor",
    #     'unique_bag_entries' : True,
    #     'min_count_for_bag' : 1,
    #     'binary_bag' : False,

    #     "graph_method" : {
    #         "name" : "pmi_tfidf",
    #         "kwargs" : {
    #             "window_size" : 10,
    #             "wordword_edges" : True,
    #             "worddoc_edges" : True,
    #             "docdoc_edges" : False,
    #             "average_docdoc_conns" : None,
    #             "average_wordword_conns" : 70,
    #         }
    #     },

    #     "embedding_layer" : None,
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
    {
        "test_name" : "TextGCN",
        "initial_repr" : "onehot",
        "repr_type" : "sparse_tensor",
        'unique_bag_entries' : True,
        'min_count_for_bag' : 1,
        'binary_bag' : True,

        "graph_method" : {
            "name" : "pmi_tfidf",
            "kwargs" : {
                "window_size" : 10,
                "wordword_edges" : True,
                "worddoc_edges" : True,
                "docdoc_edges" : False,
                "average_docdoc_conns" : None,
                "average_wordword_conns" : None,
            }
        },

        "linear_layer" : None,
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
        for language in languages:
            info_to_save = {
                "seed" : seed,
                "train_amount" : train_amount,
                "unlab_amount" : unlab_amount,
                "test_amount" : test_amount,
                "language" : language
            }

            data_name = "reuters_%s_FULL_min5" % language
            indices_name = str(seed) + "_" + str(train_amount) + "train" + str(unlab_amount) + "unlab" + str(test_amount) + "test"
            update_config({"dataset":data_name, "indices" : indices_name})
            docs, labels, tvt_idx = load_data()

            for i in range(repeats_per_seed):
                results = test_all_models(results, docs, labels, tvt_idx, info_to_save)

if __name__ == "__main__":
    main()
