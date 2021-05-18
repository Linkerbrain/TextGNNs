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
        "test_name" : "300 80 GCN",
        "embedding_layer" : 300,
        "features_as_onehot" : False, 
        "model" : {
            "name" : "gcn",
            "kwargs" : {
                "layer_dims" : [80],
                "dropout" : 0.25,
                "use_edge_weights" : False,
                "custom_impl" : False
            }
        },
    },
    {
        "test_name" : "None 200 GCN",
        "embedding_layer" : None,
        "features_as_onehot" : True,
        "model" : {
            "name" : "gcn",
            "kwargs" : {
                "layer_dims" : [200],
                "dropout" : 0.25,
                "use_edge_weights" : False,
                "custom_impl" : False
            }
        },
    },
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
            results["unique_embeddings"].append(config["unique_document_embeddings"])
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
        "ductive" : [],
        "unique_embeddings" : [],
    }
    update_config(transductive_settings)

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
                    update_config({"unique_document_embeddings" : True})
                    results = test_all_models(results, docs, labels, tvt_idx, info_to_save)
                    update_config({"unique_document_embeddings" : False})
                    results = test_all_models(results, docs, labels, tvt_idx, info_to_save)

if __name__ == "__main__":
    main()