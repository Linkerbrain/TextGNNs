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
seeds = range(15)
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

variables_to_follow = ["test_name", "graph_method"]

configs = [
    {
        "test_name" : "their implementation (NO SELF LOOPS)",
        "graph_method" : {
            "name" : "text_gcn_impl",
            "kwargs" : {
                "window_size" : 10
            }
        },
    },
    {
        "test_name" : "my implementation",
        "graph_method" : {
            "name" : "pmi_tfidf",
            "kwargs" : {
                "window_size" : 10
            }
        },
    },
]

def save_dic(results):
    pd.DataFrame(results).to_csv("./results/"+config["experiment_name"]+".csv")

def main():
    results = {
        "seed" : [],
        "unlabeled_amount" : [],
        "labeled_amount" : [],
        "acc" : [],
    }
    for var in variables_to_follow:
        results[var] = []

    for seed in seeds:
        for unlabeled_amount in unlabeled_amounts:
            for labeled_amount in labeled_amounts:
                # Load data
                indices_name = str(seed) + "_" + str(labeled_amount) + "lab" + str(unlabeled_amount) + "unlab"
                update_config({"indices" : indices_name})
                docs, labels, tvt_idx = load_data()

                # Test configs
                for cfg in configs:
                    update_config(cfg)

                    for i in range(repeats_per_seed):
                        # print to say where we at
                        summary_string = ""
                        for var in variables_to_follow:
                            summary_string += var + ":" + str(config[var]) + ","
                        print("Working on {" + summary_string[:-1] + "}")

                        # evaluate
                        try:
                            result = evaluate_current_config(docs, labels, tvt_idx, verbose=True)

                            # save
                            results["seed"].append(seed)
                            results["unlabeled_amount"].append(unlabeled_amount)
                            results["labeled_amount"].append(labeled_amount)
                            results["acc"].append(result)
                            for var in variables_to_follow:
                                results[var].append(config[var])
                            
                            save_dic(results)
                        except KeyboardInterrupt:
                            exit()
                        except Exception as e:
                            print(summary_string, "gone wrong! Error:", e)


if __name__ == "__main__":
    main()