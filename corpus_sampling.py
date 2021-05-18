from pipeline.load_data import load_data
from pipeline.vocab import make_vocab
from graph_methods.create_graph import create_graphs
from pipeline.dataset import DocumentGraphDataset
from models.create_model import create_model
from pipeline.trainer import Trainer
from config import config

import numpy as np
import random
import pandas as pd

"""
Trains and Tests according to the config specified in config.py
"""

def main():
    # load data
    docs, labels, tvt_idx = load_data()
    train_idx, val_idx, test_idx = tvt_idx
    
    model = None

    results = {
        "epochs" : [],
        "average_train_loss" : [],
        "average_val_loss" : [],
        "average_test_acc" : [],
        "split_amount" : [],
        "acc" : []
    }
    
    # create model with vocab for entire dataset
    dataset = DocumentGraphDataset(docs, labels, tvt_idx)
    model = create_model(dataset)
    trainer = Trainer(dataset, model)

    for i in range(config["epochs"]):
        number_of_splits = random.randint(1, 40)

        # split in x random divisions
        split_train_amount = len(train_idx) // number_of_splits
        split_val_amount = len(val_idx) // number_of_splits
        split_test_amount = len(test_idx) // number_of_splits

        print("Splitting %i segments, %i %i %i" % (number_of_splits, split_train_amount, split_val_amount, split_test_amount))

        # train on different random splits
        random.shuffle(train_idx)
        random.shuffle(val_idx)
        random.shuffle(test_idx)

        train_losses = []
        val_losses = []
        test_accs = []

        for split_i in range(number_of_splits):
            split_train_idx = train_idx[split_i * split_train_amount : (split_i+1) * split_train_amount]
            split_val_idx = val_idx[split_i * split_val_amount : (split_i+1) * split_val_amount]
            split_test_idx = test_idx[split_i * split_test_amount : (split_i+1) * split_test_amount]

            # this ain't right
            dataset = DocumentGraphDataset(docs, labels, (split_train_idx, split_val_idx, split_test_idx))

            if model is None:
                model = create_model(dataset)
                trainer = Trainer(dataset, model)
            else:
                trainer.update_data(dataset)

            train_loss, val_loss = trainer.train_epoch()
            test_acc = trainer.test()

            print("split %02d: (%.4f, %.4f, ! %.4f !), " % (split_i, train_loss, val_loss, test_acc), end="")
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            test_accs.append(test_acc)
        
        # test on entire graph
        dataset = DocumentGraphDataset(docs, labels, tvt_idx)
        trainer.update_data(dataset)
        test_acc = trainer.test()

        print("\n\n[epoch %02d] Test Acc on entire dataset %.4f\n\n" % (i, test_acc))

        # summary
        results["epochs"].append(i)
        results["average_train_loss"].append(float(sum(train_losses)/len(train_losses)))
        results["average_val_loss"].append(float(sum(val_losses)/len(val_losses)))
        results["average_test_acc"].append(float(sum(test_accs)/len(test_accs)))
        results["split_amount"].append(number_of_splits)
        results["acc"].append(test_acc)

        df = pd.DataFrame(results)
        df.to_csv('./results/'+config['experiment_name']+'.csv')


if __name__ == "__main__":
    main()