from pipeline.load_data import load_data
from pipeline.vocab import make_vocab
from graph_methods.create_graph import create_graphs
from pipeline.dataset import DocumentGraphDataset
from models.create_model import create_model
from pipeline.trainer import Trainer
from config import config

import numpy as np

"""
Trains and Tests according to the config specified in config.py
"""

def main():
    docs, labels, tvt_idx = load_data()

    dataset = DocumentGraphDataset(docs, labels, tvt_idx)

    model = create_model(dataset)

    trainer = Trainer(dataset, model)

    best_val_loss = float('inf')
    time_since_best = 0
    best_val_loss_acc = 0
    for i in range(config["epochs"]):
        train_loss, val_loss = trainer.train_epoch()
        # if i % config["test_every"] == 0 and i > 0:
        test_acc = trainer.test()
        print("[epoch %02d] Train loss %.4f, Val loss %.4f, Test Acc %.4f" % (i, train_loss, val_loss, test_acc))
        # else:
        #     print("[epoch %02d] Train loss %.4f, Val loss %.4f" % (i, train_loss, val_loss))

        # Early stopping
        time_since_best += 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            time_since_best = 0
            best_val_loss_acc = test_acc

        if config["terminate_early"] and time_since_best >= config["terminate_patience"]:
            test_acc = trainer.test()
            print("\n[RESULT!] Final test score: ", best_val_loss_acc)
            break


if __name__ == "__main__":
    main()