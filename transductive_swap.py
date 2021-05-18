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
    docs, labels, (t_idx, v_idx, test_idx) = load_data()

    in_training_test = test_idx[:len(test_idx)//2]
    out_training_test = test_idx[len(test_idx)//2:]

    # train on (train, val, in_training_test)
    train_docs = docs[t_idx+v_idx+in_training_test]
    train_labels = labels[t_idx+v_idx+in_training_test]
    train_and_intest_indices = (t_idx, v_idx, range(len(t_idx)+len(v_idx), len(train_labels)))

    a = 0
    b = 0

    # test on (in_training_test)
    only_in_training_docs = docs[t_idx[:a]+v_idx[:b]+in_training_test]
    only_in_training_labels = labels[t_idx[:a]+v_idx[:b]+in_training_test]
    only_in_training_indices = ([t_idx[:a]], [v_idx[:b]], range(a+b, len(only_in_training_labels)))

    # test on (out_training_test)
    only_out_training_docs = docs[t_idx[:a]+v_idx[:b]+out_training_test]
    only_out_training_labels = labels[t_idx[:a]+v_idx[:b]+out_training_test]
    only_out_training_indices = ([t_idx[:a]], [v_idx[:b]], range(a+b, len(only_out_training_labels)))

    # test on (train, val, out_training_test)
    train_and_outtest_docs = docs[t_idx+v_idx+out_training_test]
    train_and_outtest_labels = labels[t_idx+v_idx+out_training_test]
    train_and_outtest_indices = (t_idx, v_idx, range(len(t_idx)+len(v_idx), len(train_and_outtest_labels)))

    train_and_intest_dataset = DocumentGraphDataset(train_docs, train_labels, train_and_intest_indices)
    only_in_training_dataset = DocumentGraphDataset(only_in_training_docs, only_in_training_labels, only_in_training_indices, force_vocab=train_and_intest_dataset.vocab)
    only_out_training_dataset = DocumentGraphDataset(only_out_training_docs, only_out_training_labels, only_out_training_indices, force_vocab=train_and_intest_dataset.vocab)
    train_and_outtest_dataset = DocumentGraphDataset(train_and_outtest_docs, train_and_outtest_labels, train_and_outtest_indices, force_vocab=train_and_intest_dataset.vocab)

    model = None

    best_val_loss = float('inf')
    time_since_best = 0
    
    for i in range(config["epochs"]):
        if model is None:
            model = create_model(train_and_intest_dataset)
            trainer = Trainer(train_and_intest_dataset, model)
        else:
            trainer.update_data(train_and_intest_dataset)

        train_loss, val_loss = trainer.train_epoch()
        test_acc_train_and_intest = trainer.test()

        trainer.update_data(only_in_training_dataset)
        test_acc_only_in_training = trainer.test()

        trainer.update_data(only_out_training_dataset)
        test_acc_only_out_training = trainer.test()

        trainer.update_data(train_and_outtest_dataset)
        test_acc_train_and_outtest = trainer.test()

        print("[epoch %02d] Train loss %.4f, Val loss %.4f" % (i, train_loss, val_loss))
        print(" acc on in training test, with training docs in graph: %.4f" % (test_acc_train_and_intest))
        print(" acc on in training test, WITHOUT training docs in graph: %.4f" % (test_acc_only_in_training))
        print(" acc on out of training test, WITHOUT training docs in graph: %.4f" % (test_acc_only_out_training))
        print(" acc on out of training test, with training docs in graph: %.4f" % (test_acc_train_and_outtest))

        # Early stopping
        time_since_best += 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            time_since_best = 0

        if config["terminate_early"] and time_since_best >= config["terminate_patience"]:
            print("\n[RESULT!] Final test score: see above")
            break


if __name__ == "__main__":
    main()