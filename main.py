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

        # split for special debug printing
        if config["unsupervised_loss"]:
            train_loss, val_loss, unsup_train_loss_pos, unsup_train_loss_neg, unsup_val_loss_pos, unsup_val_loss_neg, unsup_test_pos, unsup_test_neg = trainer.train_epoch()


            test_acc, test_loss, unsup_test_loss_pos, unsup_test_loss_neg = trainer.test()

            total_train = train_loss + unsup_train_loss_pos + unsup_train_loss_neg
            total_val = val_loss + unsup_val_loss_pos + unsup_val_loss_neg
            total_test = test_loss + unsup_test_loss_pos + unsup_test_loss_neg
            print("[epoch %02d] Test Acc %.4f (Trained %s-supervised)" % (i, test_acc, config['sup_mode']))
            print("\t Train Loss: %.4f (%.4f / %.4f / %.4f) (%.0f%% sup)" % (total_train, train_loss, unsup_train_loss_pos, unsup_train_loss_neg, train_loss / total_train * 100))
            print("\t Val Loss: %.4f (%.4f / %.4f / %.4f) (%.0f%% sup)" % (total_val, val_loss, unsup_val_loss_pos, unsup_val_loss_neg, val_loss / total_val * 100))
            print("\t Training on test Losses: %.4f, %.4f, %.1f%% of total" % (unsup_test_pos, unsup_test_neg, (unsup_test_pos+unsup_test_neg)/(unsup_test_pos+unsup_test_neg+total_train) * 100))
            # print("\t Test Loss: %.4f (%.4f / %.4f / %.4f) (%.0f%% sup)" % (total_test, test_loss, unsup_test_loss_pos, unsup_test_loss_neg, test_loss / total_test))
            
            val_loss = total_val
        else:
            train_loss, val_loss = trainer.train_epoch()
            test_acc = trainer.test()
            print("[epoch %02d] Train loss %.4f, Val loss %.4f, Test Acc %.4f" % (i, train_loss, val_loss, test_acc))
        
        # Early stopping
        time_since_best += 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            time_since_best = 0
            best_val_loss_acc = test_acc

        if config["terminate_early"] and time_since_best >= config["terminate_patience"]:
            
            if config["unsupervised_loss"]:
                test_acc, test_loss, unsup_test_loss_pos, unsup_test_loss_neg = trainer.test()
                print("\n[RESULT!] Final test score: ", best_val_loss_acc)
            else:
                test_acc = trainer.test()
                print("\n[RESULT!] Final test score: ", best_val_loss_acc)
            break


if __name__ == "__main__":
    main()