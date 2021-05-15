from config import config
from random import shuffle
import numpy as np


def load_tsv_data(path):
    """
    Data should be saved as .txt,
    entries seperated by new line
        label \\t doc
    on each line
    """
    docs = []
    labels = []

    with open(path, encoding="utf8", errors="ignore") as f:
        for i, line in enumerate(f):
            content = line.split("\t")
            if len(content) != 2:
                print("[load_data] ERROR: %s has a faulty line %i, check for excess \\t or \\n" % (path, i))
            labels.append(content[0])
            docs.append(content[1][:-1].split(" "))

    return np.array(docs, dtype=object), np.array(labels)

def load_csv_idx(path):
    """
    idx should be one big line of numbers sperated by commas
    """
    idx = []
    with open(path, encoding="utf8", errors="ignore") as f:
        for i, line in enumerate(f):
            content = line.split(",")

            idx += [int(s) for s in content if s != ""]

    return idx

def load_data():
    # get data
    dataset_path = config["data_base_path"] + config["dataset"] + ".txt"
    docs, labels = load_tsv_data(dataset_path)
    
    # create new split
    if config["indices"] == "inplace":
        all_idx = list(range(len(docs)))
        shuffle(all_idx)

        train_idx = all_idx[ : config["idx_split"][0]]
        val_idx = all_idx[config["idx_split"][0] : config["idx_split"][0]+config["idx_split"][1]]
        test_idx = all_idx[config["idx_split"][0]+config["idx_split"][1] : config["idx_split"][0]+config["idx_split"][1]+config["idx_split"][2]]

    # load split from disk
    else:
        indices_path = config["data_base_path"] + "indices/" + config["dataset"] + "_" + config["indices"] + "_"

        train_idx = load_csv_idx(indices_path + "train.txt")
        val_idx = load_csv_idx(indices_path + "val.txt")
        test_idx = load_csv_idx(indices_path + "test.txt")

    all_idx = train_idx + val_idx + test_idx
    ordered_docs = docs[all_idx]
    ordered_labels = labels[all_idx]
    new_train_idx = list(range(0, len(train_idx)))
    new_val_idx = list(range(len(train_idx), len(train_idx)+len(val_idx)))
    new_test_idx = list(range(len(train_idx)+len(val_idx), len(all_idx)))
    print("[load_data] Succesfully loaded %i, %i, %i data (%i total)" % (len(new_train_idx), len(new_val_idx), len(new_test_idx), len(all_idx)))

    return ordered_docs, ordered_labels, (new_train_idx, new_val_idx, new_test_idx)