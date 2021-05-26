import os
import pandas as pd

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


index_path_base = r'C:\Users\Lodewijk\Desktop\scriptie2\GNNdocs\clean_data\indices/'
DATA_FILE = r'C:\Users\Lodewijk\Desktop\scriptie2\GNNdocs\clean_data\reutersENmin5.txt'

indices_to_copy = ["reutersENmin5_1_160lab600unlab"]

target_base_path = r"C:\Users\Lodewijk\Desktop\scriptie2\OfficialTextGCN\data\corpus/"

docs = []
labels = []
with open(DATA_FILE, 'r') as input_f:
    for i, line in enumerate(input_f):
        info = line.split("\t")

        doc = info[1]
        label = info[0]

        docs.append(doc)
        labels.append(label)

for index_name in indices_to_copy:
    # get indices
    indices_path = index_path_base + index_name + "_"

    train_idx = load_csv_idx(indices_path + "train.txt")
    val_idx = load_csv_idx(indices_path + "val.txt")
    test_idx = load_csv_idx(indices_path + "test.txt")

    # create subset
    subset_docs = []
    subset_label_info = []

    progress = 0
    for idx in train_idx:
        subset_docs.append(docs[idx])
        subset_label_info.append(str(progress) + "\t" + "train" + "\t" + labels[idx])
        progress += 1

    for idx in val_idx:
        subset_docs.append(docs[idx])
        subset_label_info.append(str(progress) + "\t" + "train" + "\t" + labels[idx])
        progress += 1

    for idx in test_idx:
        subset_docs.append(docs[idx])
        subset_label_info.append(str(progress) + "\t" + "test" + "\t" + labels[idx])
        progress += 1

    # write result
    sentences_path = target_base_path+index_name+"_sentences_clean.txt"
    sentences_path_2 = target_base_path+index_name+"_sentences.txt"
    labels_path = target_base_path+index_name+"_labels.txt"

    with open(sentences_path, "w") as open_sentence_f:
        for doc in subset_docs:
            open_sentence_f.write(doc)

    with open(sentences_path_2, "w") as open_sentence_f:
        for doc in subset_docs:
            open_sentence_f.write(doc)

    with open(labels_path, "w") as open_label_f:
        for label_info in subset_label_info:
            open_label_f.write(label_info + "\n")
