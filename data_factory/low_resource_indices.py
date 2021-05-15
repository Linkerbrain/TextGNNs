from collections import defaultdict
from random import Random

DATA_FILE = r'C:\Users\Lodewijk\Desktop\scriptie2\GNNdocs\clean_data\reutersENmin5.txt'
BASE_PATH = r'C:\Users\Lodewijk\Desktop\scriptie2\GNNdocs\clean_data\indices\reutersENmin5_'

TRAIN_AMOUNTS = [18, 36, 90, 180, 360, 720]
VAL_AMOUNTS = [2, 4, 10, 20, 40, 80]
TEST_AMOUNTS = [10, 25, 50, 75, 100, 150, 250, 500]

for seed in range(10):
    # get labels
    label_indices = defaultdict(list)
    with open(DATA_FILE, 'r') as input_f:
        for i, line in enumerate(input_f):
            info = line.split("\t")

            label = info[0]

            label_indices[label].append(i)

    # shuffle labels
    for lab, labels in label_indices.items():
        Random(seed).shuffle(labels)
        print(lab, len(labels))

    # save
    train_part = max([t+v for t,v in zip(TRAIN_AMOUNTS, VAL_AMOUNTS)])

    if max(TEST_AMOUNTS) + train_part > min([len(l) for l in label_indices.values()]):
        raise AssertionError("There are not enough labels for the speicfied amounts! (%i > %i)" % (max(TEST_AMOUNTS) + train_part, min([len(l) for l in label_indices.values()])))

    for train, val in zip(TRAIN_AMOUNTS, VAL_AMOUNTS):
        for test in TEST_AMOUNTS:
            total_label = (train+val) * len(label_indices)
            total_unlabel = test * len(label_indices)
            
            indice_name = str(seed) + "_" + str(total_label) + "lab" + str(total_unlabel) + "unlab"

            train_indices = []
            val_indices = []
            test_indices = []
            for labels in label_indices.values():
                train_indices += labels[:train]
                val_indices += labels[train:train+val]
                
                test_indices += labels[train_part:train_part+test]

            with open(BASE_PATH + indice_name + "_train.txt", "w") as open_f:
                for i in train_indices:
                    open_f.write(str(i)+",")
            with open(BASE_PATH + indice_name + "_val.txt", "w") as open_f:
                for i in val_indices:
                    open_f.write(str(i)+",")
            with open(BASE_PATH + indice_name + "_test.txt", "w") as open_f:
                for i in test_indices:
                    open_f.write(str(i)+",")