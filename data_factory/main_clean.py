import os
from clean_tools import split_and_clean
from collections import Counter

FROM = r"C:\Users\Lodewijk\Desktop\scriptie2\GNNdocs\data_factory\raw_data\reuters.train.10000.en"
TO = r"C:\Users\Lodewijk\Desktop\scriptie2\GNNdocs\clean_data\reuters_en_min5.txt"

MIN_WORD_FREQUENCY = 5

# 5485

labels = []
docs = []

with open(FROM, 'r') as input_f:
    for i, line in enumerate(input_f):
        info = line.split("\t")

        label = info[0]
        doc = info[1]

        labels.append(label)
        docs.append(doc)

# split and clean
fixed_docs = []
for doc in docs:
    fixed_docs.append(split_and_clean(doc))

# count frequencies
if MIN_WORD_FREQUENCY > 1:
    print("Counting frequencies..")
    frequencies = Counter()

    for doc in fixed_docs:
        frequencies.update(doc)

    # remove low frequency words
    print("removing out of vocab words")
    for i, doc in enumerate(fixed_docs):
        fixed_docs[i] = [word for word in doc if frequencies[word] > MIN_WORD_FREQUENCY]

with open(TO, 'w', encoding="utf8") as output_f:
    for label, doc in zip(labels, fixed_docs):
        sentence = ""
        for word in doc:
            sentence += word + " "
        sentence = sentence[:-1]
        output_f.write(label + "\t" + sentence + "\n")
