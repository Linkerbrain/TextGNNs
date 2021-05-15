import numpy as np
from config import config
from collections import Counter

def create_idx_mapping(list):
    index_mapping = {}

    for i, element in enumerate(list):
        index_mapping[element] = i

    return index_mapping

def build_word_vocab(docs, min_count=1, extra_tokens=["___UNK___"]):
    all_words = [word for doc in docs for word in doc]
    if min_count == 1:
        unique_words = list(set(all_words)) + extra_tokens
        return create_idx_mapping(unique_words)

    word_counter = Counter(all_words)

    common_words = []
    for (word, count) in word_counter.items():
        if count >= min_count:
            common_words.append(word)

    return create_idx_mapping(common_words + extra_tokens)

def build_label_vocab(labels):
    classes = sorted(list(set(labels)))

    return create_idx_mapping(classes)

def make_vocab(docs, labels):
    """
    Creates a vocab of the documents and labels
        a vocab is represented as a mapping from element to index
        (for quick parsing)
    """
    word_vocab = build_word_vocab(docs, min_count=config["min_word_count"])
    label_vocab = build_label_vocab(labels)

    print("[vocab] Created a vocab of %i words, %i labels" % (len(word_vocab), len(label_vocab)))

    return word_vocab, label_vocab
