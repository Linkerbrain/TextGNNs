import scipy.sparse as sp
from math import log

from collections import defaultdict
import pandas as pd

from os.path import join, exists
from tqdm import tqdm

def get_vocab(text_list):
    word_freq = defaultdict(int)
    for doc_words in text_list:
        # words = doc_words.split()
        for word in doc_words:
            word_freq[word] += 1
    return word_freq

def build_word_doc_edges(doc_list):
    # build all docs that a word is contained in
    words_in_docs = defaultdict(set)
    for i, doc_words in enumerate(doc_list):
        for word in doc_words:
            words_in_docs[word].add(i)

    word_doc_freq = {}
    for word, doc_list in words_in_docs.items():
        word_doc_freq[word] = len(doc_list)

    return words_in_docs, word_doc_freq

def build_edges(doc_list, word_id_map, vocab, word_doc_freq, window_size=20):
    # constructing all windows
    windows = []
    for doc_words in doc_list:
        words = doc_words
        doc_length = len(words)
        if doc_length <= window_size:
            windows.append(words)
        else:
            for i in range(doc_length - window_size + 1):
                window = words[i: i + window_size]
                windows.append(window)
    # constructing all single word frequency
    word_window_freq = defaultdict(int)
    for window in windows:
        appeared = set()
        for word in window:
            if word not in appeared:
                word_window_freq[word] += 1
                appeared.add(word)
    # constructing word pair count frequency
    word_pair_count = defaultdict(int)
    for window in tqdm(windows):
        for i in range(1, len(window)):
            for j in range(i):
                word_i = window[i]
                word_j = window[j]
                word_i_id = word_id_map[word_i]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_count[(word_i_id, word_j_id)] += 1
                word_pair_count[(word_j_id, word_i_id)] += 1

    row = []
    col = []
    weight = []

    # pmi as weights
    num_docs = len(doc_list)
    num_window = len(windows)
    for word_id_pair, count in tqdm(word_pair_count.items()):
        i, j = word_id_pair[0], word_id_pair[1]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(num_docs + i)
        col.append(num_docs + j)
        weight.append(pmi)

    # frequency of document word pair
    doc_word_freq = defaultdict(int)
    for i, doc_words in enumerate(doc_list):
        words = doc_words
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = (i, word_id)
            doc_word_freq[doc_word_str] += 1

    for i, doc_words in enumerate(doc_list):
        words = doc_words
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            word_id = word_id_map[word]
            freq = doc_word_freq[(i, word_id)]
            row.append(i)
            col.append(num_docs + word_id)
            idf = log(1.0 * num_docs /
                      word_doc_freq[vocab[word_id]])
            weight.append(freq * idf)
            doc_word_set.add(word)

    number_nodes = num_docs + len(vocab)

    # add self loops
    for i in range(number_nodes):
        row.append(i)
        col.append(i)
        weight.append(1)

    adj_mat = sp.csr_matrix((weight, (row, col)), shape=(number_nodes, number_nodes))
    adj = adj_mat + adj_mat.T.multiply(adj_mat.T > adj_mat) - adj_mat.multiply(adj_mat.T > adj_mat)
    return adj

def docname(i):
    return "___DOC"+str(i)+"___"

def text_gcn_graph(docs, window_size=10):

    # get vocab
    word_freq = get_vocab(docs)
    vocab = list(word_freq.keys())

    # get number of docs each word is in
    words_in_docs, word_doc_freq = build_word_doc_edges(docs)
    word_id_map = {word: i for i, word in enumerate(vocab)}

    sparse_graph = build_edges(docs, word_id_map, vocab, word_doc_freq, window_size)

    # convert csr graph back to edge indexes
    rows = []
    for i in range(0, len(sparse_graph.indptr)-1):
        for _ in range(sparse_graph.indptr[i], sparse_graph.indptr[i+1]):
            rows.append(i)

    columns = sparse_graph.indices
    weights = sparse_graph.data

    nodes = [docname(i) for i in range(len(docs))] + vocab

    nodename2idx = {n : i for i, n in enumerate(nodes)}

    edges = ([rows, columns], weights)

    return [(edges, nodes)]

from collections import Counter, defaultdict

def count_co_occurences(doc, window_size, return_count=False, co_occurences=None):
    
    if not co_occurences:
        co_occurences = defaultdict(int)
    window_count = 0
    for i, w in enumerate(doc):
        for j in range(i + 1, min(i + window_size + 1, len(doc))):
            window_count += 1
            if (doc[i], doc[j]) in co_occurences:
                co_occurences[(doc[i], doc[j])] += 1 # Could add weighting based on distance
            else:
                co_occurences[(doc[j], doc[i])] += 1

    if return_count:
        return co_occurences, window_count
    return co_occurences
    
def create_idx_mapping(list):
    index_mapping = {}

    for i, element in enumerate(list):
        index_mapping[element] = i

    return index_mapping
    
def pmi_tfidf_graph(docs, window_size=10):
    all_words = [word for doc in docs for word in doc]
    all_unique_words = list(set(all_words))

    doc_names = [docname(i) for i in range(len(docs))]


    nodes = doc_names+all_unique_words
    nodename2idx = create_idx_mapping(nodes)

    rows = []
    columns = []
    weights = []

    ### Word to word edges based on PMI

    # count how many times words occur
    freq = Counter(all_words)
    word_count = len(all_words)
    # count how many times words occur together
    total_window_count = 0
    co_occurences = defaultdict(int)
    for doc in docs:
        co_occurences, window_count = count_co_occurences(doc, window_size=window_size, return_count=True, co_occurences=co_occurences)
        total_window_count += window_count

    # combine and calculate pmi
    for ((word_a, word_b), count) in co_occurences.items():
        # pmi = log ( p(x,y) / (p(x)p(y)) )
        pmi = log((count / total_window_count) / ((freq[word_a] * freq[word_b]) / (word_count ** 2))) # TODO: not tested if correct

        if pmi <= 0:
            continue
        
        word_a_id =  nodename2idx[word_a] # np.random.randint(len(docs), len(docs)+len(all_unique_words))
        word_b_id = nodename2idx[word_b]

        # add twice for symmetry
        rows += [word_a_id, word_b_id]
        columns += [word_b_id, word_a_id]
        weights += [pmi, pmi]

    ### Document to words edges based on TDIDF

    # count in how many docs each word is
    in_document_count = defaultdict(int)
    # count how often each word appears in each doc
    frequency_in_doc = defaultdict(int)

    for i, doc in enumerate(docs):
        already_counted_words = set()
        for word in doc:
            frequency_in_doc[(i, word)] += 1

            if word not in already_counted_words:
                in_document_count[word] += 1
                already_counted_words.add(word)

    doc_count = len(docs)
    # save the edges
    for ((doc_idx, word), count) in frequency_in_doc.items():
        # tf = count in doc / total doc length
        tf = count # / len(docs[doc_idx])
        # idf = log (N / docs with that word)
        idf = log(doc_count / in_document_count[word])

        tfidf = tf * idf

        # add twice for symmetry
        doc_id = doc_idx
        word_id = nodename2idx[word]

        rows += [doc_id, word_id]
        columns += [word_id, doc_id]
        weights += [tfidf, tfidf]

    ### Self edges (could also be added with add self loops=True)

    for i in range(len(nodes)):
        rows.append(i)
        columns.append(i)
        weights.append(1)

    edges = ([rows, columns], weights)

    return [(edges, nodes)]

if __name__ == "__main__":
    docs = [["I", "ate", "a", "hotdog"], ["I", "love", "hotdog", "because", "fuck", "you"],
                ["another", "example", "doc"], ["I", "present", "a", "fourth", "doc"]]

    [(([rows_theirs, columns_theirs], weights_theirs), nodes_theirs)], nodename2idx = text_gcn_graph(docs, window_size=3)

    [(([rows, columns], weights), nodes)], vocab = pmi_tfidf_graph(docs, window_size=2)

    debug_dic_theirs = {}
    for r, c, w in zip(rows_theirs, columns_theirs, weights_theirs):
        debug_dic_theirs[(nodes_theirs[r],nodes_theirs[c])] = w

    debug_dic = {}
    for r, c, w in zip(rows, columns, weights):
        debug_dic[(nodes[r],nodes[c])] = w

    for key, value in debug_dic_theirs.items():
        ours = "Not There!"
        if key in debug_dic:
            ours = "%.4f" % debug_dic[key]
        print(key, "\t%.4f\t" % value, ours)

    for key, value in debug_dic.items():
        if key not in debug_dic_theirs:
            print(key, "\tNot There!\t%.4f" % value)

    print(nodes_theirs, "\n", nodes)
