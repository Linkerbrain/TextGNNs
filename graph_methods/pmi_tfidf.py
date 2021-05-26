from .graph_utils import count_co_occurences, create_idx_mapping, docname

from config import config

from collections import Counter, defaultdict
from math import log
    
def pmi_tfidf_graph(docs, window_size=10, wordword_edges=True, worddoc_edges=True, docdoc_edges=True, docdoc_min_weight=0):
    all_words = [word for doc in docs for word in doc]
    all_unique_words = list(set(all_words))

    # each unique word and docs get a vertex
    num_words = len(all_unique_words)
    num_docs = len(docs)
    num_nodes = num_words + num_docs

    nodes =  [docname(i) for i in range(num_docs)] + all_unique_words
    word2idx = create_idx_mapping(all_unique_words, offset=num_docs)

    rows = []
    columns = []
    weights = []

    ### Word to word edges based on PMI

    if wordword_edges:
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
            
            word_a_id =  word2idx[word_a] # np.random.randint(len(docs), len(docs)+len(all_unique_words))
            word_b_id = word2idx[word_b]

            # add twice for symmetry
            rows += [word_a_id, word_b_id]
            columns += [word_b_id, word_a_id]
            weights += [pmi, pmi]

    ### Document to words edges based on TDIDF

    if worddoc_edges:
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
            word_id = word2idx[word]

            rows += [doc_id, word_id]
            columns += [word_id, doc_id]
            weights += [tfidf, tfidf]

    if docdoc_edges:
        word_in_docs = defaultdict(set)
        for i, doc in enumerate(docs):
            for word in doc:
                word_in_docs[word].add(i)

        overlapping_words = defaultdict(int)
        for word, doc_set in word_in_docs.items():
            doc_list = list(doc_set)
            doc_list_len = len(doc_list)
            for i in range(doc_list_len):
                for j in range(i+1, doc_list_len):
                    if (doc_list[i], doc_list[j]) in overlapping_words:
                        overlapping_words[(doc_list[i], doc_list[j])] += 1 / doc_list_len
                    else:
                        overlapping_words[(doc_list[j], doc_list[i])] += 1 / doc_list_len

        # combine and calculate pmi
        num_docdoc_edges = 0
        for ((doc_a_idx, doc_b_idx), count) in overlapping_words.items():
            if count < docdoc_min_weight:
                continue

            num_docdoc_edges += 1

            # add twice for symmetry
            rows += [doc_a_idx, doc_b_idx]
            columns += [doc_b_idx, doc_a_idx]
            weights += [count, count]

        print("NUM OF DOCDOC EDGES: %i, AVERAGE: %.2f" % (num_docdoc_edges, num_docdoc_edges/num_docs))

    ### Self edges (could also be added with add self loops=True)

    for i in range(num_nodes):
        rows.append(i)
        columns.append(i)
        weights.append(1)

    edges = ([rows, columns], weights)

    return [(edges, nodes)]