from .graph_utils import count_co_occurences, create_idx_mapping

from config import config

def co_occurence_graph(docs, window_size=2):
    # create vocab
    all_words = [word for doc in docs for word in doc]
    all_unique_words = list(set(all_words))

    nodename2idx = create_idx_mapping(all_unique_words)

    graphs = []
    for doc in docs:
        # vertex for each unique word
        unique_words = list(set(doc))
        word2vertex = {word: ix for ix, word in enumerate(unique_words)}

        # make edges
        co_occurences = count_co_occurences(doc, window_size)
        
        rows = []
        columns = []
        weights = []
        for ((word_a, word_b), count) in co_occurences.items():
            word_a_id = word2vertex[word_a]
            word_b_id = word2vertex[word_b]

            # add twice for symmetry
            rows += [word_a_id, word_b_id]
            columns += [word_b_id, word_a_id]
            weights += [count, count]

        edges = ([rows, columns], weights)
        graph = (edges, unique_words)

        graphs.append(graph)

    return graphs