from config import config

from .pmi_tfidf import pmi_tfidf_graph
from .co_occurence import co_occurence_graph

import numpy as np

available_graph_methods = {
    "pmi_tfidf" : {
        "ductive" : "trans",
        "function" : pmi_tfidf_graph
    },
    "co_occurence" : {
        "ductive" : "in",
        "function" : co_occurence_graph
    }
}

def create_graphs(docs):
    # check if graph method is implemented
    if not config["graph_method"]["name"] in available_graph_methods.keys():
        raise NotImplementedError("[create_graph] The graph method \"%s\" is not implemented!" % config["graph_method"]["name"])
    
    # check if graph method is the right type
    if available_graph_methods[config["graph_method"]["name"]]["ductive"] != config["ductive"]:
        raise AssertionError("[create_graph] The graph method does not match to the in/transductiveness of the config")

    # create graph
    graphs, vocab = available_graph_methods[config["graph_method"]["name"]]["function"](docs=docs, **config["graph_method"]["kwargs"])

    print(graph_summary(graphs))

    return graphs, vocab

def graph_summary(graphs):
    graph_count = len(graphs)
    node_average = np.mean([len(nodes) for (edges, nodes) in graphs])
    edge_average = np.mean([len(edges[1]) for (edges, nodes) in graphs])

    if graph_count == 1:
        return "[create_graph] Created a graph consisting of %i nodes and %i edges" % (node_average, edge_average)

    return "[create_graph] Created %i graphs consisting of %i nodes and %i edges on average" % (graph_count, node_average, edge_average)