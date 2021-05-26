config = {
    "experiment_name" : "text_gcn_graph_comparison",
    "test_name" : "placeholder",

    # Data Settings
    "data_base_path" : r'./clean_data/',
    "dataset" : "reutersENmin5",
    "indices" : "3_80lab1600unlab", # 'inplace' splits it according to settings below (otherwise from disk)

    "idx_split" : [80, 20, 10],

    # Transductive or Inductive
    "ductive" : "trans",

    # Graph Vocab Settings
    "initial_repr" : "bag_of_words", # onehot or bag_of_words
    "repr_type" : "sparse_tensor", # 'sparse_tensor' or 'index'
    'unique_document_entries' : True,

    # Graph Settings
    "graph_method" : {
        "name" : "pmi_tfidf",
        "kwargs" : {
            "window_size" : 10,
            "wordword_edges" : True,
            "worddoc_edges" : True,
            "docdoc_edges" : True,
            "docdoc_min_weight" : 0.3
        }
    },

    # Model Settings
    "embedding_layer" : None, # Size of feature to embed to, else "None"
    "model" : {
        "name" : "gcn",
        "kwargs" : {
            "layer_dims" : [80],
            "dropout" : 0.5,
            "use_edge_weights" : True,
            "custom_impl" : False,
        }
    },

    # Inductive Settings
    "batch_size" : 128,
    "inductive_head" : {
        "layer_dims" : [32],
        "kwargs" : {
            "pooling" : "both", # 'max', 'mean' or 'both'
        }
    },

    # GraphSAGE Settings
    "sampled_training" : False,
    "sample_batch_size" : 1024,
    "sample_sizes" : [50, 10],

    # Training Settings
    "try_gpu" : True,
    "terminate_early" : True,
    "terminate_patience" : 8, # stop if validation loss has not reached new high in x episodes
    "epochs" : 200,
    "lr" : 0.01,
    "test_every" : 1,
}

def update_config(dic):
    for key, value in dic.items():
        config[key] = value


"""
Config dump to copy and paste from
    "model" : {
        "name" : "sage",
        "kwargs" : {
            "layer_dims" : [80],
            "dropout" : 0.5
        }
    },

    "graph_method" : {
        "name" : "pmi_tfidf",
        "kwargs" : {
            "window_size" : 10
        }
    },
    "graph_method" : {
        "name" : "co_occurence",
        "kwargs" : {
            "window_size" : 4
        }
    },
    "model" : {
        "name" : "gcn",
        "kwargs" : {
            "layer_dims" : [80],
            "dropout" : 0.1,
            "use_edge_weights" : False,
        }
    },

    "model" : {
        "name" : "simplegcn",
        "kwargs" : {
            "num_hops" : 200,
            "dropout" : 0.1
        }
    },

    "model" : {
        "name" : "gat",
        "kwargs" : {
            "layer_dims" : [40],
            "num_heads" : 4,
            "concat" : True,
            "dropout" : 0.1
        }
    },













"""