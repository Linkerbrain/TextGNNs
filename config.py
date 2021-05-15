config = {
    "experiment_name" : "reuters_lowresource_noedgeweights",
    "test_name" : "placeholder",

    # Data Settings
    "data_base_path" : r'./clean_data/',
    "dataset" : "reutersENmin5",
    "indices" : "0_3200lab400unlab", # 'inplace' splits it according to settings below (otherwise from disk)

    "idx_split" : [80, 20, 10],

    # Transductive or Inductive
    "ductive" : "in",

    # Graph Respresentation Settings
    "unique_document_embeddings" : True,
    "features_as_onehot" : False,

    # Graph Settings
    "graph_method" : {
        "name" : "co_occurence",
        "kwargs" : {
            "window_size" : 4
        }
    },

    # Model Settings
    "embedding_layer" : 200, # Size of feature to embed to, else "None"
    "model" : {
        "name" : "gcn",
        "kwargs" : {
            "layer_dims" : [80],
            "dropout" : 0.1,
            "use_edge_weights" : True,
            "custom_impl" : True
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

    # Training Settings
    "try_gpu" : True,
    "terminate_early" : True,
    "terminate_patience" : 8, # stop if validation loss has not reached new high in x episodes
    "epochs" : 400,
    "lr" : 0.001,
}

def update_config(dic):
    for key, value in dic.items():
        config[key] = value


"""
Config dump to copy and paste from

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