from config import config
from graph_methods.create_graph import create_graphs
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import numpy as np
from collections import Counter
from torch_geometric.data import NeighborSampler

class DocumentGraphDataset():
    def __init__(self, docs, labels, tvt_idx, force_vocab=None):
        (self.train_idx, self.val_idx, self.test_idx) = tvt_idx
        
        self.transductive = config["ductive"] == "trans"

        if force_vocab:
            docs = self.remove_out_of_vocab(docs, force_vocab)

        self.docs = docs
        self.labels = labels

        if self.transductive:
            self.graph, self.vocab_size, self.num_labels = self.create_corpus_graph(docs, labels, force_vocab)
        else:
            self.graphs, self.vocab_size, self.num_labels = self.create_doc_graphs(docs, labels)

        print("[dataset] Succesfully prepared PyTorch Geometric data ")

        print("Vocab size: %i" % self.vocab_size)

    def remove_out_of_vocab(self, docs, vocab):
        total = 0
        failed = 0
        for i in range(len(docs)):
            fixed_doc = []
            for word in docs[i]:
                total += 1
                if word in vocab:
                    fixed_doc.append(word)
                else:
                    failed += 1

            docs[i] = fixed_doc

        print("[dataset] Removed %i/%i words because they were not in vocab" % (failed, total))
        return docs

    def embed_nodes(self, nodes, word_vocab=None):
        # give number to each word (if not already provided)
        if not word_vocab:
            word_vocab = {}
            for nodename in list(set(nodes)):
                if nodename[:6] != "___DOC":
                    word_vocab[nodename] = len(word_vocab)

        vocab_size = len(word_vocab)

        # one hot across vocab and docs
        if config["initial_repr"] == "onehot":
            idxs = []
            for nodename in nodes:
                # words map to vocab index
                if nodename[:6] != "___DOC":
                    idxs.append(word_vocab[nodename])
                # documents map after vocab
                else:
                    doc_id = int(nodename[6:-3])
                    if config["unique_document_entries"]:
                        idxs.append(vocab_size + doc_id)
                    else:
                        idxs.append(vocab_size)

            if config["repr_type"] == "index":
                feature_size = max(idxs)+1
                return torch.tensor(idxs, dtype=torch.long), feature_size

            elif config["repr_type"] == "sparse_tensor":
                feature_size = max(idxs)+1
                onehots = torch.sparse_coo_tensor((range(len(idxs)), idxs), [1 for _ in range(len(idxs))], size=(len(idxs), feature_size), dtype=torch.float)
                return onehots, feature_size

            else:
                raise NotImplementedError("[dataset] The repr type %s has not been implemented for method %s" % (config["repr_type"], config["initial_repr"]))

        # one hot tensor across all vocab words and documents
        elif config["initial_repr"] == "bag_of_words":
            node_indices = []
            feature_indices = []
            values = []
            for i, nodename in enumerate(nodes):
                # words map to vocab index (as a one hot)
                if nodename[:6] != "___DOC":
                    node_indices.append(i)
                    feature_indices.append(word_vocab[nodename])
                    values.append(1)
                # documents map to their counts of each words and a 1 at the document index
                else:
                    doc_id = int(nodename[6:-3])

                    # make bag of words
                    word_counts = Counter(self.docs[doc_id])
                    for word, count in word_counts.items():
                        node_indices.append(i)
                        feature_indices.append(word_vocab[word])
                        values.append(count)

                    # add document entry
                    if config["unique_document_entries"]:
                        node_indices.append(i)
                        feature_indices.append(vocab_size + doc_id)
                        values.append(1)
                    else:
                        node_indices.append(i)
                        feature_indices.append(vocab_size)
                        values.append(1)
                        
            if config["repr_type"] == "sparse_tensor":
                feature_size = max(feature_indices)+1
                onehots = torch.sparse_coo_tensor((node_indices, feature_indices), values, dtype=torch.float)
                return onehots, feature_size

            else:
                raise NotImplementedError("[dataset] The repr type %s has not been implemented for method %s" % (config["repr_type"], config["initial_repr"]))


        else:
            raise NotImplementedError("[dataset] The initial representation in the config (%s) is not implemented" % config["initial_repr"])

    def nodenames_to_tensors(self, nodes, vocab):
        vocab_size = max(vocab.values()) + 1
 
        idxs = []
        for node in nodes:
            if node in vocab:
                idxs.append(vocab[node])
            else:
                raise AssertionError("The generated vocab does not include node name %s" % (node))

        if config["features_as_onehot"]:
            onehots = torch.sparse_coo_tensor((range(len(idxs)), idxs), [1 for _ in range(len(idxs))], size=(len(idxs), vocab_size), dtype=torch.float)
            return onehots, vocab_size

        return torch.tensor(idxs, dtype=torch.long), vocab_size

    def labels_to_tensors(self, labels):
        unique_labels = list(set(labels))
        num_labels = len(unique_labels)

        label_mapping = {label : i for i, label in enumerate(unique_labels)}

        idxs = []
        for label in labels:
            idxs.append(label_mapping[label])

        return torch.tensor(idxs, dtype=torch.long), num_labels

    def create_corpus_graph(self, docs, labels, force_vocab=None):
        """
        Transductive preperation
        """
        graphs = create_graphs(docs)

        ((edge_indices, edge_weights), nodes) = graphs[0]

        torch_edges = torch.tensor(edge_indices, dtype=torch.long)
        torch_edge_weights = torch.tensor(edge_weights, dtype=torch.float)

        node_numbers, vocab_size = self.embed_nodes(nodes, force_vocab)
        label_numbers, num_labels = self.labels_to_tensors(labels)

        graph = Data(x=node_numbers, edge_index=torch_edges, edge_attr=torch_edge_weights, y=label_numbers)
        graph.num_nodes = len(nodes)

        return graph, vocab_size, num_labels

    def create_doc_graphs(self, docs, labels):
        """
        Inductive preperation
        """
        graphs, vocab = create_graphs(docs)
        label_numbers, num_labels = self.labels_to_tensors(labels)

        processed_graphs = []

        for graph, label in zip(graphs, label_numbers):
            ((edge_indices, edge_weights), nodes) = graph

            torch_edges = torch.tensor(edge_indices, dtype=torch.long)
            torch_edge_weights = torch.tensor(edge_weights, dtype=torch.float)

            node_numbers, vocab_size = self.nodenames_to_tensors(nodes, vocab)

            graph = Data(x=node_numbers, edge_index=torch_edges, edge_attr=torch_edge_weights, y=label)
            graph.num_nodes = len(nodes)

            processed_graphs.append(graph)

        return processed_graphs, vocab_size, num_labels

    def get_tvt_indices(self):
        return self.train_idx, self.val_idx, self.test_idx

    def get_tvt_dataloaders(self, batch_size, shuffle=True):
        train_loader = DataLoader([self.graphs[i] for i in self.train_idx], batch_size, shuffle)
        val_loader = DataLoader([self.graphs[i] for i in self.val_idx], batch_size, shuffle)
        test_loader = DataLoader([self.graphs[i] for i in self.test_idx], batch_size, shuffle)

        return train_loader, val_loader, test_loader

    def get_tvt_samplers(self, batch_size, sizes):
        train_sampler = NeighborSampler(self.graph.edge_index, node_idx=torch.tensor(self.train_idx, dtype=torch.long),
                               sizes=sizes, batch_size=batch_size, shuffle=True, num_workers=12)
        val_sampler = NeighborSampler(self.graph.edge_index, node_idx=torch.tensor(self.val_idx, dtype=torch.long),
                               sizes=sizes, batch_size=batch_size, shuffle=True, num_workers=12)
        test_sampler = NeighborSampler(self.graph.edge_index, node_idx=torch.tensor(self.test_idx, dtype=torch.long),
                               sizes=sizes, batch_size=batch_size, shuffle=True, num_workers=12)

        entire_graph_sampler = NeighborSampler(self.graph.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=batch_size, shuffle=False,
                                  num_workers=12)

        return train_sampler, val_sampler, test_sampler, entire_graph_sampler