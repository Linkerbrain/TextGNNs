from config import config
from graph_methods.create_graph import create_graphs
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import numpy as np

class DocumentGraphDataset():
    def __init__(self, docs, labels, tvt_idx, force_vocab=None):
        (self.train_idx, self.val_idx, self.test_idx) = tvt_idx
        
        self.transductive = config["ductive"] == "trans"

        if force_vocab:
            docs = self.remove_out_of_vocab(docs, force_vocab)

        if self.transductive:
            self.graph, self.vocab_size, self.num_labels = self.create_corpus_graph(docs, labels, force_vocab)
        else:
            self.graphs, self.vocab_size, self.num_labels = self.create_doc_graphs(docs, labels)

        print("[dataset] Succesfully prepared PyTorch Geometric data")

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

        print("Removed %i/%i words because they were not in vocab" % (failed, total))
        return docs

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
        graphs, vocab = create_graphs(docs)

        if force_vocab:
            vocab = force_vocab

        self.vocab = vocab

        ((edge_indices, edge_weights), nodes) = graphs[0]

        torch_edges = torch.tensor(edge_indices, dtype=torch.long)
        torch_edge_weights = torch.tensor(edge_weights, dtype=torch.float)

        node_numbers, vocab_size = self.nodenames_to_tensors(nodes, vocab)
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