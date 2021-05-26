import torch
import torch.nn.functional as F
import torch.nn as nn

from config import config

class Trainer():
    def __init__(self, data, model):
        self.device = "cuda" if config["try_gpu"] and torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            torch.cuda.empty_cache()

        print("[gym] Working with device:", self.device)

        self.model = model.to(self.device)

        # self.optimizer = Ranger(model.parameters(), lr=lr)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

        if config["ductive"] == "trans":
            self.training_style = "transductive"
        elif config["ductive"] == "in":
            self.training_style = "inductive"
        
        if config["sampled_training"]:
            self.training_style = "graphsage"

        self.load_data(data)

        self.epoch = 0

        self.loss = nn.CrossEntropyLoss()

    def load_data(self, data):
        if self.training_style == "transductive":
            self.graph = data.graph.to(self.device)
            self.train_idx, self.val_idx, self.test_idx = data.get_tvt_indices()
            self.train_idx = torch.tensor(self.train_idx, dtype=torch.long).to(self.device)
            self.val_idx = torch.tensor(self.val_idx, dtype=torch.long).to(self.device)
            self.test_idx = torch.tensor(self.test_idx, dtype=torch.long).to(self.device)

        elif self.training_style == "inductive":
            self.train_loader, self.val_loader, self.test_loader = data.get_tvt_dataloaders(batch_size=config["batch_size"])

        elif self.training_style == "graphsage":
            self.graph = data.graph.to(self.device)
            self.train_idx, self.val_idx, self.test_idx = data.get_tvt_indices()
            self.train_idx = torch.tensor(self.train_idx, dtype=torch.long).to(self.device)
            self.val_idx = torch.tensor(self.val_idx, dtype=torch.long).to(self.device)
            self.test_idx = torch.tensor(self.test_idx, dtype=torch.long).to(self.device)

            self.train_sampler, self.val_sampler, self.test_sampler, self.entire_graph_sampler = \
                data.get_tvt_samplers(batch_size=config["sample_batch_size"], sizes=config["sample_sizes"])
        
        else:
            raise NotImplementedError()

    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()

        # Transductive
        if self.training_style == "transductive":
            # Train
            preds = self.model(self.graph)

            train_nodes_preds = preds[self.train_idx]
            train_nodes_true = self.graph.y[self.train_idx]

            train_loss = self.loss(train_nodes_preds, train_nodes_true)

            train_loss.backward()
            self.optimizer.step()

            # Validate
            self.model.eval()
            val_nodes_preds = preds[self.val_idx]
            val_nodes_true = self.graph.y[self.val_idx]

            val_loss = self.loss(val_nodes_preds, val_nodes_true)

        # Inductive
        elif self.training_style == "inductive":
            # Train
            train_loss = 0
            for data in self.train_loader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)

                loss = self.loss(output, data.y)
                loss.backward()
                train_loss += data.num_graphs * loss.item()
                self.optimizer.step()

            train_loss /= len(self.train_loader)

            # Validate
            self.model.eval()
            val_loss = 0
            for data in self.val_loader:
                data = data.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)

                loss = self.loss(output, data.y)
                loss.backward()
                val_loss += data.num_graphs * loss.item()
                self.optimizer.step()

            val_loss /= len(self.val_loader)

        # GraphSAGE
        elif self.training_style == "graphsage":
            train_loss = 0
            val_loss = 0
            # Train
            print("[model] Training...")
            for batch_size, n_id, adjs in self.train_sampler:
                print("Batch!")
                # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
                adjs = [adj.to(self.device) for adj in adjs]

                self.optimizer.zero_grad()
                out = self.model(self.graph.x[n_id], adjs)
                
                loss = self.loss(out, self.graph.y[n_id[:batch_size]])
                loss.backward()
                self.optimizer.step()

                train_loss += float(loss)
                
            # Validate
            self.model.eval()
            print("[model] Validating...")
            for batch_size, n_id, adjs in self.val_sampler:
                print("Batch!")
                # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
                adjs = [adj.to(self.device) for adj in adjs]
                out = self.model(self.graph.x[n_id], adjs)
                
                loss = self.loss(out, self.graph.y[n_id[:batch_size]])
                val_loss += float(loss)

        self.epoch += 1
        return train_loss, val_loss

    def test(self):
        # This could be done in the train step as well since it does not require much extra calculation

        self.model.eval()

        if self.training_style == "transductive":
            preds = self.model(self.graph)

            test_nodes_preds = preds[self.test_idx].max(dim=1).indices
            test_nodes_true = self.graph.y[self.test_idx]

            correct = test_nodes_preds.eq(test_nodes_true).sum().item()

            return float(correct / len(test_nodes_true))

        elif self.training_style == "inductive":
            correct = 0
            total = 0
            for data in self.test_loader:
                data = data.to(self.device)

                output = self.model(data)

                preds = output.max(dim=1).indices
                
                correct += preds.eq(data.y).sum().item()
                total += data.num_graphs

            return correct / total

        elif self.training_style == "graphsage":
            # to use the entire graph we work layer by layer on everything at once since that's faster
            # print("Inferencing..")
            # preds = self.model.inference(self.graph.x, self.entire_graph_sampler, self.device)

            # test_nodes_preds = preds[self.test_idx].max(dim=1).indices.cpu()
            # test_nodes_true = self.graph.y[self.test_idx].cpu()

            # correct = test_nodes_preds.eq(test_nodes_true).sum().item()

            # return float(correct / len(test_nodes_true))
            print("[model] Testing...")
            total_correct = 0
            total = 0
            for batch_size, n_id, adjs in self.test_sampler:
                print("Batch!")
                # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
                adjs = [adj.to(self.device) for adj in adjs]
                out = self.model(self.graph.x[n_id], adjs)
                
                total_correct += int(out.argmax(dim=-1).eq(self.graph.y[n_id[:batch_size]]).sum())
                total += batch_size

            return total_correct / total


    def save_model(self, path):
        pass

    def load_model(self, path):
        pass