import torch
import torch.nn.functional as F
import torch.nn as nn

from config import config

import pickle

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

            self.graph.x = self.graph.x.to_dense() # pytorch sparse tensor do not support slicing which is needed for the sampling

            self.train_idx, self.val_idx, self.test_idx = data.get_tvt_indices()
            self.train_idx = torch.tensor(self.train_idx, dtype=torch.long).to(self.device)
            self.val_idx = torch.tensor(self.val_idx, dtype=torch.long).to(self.device)
            self.test_idx = torch.tensor(self.test_idx, dtype=torch.long).to(self.device)

            self.train_sampler, self.val_sampler, self.test_sampler, self.entire_graph_sampler = \
                data.get_tvt_samplers()
        
        else:
            raise NotImplementedError()

    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()

        # Transductive
        if self.training_style == "transductive":
            # Train
            preds = self.model(self.graph)

            # print("Received predictions:", preds, preds.shape)

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
            unsup_train_loss_pos = 0
            unsup_train_loss_neg = 0
            val_loss = 0
            unsup_val_loss_pos = 0
            unsup_val_loss_neg = 0

            unsup_test_pos = 0
            unsup_test_neg = 0

            unsup_test_pos_test = 0
            unsup_test_neg_test = 0

            # train on test
            if config["unsupervised_loss"] and config['sup_mode'] != 'sup':
                pass

                # LIMIT = 50 # len(self.test_sampler)
                # train_test_seperation = LIMIT / 2

                # print("[model] Training on TEST...")
                # for i, (batch_size, n_id, adjs) in enumerate(self.test_sampler):
                #     # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
                #     adjs = [adj.to(self.device) for adj in adjs]

                #     self.optimizer.zero_grad()
                    
                #     # for the unsupervised loss we assume the batch is concetatenated with the pos, then the neg nodes
                    
                #     out, penultimate = self.model.sampled_forward(self.graph.x[n_id], adjs)

                #     out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)
                #     pen, pos_pen, neg_pen = penultimate.split(penultimate.size(0) // 3, dim=0)

                #     # sup_loss = self.loss(out, self.graph.y[n_id[:batch_size // 3]]) * config["unsup_sup_boost"]

                #     pos_loss = -F.logsigmoid((pen * pos_pen).sum(-1)).mean()
                #     neg_loss = -F.logsigmoid(-(pen * neg_pen).sum(-1)).mean()

                #     if i < train_test_seperation:
                #         unsup_test_pos_test += pos_loss
                #         unsup_test_neg_test += neg_loss
                #     elif i > LIMIT:
                #         break
                #     else:
                #         unsup_test_pos += pos_loss
                #         unsup_test_neg += neg_loss

                #         loss = pos_loss + neg_loss

                #         loss.backward()
                #         self.optimizer.step()

            print("!!! UNSUP pos loss: %.4f, UNSUP neg loss: %.4f" % (unsup_test_pos_test, unsup_test_neg_test))

            # Train
            print("[model] Training...")
            for batch_size, n_id, adjs in self.train_sampler:
                # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
                adjs = [adj.to(self.device ) for adj in adjs]

                self.optimizer.zero_grad()
                
                if config["unsupervised_loss"]:
                    # for the unsupervised loss we assume the batch is concetatenated with the pos, then the neg nodes
                    
                    out, penultimate = self.model.sampled_forward(self.graph.x[n_id], adjs)

                    out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)
                    pen, pos_pen, neg_pen = penultimate.split(penultimate.size(0) // 3, dim=0)

                    sup_loss = self.loss(out, self.graph.y[n_id[:batch_size // 3]])

                    pos_loss = -F.logsigmoid((out * pos_out).sum(-1)).mean() / config["unsup_sup_boost"]
                    neg_loss = -F.logsigmoid(-(out * neg_out).sum(-1)).mean() / config["unsup_sup_boost"]

                    train_loss += sup_loss
                    unsup_train_loss_pos += pos_loss
                    unsup_train_loss_neg += neg_loss

                    if config['sup_mode'] == 'semi':
                        loss = sup_loss + pos_loss + neg_loss
                    elif config['sup_mode'] == 'un':
                        loss = pos_loss + neg_loss
                    elif config['sup_mode'] == 'sup':
                        loss = sup_loss
                    else:
                        raise NotImplementedError("[trainer] The sup mode %s has not been implemented" % config['sup_mode'])
                else:
                    out = self.model.sampled_forward(self.graph.x[n_id], adjs)

                    loss = self.loss(out, self.graph.y[n_id[:batch_size]])
                    train_loss += float(loss)

                loss.backward()
                self.optimizer.step()

            # Validate
            self.model.eval()
            print("[model] Validating...")
            for batch_size, n_id, adjs in self.val_sampler:
                # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
                adjs = [adj.to(self.device) for adj in adjs]
                

                if config["unsupervised_loss"]:
                    # for the unsupervised loss we assume the batch is concetatenated with the pos, then the neg nodes
                    out, penultimate = self.model.sampled_forward(self.graph.x[n_id], adjs)

                    out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)
                    pen, pos_pen, neg_pen = penultimate.split(penultimate.size(0) // 3, dim=0)

                    sup_loss = self.loss(out, self.graph.y[n_id[:batch_size // 3]])

                    pos_loss = -F.logsigmoid((pen * pos_pen).sum(-1)).mean() / config["unsup_sup_boost"]
                    neg_loss = -F.logsigmoid(-(pen * neg_pen).sum(-1)).mean() / config["unsup_sup_boost"]

                    val_loss += sup_loss
                    unsup_val_loss_pos += pos_loss
                    unsup_val_loss_neg += neg_loss
            
                else:
                    out = self.model.sampled_forward(self.graph.x[n_id], adjs)

                    loss = self.loss(out, self.graph.y[n_id[:batch_size]])
                    val_loss += float(loss)

        self.epoch += 1

        if self.epoch == 20:
            config['sup_mode'] = 'sup'

        # if config["sup_mode"] == 'sup':
        #     config["sup_mode"] = 'un'
        # elif config["sup_mode"] == 'un':
        #     config["sup_mode"] = 'semi'
        # elif config["sup_mode"] == 'semi':
        #     config["sup_mode"] = 'sup'

        if config["sampled_training"] and config["unsupervised_loss"]:
            return train_loss, val_loss, unsup_train_loss_pos, unsup_train_loss_neg, unsup_val_loss_pos, unsup_val_loss_neg, unsup_test_pos, unsup_test_neg
        
        return train_loss, val_loss

    def save_initial_reps(self):
        tosave = {"reps" : self.graph.x[self.test_idx], "y" : self.graph.y[self.test_idx]}

        with open('xs_onehot.pickle', 'wb') as handle:
            pickle.dump(tosave, handle)

    def save_sage_reps(self):
        self.model.eval()
        print("Saving reps")

        pens = []
        ys = []
        for i, (batch_size, n_id, adjs) in enumerate(self.test_sampler):
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            adjs = [adj.to(self.device) for adj in adjs]

            self.optimizer.zero_grad()
            
            # for the unsupervised loss we assume the batch is concetatenated with the pos, then the neg nodes
            
            out, penultimate = self.model.sampled_forward(self.graph.x[n_id], adjs)

            out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)
            pen, pos_pen, neg_pen = penultimate.split(penultimate.size(0) // 3, dim=0)

            y = self.graph.y[n_id[:batch_size // 3]]

            pens.append(pen)
            ys.append(y)

        tosave = {"reps" : pens, "y" : ys}

        with open('features_SEMI_LASTHOPE_epoch_%i.pickle' % self.epoch, 'wb') as handle:
            pickle.dump(tosave, handle)

    def test(self, limit=None):
        # This could be done in the train step as well since it does not require much extra calculation

        self.model.eval()

        if self.training_style == "transductive":
            preds = self.model(self.graph)

            idx = self.test_idx[:limit] if limit else self.test_idx

            test_nodes_preds = preds[idx].max(dim=1).indices
            test_nodes_true = self.graph.y[idx]

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
            # to use the entire graph we work layer by layer on everything at once since that's faster <- does not fit in memory
            # print("Inferencing..")
            # preds = self.model.inference(self.graph.x, self.entire_graph_sampler, self.device)

            # test_nodes_preds = preds[self.test_idx].max(dim=1).indices.cpu()
            # test_nodes_true = self.graph.y[self.test_idx].cpu()

            # correct = test_nodes_preds.eq(test_nodes_true).sum().item()

            # return float(correct / len(test_nodes_true))

            print("[model] Testing...")
            total_correct = 0
            total = 0
            test_loss = 0
            unsup_test_loss_pos = 0
            unsup_test_loss_neg = 0

            for batch_size, n_id, adjs in self.test_sampler:
                # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
                adjs = [adj.to(self.device) for adj in adjs]

                if config["unsupervised_loss"]:
                    out, penultimate = self.model.sampled_forward(self.graph.x[n_id], adjs)
                else:
                    out = self.model.sampled_forward(self.graph.x[n_id], adjs)

                # out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)
                if config["unsupervised_loss"]:
                    # for the unsupervised loss we assume the batch is concetatenated with the pos, then the neg nodes
                    out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)

                    sup_loss = self.loss(out, self.graph.y[n_id[:batch_size // 3]])

                    pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
                    neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()

                    test_loss += sup_loss
                    unsup_test_loss_pos += pos_loss
                    unsup_test_loss_neg += neg_loss
                    
                    total_correct += int(out.argmax(dim=-1).eq(self.graph.y[n_id[:batch_size // 3]]).sum())
                    total += batch_size // 3
                else:
                    total_correct += int(out.argmax(dim=-1).eq(self.graph.y[n_id[:batch_size]]).sum())
                    total += batch_size

            if config["unsupervised_loss"]:
                return total_correct / total, test_loss, unsup_test_loss_pos, unsup_test_loss_neg

            return total_correct / total


    def save_model(self, path):
        pass

    def load_model(self, path):
        pass