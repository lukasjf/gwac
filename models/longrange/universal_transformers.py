import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GATv2Conv, global_add_pool, global_mean_pool
from torch_geometric.nn.pool import avg_pool
import torch_scatter
from collections import defaultdict


class UniversalTransformers(nn.Module):
    def __init__(self, num_features, num_hidden, num_classes, num_predictions, graph_class):
        super(UniversalTransformers, self).__init__()

        dim = num_hidden
        self.dim = dim
        self.graph_class = graph_class
        self.encoder = nn.Linear(num_features, num_hidden)
        self.decoders = torch.nn.ModuleList([torch.nn.Linear(num_hidden, num_classes) for _ in range(num_predictions)])
        self.process1 = GATv2Conv(in_channels=dim, out_channels=dim, heads=4)
        self.process2 = nn.Sequential(nn.Linear(4 * dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.act = nn.Linear(dim, 1)
        self.num_predictions = num_predictions

    def reset_parameters(self):
        self.encoder.reset_parameters()
        for decoder in self.decoders:
            decoder.reset_parameters()
        self.process1.reset_parameters()
        self.process2[0].reset_parameters()
        self.process2[2].reset_parameters()
        self.act.reset_parameters()

    def forward(self, data):
        preds = defaultdict(list)
        for graph in data.to_data_list():
            x = graph.x
            edge_index = graph.edge_index
            hidden_states = []
            terminations = []
            total_terminated = torch.zeros(graph.num_nodes, 1)
            h = self.encoder(x)
            #print("#####################")
            for n in range(graph.num_nodes):
                h = self.process2(self.process1(h, edge_index))
                hidden_states.append(h)
                termination = torch.sigmoid(self.act(h))
                new_termination = torch.minimum(total_terminated + termination, torch.ones(graph.num_nodes, 1))
                termination_delta = torch.minimum(termination, new_termination - total_terminated)
                #print(total_terminated.squeeze(1))
                #print(termination.squeeze(1))
                #print(new_termination.squeeze(1))
                #print(termination_delta.squeeze(1))
                #print("__________________")
                total_terminated += termination_delta
                terminations.append(termination_delta)
                if all(total_terminated.squeeze(1) >= 1.0):
                    break
            final_states = torch.zeros(graph.num_nodes, self.dim)
            for i in range(len(terminations)):
                for n in range(graph.num_nodes):
                    #print(terminations[i][n,0])
                    #print(hidden_states[i][n])
                    #print(terminations[i][n,0] * hidden_states[i][n])
                    final_states[n] += terminations[i][n,0] * hidden_states[i][n]
            if self.graph_class:
                final_states = torch.sum(final_states, dim=0, keepdim=True)
            for p in range(self.num_predictions):
                preds[p].append(torch.log_softmax(self.decoders[p](final_states), dim=-1))
        return [torch.cat(preds[p], dim=0) for p in preds], 0
