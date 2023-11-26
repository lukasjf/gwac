import torch
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
import numpy as np
import torch.multiprocessing as mp
from os import getpid
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm
from collections import defaultdict


class GwACNode(torch.nn.Module):
    def __init__(self, in_features, hidden, out_features, message_size, graph_class, num_starts, num_messages,
                 use_skip):
        super(GwACNode, self).__init__()

        self.encoder = torch.nn.Linear(in_features, hidden)
        self.newstate = torch.nn.Linear(hidden + message_size, hidden)
        self.new_message = torch.nn.Linear(hidden + message_size, message_size)
        self.decoder = torch.nn.Linear(hidden, out_features)
        self.skipdecoder = torch.nn.Linear(4*hidden, out_features)

        self.message_size = message_size
        self.graph_class = graph_class
        self.num_starts = num_starts
        self.num_messages = num_messages
        self.hidden = hidden
        self.use_skip = use_skip

    def forward(self, data, p_starts=None, nbs=None):
        if p_starts is not None:
            starts = []
            for i, is_start in enumerate(p_starts):
                if is_start:
                    starts.append(i)
        else:
            starts = None
        pred = []
        if self.graph_class:
            for graph in data.to_data_list():
                pred.append(self.run_on_graph(graph, None, starts, nbs))
        else:
            pred.append(self.run_on_graph(data, None, starts, nbs))
        return torch.cat(pred, dim=0), 0

    def run_on_graph(self, graph, opt, p_starts=None, nbs=None):
        neighbors = nbs
        nodestates = torch.zeros(graph.num_nodes, self.hidden)
        skipnodestates = torch.zeros(graph.num_nodes, 4 * self.hidden)

        starts = range(graph.num_nodes) if p_starts == None else p_starts
        for i in starts:
            encoded_nodes = self.encoder(graph.x)
            predictions = defaultdict(list)
            for node in range(graph.num_nodes):
                node_encoded = encoded_nodes[node:node+1, :]
                predictions[node].append(node_encoded)
            queue = []
            queue.append((i, torch.ones(1, self.message_size)))
            messages = 0
            while queue and messages < self.num_messages:
                messages += 1
                node, message = queue.pop(0)
                features = predictions[node][-1]
                newstate = torch.relu(self.newstate(torch.cat([features, message], dim=1)))
                newmessage = self.new_message(torch.cat([newstate, message], dim=1))
                for nb in neighbors[node]:
                    queue.append((nb, newmessage))
                predictions[node].append(newstate)
            nodestates[i, :] = predictions[i][-1]
            skipstates = predictions[i][-4:]
            while len(skipstates) < 4:
                skipstates = [torch.zeros(1, self.hidden)] + skipstates
            skipnodestates[i:i+1, :] = torch.cat(skipstates, dim=1)
        if self.graph_class:
            if self.use_skip:
                predictions = global_add_pool(skipnodestates, batch=torch.zeros(graph.num_nodes).long())
                logits = torch.log_softmax(self.skipdecoder(predictions), dim=-1)
            else:
                predictions = global_add_pool(nodestates, batch=torch.zeros(graph.num_nodes).long())
                logits = torch.log_softmax(self.decoder(predictions), dim=-1)
            return logits
        else:
            logits = torch.log_softmax(self.decoder(nodestates), dim=-1)
            return logits

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        self.newstate.reset_parameters()
        self.new_message.reset_parameters()
        self.skipdecoder.reset_parameters()