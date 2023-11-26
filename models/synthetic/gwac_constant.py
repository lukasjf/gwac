import torch
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool

from collections import defaultdict

class ConstantDelayGwAC(torch.nn.Module):
    def __init__(self, in_features, hidden, out_features, message_size, graph_class, first_message, aggr):
        super(ConstantDelayGwAC, self).__init__()
        self.first_message = first_message
        self.encoder = torch.nn.Linear(in_features, hidden)
        self.newstate = torch.nn.Linear(hidden + message_size, hidden)
        self.new_message = torch.nn.Linear(hidden + message_size, message_size)
        self.decoder = torch.nn.Linear(hidden, out_features)
        self.graph_class = graph_class

        self.pooling = global_add_pool if aggr == "add" else global_mean_pool if aggr == "mean" else global_max_pool

    def neighbors(self, data):
        nbs = defaultdict(set)
        source, dest = data.edge_index
        for node in range(data.num_nodes):
            for i in range(len(source)):
                if int(source[i]) == node:
                    nbs[node].add(int(dest[i]))
                if int(dest[i]) == node:
                    nbs[node].add(int(source[i]))
        return nbs

    def forward(self, data):
        pred = []
        for graph in data.to_data_list():
            neighbors = self.neighbors(graph)
            finalstates = []
            for i in range(graph.num_nodes):
                encoded_nodes = self.encoder(graph.x)
                predictions = encoded_nodes
                queue = []
                queue.append((i, self.first_message))
                messages = 0
                while queue and messages < graph.num_nodes * 5:
                    messages += 1
                    node, message = queue.pop(0)
                    features = predictions[node:node + 1, :]
                    newstate = torch.relu(self.newstate(torch.cat([features, message], dim=1)))
                    newmessage = self.new_message(torch.cat([newstate, message], dim=1))
                    for nb in neighbors[node]:
                        queue.append((nb, newmessage))
                    predictions[node:node + 1, :] = newstate
                if not self.graph_class:
                    pred.append(torch.log_softmax(self.decoder(predictions[i:i+1, :]), dim=-1))
                else:
                    finalstates.append(predictions[i:i+1, :])
            if self.graph_class:
                predictions = self.pooling(torch.cat(finalstates, dim=0), batch=torch.zeros(graph.num_nodes).long())
                pred.append(torch.log_softmax(self.decoder(predictions), dim=-1))
        return torch.cat(pred, dim=0), 0

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        self.newstate.reset_parameters()
        self.new_message.reset_parameters()