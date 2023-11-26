import torch
from torch.nn import LSTMCell

from collections import defaultdict


class GwACLSTM(torch.nn.Module):
    def __init__(self, in_features, hidden, out_features, num_predictions, message_size, graph_class):
        super(GwACLSTM, self).__init__()
        self.message_size = message_size
        self.encoder = torch.nn.Linear(in_features, hidden)
        self.newstate = LSTMCell(message_size, hidden)
        self.new_message = torch.nn.Linear(hidden + message_size, message_size)
        self.decoders = torch.nn.ModuleList([torch.nn.Linear(hidden, out_features) for _ in range(num_predictions)])
        self.graph_class = graph_class
        self.num_predictions = num_predictions

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
        pred = defaultdict(list)
        for graph in data.to_data_list():
            neighbors = self.neighbors(graph)
            encoded_nodes = self.encoder(graph.xa)
            encoded = encoded_nodes

            hidden_states = defaultdict(list)
            cells = defaultdict(list)
            for i in range(graph.num_nodes):
                hidden_states[i].append(encoded[i:i+1, :])
                cells[i].append(torch.zeros_like(hidden_states[i][-1]))

            queue = []
            for i, start in enumerate(graph.starts):
                if start:
                    queue.append((i, data.first_message[i:i+1, :]))
            messages = 0
            while queue and messages < graph.num_nodes * 10:
                messages += 1
                node, message = queue.pop(0)
                features = hidden_states[node][-1]

                new_hidden, new_cell = self.newstate(message, (features, cells[node][-1]))
                newmessage = self.new_message(torch.cat([new_hidden, message], dim=1))
                for nb in neighbors[node]:
                    queue.append((nb, newmessage))
                hidden_states[node].append(new_hidden)
                cells[node].append(new_cell)

            final_features = torch.cat([hidden_states[i][-1] for i in range(graph.num_nodes)], dim=0)
            if self.graph_class:
                final_features = torch.sum(final_features, dim=0, keepdim=True)
            for p in range(self.num_predictions):
                pred[p].append(torch.log_softmax(self.decoders[p](final_features), dim=-1))
        return [torch.cat(pred[p], dim=0) for p in pred], 0

    def reset_parameters(self):
        self.encoder.reset_parameters()
        for decoder in self.decoders:
            decoder.reset_parameters()
        self.newstate.reset_parameters()
        self.new_message.reset_parameters()