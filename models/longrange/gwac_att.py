import torch
import math

from collections import defaultdict


class GwACAttention(torch.nn.Module):
    def __init__(self, in_features, hidden, out_features, num_predictions, message_size, graph_class):
        super(GwACAttention, self).__init__()
        self.message_size = message_size
        self.hidden = hidden
        self.encoder = torch.nn.Linear(in_features, hidden)
        self.attention_heads = 4
        self.queries = [torch.nn.Linear(message_size, hidden) for _ in range(self.attention_heads)]
        self.keys = [torch.nn.Linear(hidden, hidden) for _ in range(self.attention_heads)]
        self.multihead_readout = torch.nn.Linear(self.attention_heads * hidden, hidden)
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

            predictions = defaultdict(list)
            for i in range(graph.num_nodes):
                predictions[i].append(encoded[i:i+1, :])

            queue = []
            for i, start in enumerate(graph.starts):
                if start:
                    queue.append((i, data.first_message[i:i+1, :]))
            messages = 0
            while queue and messages < graph.num_nodes * 10:
                messages += 1
                node, message = queue.pop(0)

                features = torch.cat([predictions[node][-i-1] for i in range(0, min(10, len(predictions[node])))], dim=0)
                heads = []
                for i in range(self.attention_heads):
                    query = self.queries[0](message)
                    keys = self.keys[0](features)
                    logits = torch.softmax(keys @ query.T / math.sqrt(self.hidden), dim=-0)
                    values = torch.sum(logits * features, dim=0, keepdim=True)
                    heads.append(values)

                newstate = self.multihead_readout(torch.cat(heads, dim=1))
                newmessage = self.new_message(torch.cat([newstate, message], dim=1))
                for nb in neighbors[node]:
                    queue.append((nb, newmessage))
                predictions[node].append(newstate)

            final_features = torch.cat([predictions[i][-1] for i in range(graph.num_nodes)], dim=0)
            if self.graph_class:
                final_features = torch.sum(final_features, dim=0, keepdim=True)
            for p in range(self.num_predictions):
                pred[p].append(torch.log_softmax(self.decoders[p](final_features), dim=-1))
        return [torch.cat(pred[p], dim=0) for p in pred], 0

    def reset_parameters(self):
        self.encoder.reset_parameters()
        for decoder in self.decoders:
            decoder.reset_parameters()
        self.new_message.reset_parameters()