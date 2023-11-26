import torch

from collections import defaultdict


# version that uses a single gate, and regularizes with NLL
class GwACIter(torch.nn.Module):
    def __init__(self, in_features, hidden, out_features, num_predictions, message_size, graph_class):
        super(GwACIter, self).__init__()
        self.message_size = message_size
        self.encoder = torch.nn.Linear(in_features, hidden)
        self.newstate = torch.nn.Linear(hidden + message_size, hidden)
        self.new_message = torch.nn.Linear(hidden + message_size, message_size)
        self.graph_class = graph_class
        self.num_predictions = num_predictions
        self.decoders = torch.nn.ModuleList([torch.nn.Linear(hidden, out_features) for _ in range(num_predictions)])

        self.confidence = torch.nn.Linear(hidden+message_size, 1)

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

    def forward(self, data, test=False):
        pred = defaultdict(list)
        for graph in data.to_data_list():
            neighbors = self.neighbors(graph)
            encoded_nodes = self.encoder(graph.xa)

            predictions = defaultdict(list)
            remaining_confidence = defaultdict(list)
            final_features = defaultdict(list)

            for i in range(graph.num_nodes):
                predictions[i].append(encoded_nodes[i:i+1, :])
                remaining_confidence[i].append(torch.ones(1, 1))
                final_features[i].append(torch.zeros_like(predictions[i][-1]))

            queue = []
            for i, start in enumerate(graph.starts):
                if start:
                    queue.append((i, graph.first_message[i:i+1, :]))
            messages = 0

            while queue and messages < graph.num_nodes * 10:
                node, message = queue.pop(0)
                features = predictions[node][-1]

                if remaining_confidence[node][-1] < 1e-7:
                    continue
                messages += 1

                step_confidence = torch.sigmoid(self.confidence(torch.cat([features, message], dim=1)))
                newstate = torch.relu(self.newstate(torch.cat([features, message], dim=1)))
                newmessage = self.new_message(torch.cat([newstate, message], dim=1))
                for nb in neighbors[node]:
                    queue.append((nb, newmessage))
                predictions[node].append(newstate)

                final_features[node].append(final_features[node][-1] + remaining_confidence[node][-1] *
                                            step_confidence * newstate)
                remaining_confidence[node].append(remaining_confidence[node][-1] * (1-step_confidence))

            final_features = torch.cat([final_features[i][-1] for i in range(graph.num_nodes)], dim=0)
            if self.graph_class:
                final_features = torch.sum(final_features, dim=0, keepdim=True)
            readouts = []
            for p in range(self.num_predictions):
                pred[p].append(torch.log_softmax(self.decoders[p](final_features),dim=-1))
        return [torch.cat(pred[p], dim=0) for p in pred], 0

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.newstate.reset_parameters()
        self.new_message.reset_parameters()
        for decoder in self.decoders:
            decoder.reset_parameters()