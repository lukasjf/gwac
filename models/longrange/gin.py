import torch
from torch import nn
from torch_geometric.nn import GINConv, global_add_pool
from collections import defaultdict

class GIN(nn.Module):
    def __init__(self, num_features, num_hidden, num_classes, num_predictions, graph_class):
        super(GIN, self).__init__()

        dim = num_hidden
        self.graph_class = graph_class
        self.num_predictions = num_predictions

        self.encoder = nn.Linear(num_features, num_hidden)
        self.decoders = torch.nn.ModuleList([torch.nn.Linear(num_hidden, num_classes) for _ in range(num_predictions)])
        self.process = GINConv(nn=nn.Sequential(nn.Linear(dim, dim)), **{"aggr": "max"})

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.process.reset_parameters()
        for decoder in self.decoders:
            decoder.reset_parameters()

    def forward(self, data):
        preds = defaultdict(list)
        for graph in data.to_data_list():
            x = graph.x
            edge_index = graph.edge_index
            diameter = graph.diameter

            h = self.encoder(x)
            for n in range(diameter):
                h = self.process(h, edge_index)
            if self.graph_class:
                h = global_add_pool(h)
            for p in range(self.num_predictions):
                preds[p].append(torch.log_softmax(self.decoders[p](h), dim=-1))
        return [torch.cat(preds[p], dim=0) for p in preds], 0
