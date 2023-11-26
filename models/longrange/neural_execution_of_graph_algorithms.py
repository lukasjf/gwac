import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool
from torch_geometric.nn.pool import avg_pool
import torch_scatter
from collections import defaultdict

class NEG(nn.Module):
    def __init__(self, num_features, num_hidden, num_classes, num_predictions, graph_class):
        super(NEG, self).__init__()

        dim = num_hidden
        self.graph_class = graph_class
        self.num_predictions = num_predictions

        self.encoder = nn.Linear(num_features, num_hidden)
        self.decoders = torch.nn.ModuleList([torch.nn.Linear(num_hidden, num_classes) for _ in range(num_predictions)])
        self.process = GINConv(nn=nn.Sequential(nn.Linear(dim, dim)), **{"aggr": "max"})
        self.termination1 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.termination2 = nn.Linear(dim, 1)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.process.reset_parameters()
        for decoder in self.decoders:
            decoder.reset_parameters()

    def forward(self, data):
        preds = defaultdict(list)
        for graph in data.to_data_list():
            auxloss = []
            x = graph.x
            edge_index = graph.edge_index
            diameter = graph.diameter

            h = self.encoder(x)
            for n in range(min(diameter*2, graph.num_nodes)):
                h = self.process(h, edge_index)
                h_mean = torch_scatter.scatter(self.termination1(h), index=torch.zeros(graph.num_nodes).long(), dim=0,
                                               reduce="mean")
                termination = torch.sigmoid(self.termination2(h_mean))
                #print(termination, 0. if n+1 < diameter else 1.)
                termination_loss = nn.BCELoss()(termination.squeeze(0), torch.tensor([0. if n+1 < diameter else 1.]))
                auxloss.append(termination_loss)
                if termination > 0.9:
                    break
            if self.graph_class:
                h = global_add_pool(h)
            for p in range(self.num_predictions):
                preds[p].append(torch.log_softmax(self.decoders[p](h), dim=-1))
        return [torch.cat(preds[p], dim=0) for p in preds], torch.stack(auxloss).mean()
        #torch.cat(preds, dim=0), torch.mean(torch.stack(auxloss))