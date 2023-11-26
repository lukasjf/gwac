import torch.nn  as nn
import torch

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import torch.nn.functional as F

class GIN(nn.Module):
    def __init__(self, num_features, num_hidden, num_classes, num_layers, graph_class, conv, augmentation, aggr):
        super(GIN, self).__init__()

        dim = num_hidden

        self.graph_class = graph_class
        self.augmentation = augmentation
        self.conv = conv

        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()

        self.convs.append(
            conv(nn.Sequential(nn.Linear(num_features, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim)),
                 **{"aggr": aggr}))
        self.bns.append(nn.BatchNorm1d(dim))
        self.fcs.append(nn.Linear(num_features, num_classes))
        self.fcs.append(nn.Linear(dim, num_classes))

        for i in range(self.num_layers - 1):
            self.convs.append(
                conv(nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim)),
                     **{"aggr": aggr}))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(dim, num_classes))
        self.pooling = global_add_pool if aggr == "add" else global_mean_pool if aggr == "mean" else global_max_pool
        print(self.pooling)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, self.conv):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        if self.augmentation == 'ids':
            x = torch.cat([x, data.id.float()], dim=1)
        elif self.augmentation == 'random':
            x = torch.cat([x, torch.randint(0, 100, (x.size(0), 1), device=x.device) / 100.0], dim=1)

        outs = [x]
        for i in range(self.num_layers):
            if self.augmentation == 'ports':
                x = self.convs[i](x, edge_index, data.ports.expand(-1, x.size(-1)))
            else:
                x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            outs.append(x)

        out = None
        for i, x in enumerate(outs):
            if self.graph_class:
                x = self.pooling(x, batch)
            x = self.fcs[i](x)  # No dropout for these experiments
            if out is None:
                out = x
            else:
                out += x
        return F.log_softmax(out, dim=-1), 0