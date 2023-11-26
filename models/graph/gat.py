import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool

class GAT(nn.Module):
    def __init__(self, num_features, num_hidden, num_classes, dropout):
        super(GAT, self).__init__()

        num_features = num_features
        dim = num_hidden
        self.dropout = dropout

        self.num_layers = 4

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()

        self.convs.append(GATConv(num_features, dim))
        self.bns.append(nn.BatchNorm1d(dim))
        self.fcs.append(nn.Linear(num_features, num_classes))
        self.fcs.append(nn.Linear(dim, num_classes))

        for i in range(self.num_layers - 1):
            self.convs.append(GATConv(dim, dim))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(dim, num_classes))

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, GATConv):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        outs = [x]
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            outs.append(x)

        out = None
        for i, x in enumerate(outs):
            x = global_add_pool(x, batch)
            x = F.dropout(self.fcs[i](x), p=self.dropout, training=self.training)
            if out is None:
                out = x
            else:
                out += x
        return F.log_softmax(out, dim=-1), 0