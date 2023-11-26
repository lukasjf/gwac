import torch.nn  as nn
import torch

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GINConv
import torch.nn.functional as F

class DropGIN(nn.Module):
    def __init__(self, num_features, num_hidden, num_classes, num_layers, p, runs, use_aux_loss, graph_class, aggr):
        super(DropGIN, self).__init__()

        dim = num_hidden

        self.num_layers = num_layers
        self.aggr = aggr
        self.pooling = global_add_pool if aggr == "add" else global_mean_pool if aggr == "mean" else global_max_pool
        self.graph_class = graph_class
        self.p = p
        self.runs = runs

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()

        self.convs.append(
            GINConv(nn.Sequential(nn.Linear(num_features, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim)),
                 **{"aggr": aggr}))
        self.bns.append(nn.BatchNorm1d(dim))
        self.fcs.append(nn.Linear(num_features, num_classes))
        self.fcs.append(nn.Linear(dim, num_classes))

        for i in range(self.num_layers - 1):
            self.convs.append(
                GINConv(nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim)),
                     **{"aggr": aggr}))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(dim, num_classes))

        self.use_aux_loss = use_aux_loss
        if use_aux_loss:
            self.aux_fcs = nn.ModuleList()
            self.aux_fcs.append(nn.Linear(num_features, num_classes))
            for i in range(self.num_layers):
                self.aux_fcs.append(nn.Linear(dim, num_classes))

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, GINConv):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        # Do runs in parallel, by repeating the graphs in the batch
        x = x.unsqueeze(0).expand(self.runs, -1, -1).clone()
        drop = torch.bernoulli(torch.ones([x.size(0), x.size(1)], device=x.device) * self.p).bool()
        x[drop] = 0.0
        del drop
        outs = [x]
        x = x.view(-1, x.size(-1))
        run_edge_index = edge_index.repeat(1, self.runs) + torch.arange(self.runs,
                                                                       device=edge_index.device).repeat_interleave(
            edge_index.size(1)) * (edge_index.max() + 1)
        for i in range(self.num_layers):
            x = self.convs[i](x, run_edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            outs.append(x.view(self.runs, -1, x.size(-1)))
        del run_edge_index

        out = None
        for i, x in enumerate(outs):
            x = x.mean(dim=0)
            if self.graph_class:
                x = self.pooling(x, batch)
            x = self.fcs[i](x)  # No dropout layer in these experiments
            if out is None:
                out = x
            else:
                out += x

        if self.use_aux_loss:
            aux_out = torch.zeros(self.runs, out.size(0), out.size(1), device=out.device)
            run_batch = batch.repeat(self.runs) + torch.arange(self.runs, device=edge_index.device).repeat_interleave(
                batch.size(0)) * (batch.max() + 1)
            for i, x in enumerate(outs):
                if self.graph_class:
                    x = x.view(-1, x.size(-1))
                    x = self.pooling(x, run_batch)
                x = x.view(self.runs, -1, x.size(-1))
                x = self.aux_fcs[i](x)  # No dropout layer in these experiments
                aux_out += x

            return F.log_softmax(out, dim=-1), F.log_softmax(aux_out, dim=-1)
        else:
            return F.log_softmax(out, dim=-1), 0