import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch_geometric.transforms.to_dense as Dense
from torch_geometric.utils import to_dense_adj


class UnitMLP(nn.Module):
    def __init__(self, in_feat: int, out_feat: int, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(in_feat, out_feat, (1, 1)))
        for i in range(1, num_layers):
            self.layers.append(nn.Conv2d(out_feat, out_feat, (1, 1)))

    def forward(self, x: Tensor):
        """ x: batch x N x N x channels"""
        # Convert for conv2d
        x = x.permute(0, 3, 1, 2).contiguous()                   # channels, N, N
        for layer in self.layers[:-1]:
            x = F.relu(layer.forward(x))
        x = self.layers[-1].forward(x)
        x = x.permute(0, 2, 3, 1)           # batch_size, N, N, channels
        return x

    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()


class PowerfulLayer(nn.Module):
    def __init__(self, in_feat: int, out_feat: int, num_layers: int):
        super().__init__()
        a = in_feat
        b = out_feat
        self.m1 = UnitMLP(a, b, num_layers)
        self.m2 = UnitMLP(a, b, num_layers)
        self.m4 = nn.Linear(a + b, b, bias=True)

    def forward(self, x):
        """ x: batch x N x N x in_feat"""
        out1 = self.m1.forward(x).permute(0, 3, 1, 2)                 # batch, out_feat, N, N
        out2 = self.m2.forward(x).permute(0, 3, 1, 2)                 # batch, out_feat, N, N
        out3 = x
        mult = out1 @ out2                                         # batch, out_feat, N, N
        out = torch.cat((mult.permute(0, 2, 3, 1), out3), dim=3)      # batch, N, N, out_feat
        suffix = self.m4.forward(out)
        return suffix

    def reset_parameters(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()
        self.m4.reset_parameters()


class FeatureExtractor(nn.Module):
    def __init__(self, in_features: int, out_features: int, graph_class: bool, aggr:str):
        super().__init__()
        self.lin1 = nn.Linear(in_features, out_features, bias=True)
        self.lin2 = nn.Linear(in_features, out_features, bias=False)
        self.lin3 = torch.nn.Linear(out_features, out_features, bias=False)
        self.graph_class = graph_class
        self.aggr = aggr

    def forward(self, u):
        """ u: (batch_size, num_nodes, num_nodes, in_features)
            output: (batch_size, out_features). """
        n = u.shape[1]
        if self.graph_class:
            diag = u.diagonal(dim1=1, dim2=2)       # batch_size, in_features, num_nodes
            trace = self.aggregate(diag, dim=2)          # batch_size, in_features
            out1 = self.lin1.forward(trace / n)     # batch_size, out_features
            s = self.aggregate(u, dim=1)
            s = self.aggregate(s, dim=1)
            s = (s - trace) / (n * (n-1))    # batch_size, in_features
            out2 = self.lin2.forward(s)                             # bs, out_feat
            out = out1 + out2
            out = out + self.lin3.forward(F.relu(out))
            return out
        else:
            u = u.squeeze(0)
            diag = u.diagonal(dim1=0, dim2=1)
            out1 = self.lin1.forward(diag.permute(1, 0))
            s = self.aggregate(u, dim=1) / n
            out2 = self.lin2.forward(s)                             # bs, out_feat
            out = out1 + out2
            out = out + self.lin3.forward(F.relu(out))
            return out

    def aggregate(self, tensor, dim):
        if self.aggr == "add":
            return torch.sum(tensor, dim=dim)
        elif self.aggr == "mean":
            return torch.mean(tensor, dim=dim)
        elif self.aggr == "max":
            return torch.max(tensor, dim=dim)[0]
        else:
            1/0


    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()


class Powerful(nn.Module):
    def __init__(self, num_classes: int, num_layers: int, num_features:int, hidden: int, hidden_final: int,
                 dropout_prob: float, simplified: bool, graph_class:bool, aggr:str):
        super().__init__()
        layers_per_conv = 1
        self.num_features = 2 * num_features
        self.graph_class = graph_class
        self.layer_after_conv = not simplified
        self.dropout_prob = dropout_prob
        self.no_prop = FeatureExtractor(self.num_features, hidden_final, self.graph_class, aggr,)
        initial_conv = PowerfulLayer(self.num_features, hidden, layers_per_conv)
        self.convs = nn.ModuleList([initial_conv])
        self.bns = nn.ModuleList([])
        for i in range(1, num_layers):
            self.convs.append(PowerfulLayer(hidden, hidden, layers_per_conv))

        self.feature_extractors = torch.nn.ModuleList([])
        for i in range(num_layers):
            self.bns.append(nn.BatchNorm2d(hidden))
            self.feature_extractors.append(FeatureExtractor(hidden, hidden_final, self.graph_class, aggr))
        if self.layer_after_conv:
            self.after_conv = nn.Linear(hidden_final, hidden_final)
        self.final_lin = nn.Linear(hidden_final, num_classes)

    def forward(self, data):
        preds = []
        for single_graph in data.to_data_list():
            source, dest = single_graph.edge_index[0], single_graph.edge_index[1]
            edge_attr = torch.zeros(len(source), self.num_features)
            for i in range(len(source)):
                edge_attr[i] = torch.cat([single_graph.x[source[i]], single_graph.x[dest[i]]], dim=0)
            single_graph.A = to_dense_adj(single_graph.edge_index, edge_attr=edge_attr)
            u = single_graph.A          # batch, N, N, 1
            out = self.no_prop.forward(u)
            for conv, extractor, bn in zip(self.convs, self.feature_extractors, self.bns):
                u = conv(u)
                u = bn(u.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                out = out + extractor.forward(u)
            out = F.relu(out) / len(self.convs)
            if self.layer_after_conv:
                out = out + F.relu(self.after_conv(out))
            out = F.dropout(out, p=self.dropout_prob, training=self.training)
            out = self.final_lin(out)
            preds.append(out)
        return F.log_softmax(torch.cat(preds), dim=-1), 0

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for fe in self.feature_extractors:
            fe.reset_parameters()
        self.final_lin.reset_parameters()