import torch
import torch.nn.functional as F
import torch_scatter
from torch_geometric.nn import GraphConv
import torch.nn as nn

from models.synthetic.esan.esan_data import EdgeDeleted, NodeDeleted, EgoNets


def subgraph_pool(h_node, batched_data, pool):
    # Represent each subgraph as the pool of its node representations
    num_subgraphs = batched_data.num_subgraphs
    tmp = torch.cat([torch.zeros(1, device=num_subgraphs.device, dtype=num_subgraphs.dtype),
                     torch.cumsum(num_subgraphs, dim=0)])
    graph_offset = tmp[batched_data.batch]

    subgraph_idx = graph_offset + batched_data.subgraph_batch

    return pool(h_node, subgraph_idx)


class DSnetwork(torch.nn.Module):
    def __init__(self, num_layers, in_dim, emb_dim, num_tasks, num_nodes, graph_class, permute, aggr):
        super(DSnetwork, self).__init__()

        self.emb_dim = emb_dim
        self.num_nodes = num_nodes
        self.graph_class = graph_class
        self.aggr = aggr
        self.aggr_to_reduce = "mean" if aggr == "add" else "mean" if aggr == "mean" else "max"
        self.permute = permute

        gnn_list = []
        bn_list = []
        readout = []
        for i in range(num_layers):
            dim = emb_dim
            gnn_list.append(GraphConv(emb_dim if i != 0 else in_dim, emb_dim, aggr=aggr))
            bn_list.append(torch.nn.BatchNorm1d(emb_dim))
            readout.append(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, num_tasks)))

        self.gnn_list = torch.nn.ModuleList(gnn_list)

        self.bn_list = torch.nn.ModuleList(bn_list)

        self.readout_list = torch.nn.ModuleList(readout)

        self.final_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=emb_dim, out_features=2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2 * emb_dim, out_features=num_tasks)
        )

    def forward(self, input):
        predictions = []
        for graph in input.to_data_list():
            perm = NodeDeleted() if self.permute == "node" else EdgeDeleted() if self.permute == "edge" else EgoNets(2)
            subgraph = perm(graph)
            x, edge_index, batch = subgraph.x, subgraph.edge_index, torch.zeros(subgraph.num_subgraphs).long()

            out = None
            for i in range(len(self.gnn_list)):
                gnn, bn = self.gnn_list[i], self.bn_list[i]

                h1 = bn(gnn(x, edge_index))
                x = F.relu(h1)

            if self.graph_class:
                h_subgraph = torch_scatter.scatter(x, index=subgraph.subgraph_batch, dim=0, reduce=self.aggr_to_reduce)
                # aggregate to obtain a representation of the graph given the representations of the subgraphs
                h_graph = torch_scatter.scatter(src=h_subgraph, index=batch, dim=0, reduce=self.aggr_to_reduce)
                predictions.append(torch.log_softmax(self.final_layers(h_graph), dim=-1))
            else:
                x_nodes = torch_scatter.scatter(x, index=subgraph.subgraph_node_idx, dim=0, reduce=self.aggr_to_reduce)
                predictions.append(torch.log_softmax(self.final_layers(x_nodes), dim=-1))
        return torch.cat(predictions, dim=0), 0

    def reset_parameters(self):
        for l in self.gnn_list:
            l.reset_parameters()
        for bn in self.bn_list:
            bn.reset_parameters()
        for r in self.readout_list:
            r[0].reset_parameters()
            r[2].reset_parameters()
        self.final_layers[0].reset_parameters()
        self.final_layers[2].reset_parameters()


class DSSnetwork(torch.nn.Module):
    def __init__(self, num_layers, in_dim, emb_dim, num_tasks, num_nodes, graph_class, permute, aggr):
        super(DSSnetwork, self).__init__()

        self.emb_dim = emb_dim
        self.num_nodes = num_nodes
        self.graph_class = graph_class
        self.aggr = aggr
        self.aggr_to_reduce = "mean" if aggr == "add" else "mean" if aggr == "mean" else "max"
        self.permute = permute

        gnn_list = []
        gnn_sum_list = []
        bn_list = []
        bn_sum_list = []
        for i in range(num_layers):
            gnn_list.append(GraphConv(emb_dim if i != 0 else in_dim, emb_dim, aggr=aggr))
            bn_list.append(torch.nn.BatchNorm1d(emb_dim))

            gnn_sum_list.append(GraphConv(emb_dim if i != 0 else in_dim, emb_dim, aggr=aggr))
            bn_sum_list.append(torch.nn.BatchNorm1d(emb_dim))

        self.gnn_list = torch.nn.ModuleList(gnn_list)
        self.gnn_sum_list = torch.nn.ModuleList(gnn_sum_list)

        self.bn_list = torch.nn.ModuleList(bn_list)
        self.bn_sum_list = torch.nn.ModuleList(bn_sum_list)

        self.final_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=emb_dim, out_features=2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2 * emb_dim, out_features=num_tasks)
        )

    def forward(self, input):
        predictions = []
        for graph in input.to_data_list():
            perm = NodeDeleted() if self.permute == "node" else EdgeDeleted() if self.permute == "edge" else EgoNets(2)
            subgraph = perm(graph)

            x, edge_index, batch = subgraph.x, subgraph.edge_index, torch.zeros(subgraph.num_subgraphs).long()

            for i in range(len(self.gnn_list)):
                gnn, bn, gnn_sum, bn_sum = self.gnn_list[i], self.bn_list[i], self.gnn_sum_list[i], self.bn_sum_list[i]

                h1 = bn(gnn(x, edge_index))

                x_sum = torch_scatter.scatter(src=x, index=subgraph.subgraph_node_idx, dim=0,
                                              reduce=self.aggr_to_reduce)
                h2 = bn_sum(gnn_sum(x_sum, subgraph.original_edge_index))
                h2_list = [h2 for _ in range(subgraph.num_subgraphs)]
                x = F.relu(h1 + torch.cat(h2_list, dim=0))

            if self.graph_class:
                h_subgraph = torch_scatter.scatter(x, index=subgraph.subgraph_batch, dim=0, reduce=self.aggr_to_reduce)
                # aggregate to obtain a representation of the graph given the representations of the subgraphs
                h_graph = torch_scatter.scatter(src=h_subgraph, index=batch, dim=0, reduce=self.aggr_to_reduce)
                predictions.append(torch.log_softmax(self.final_layers(h_graph), dim=-1))
            else:
                x_nodes = torch_scatter.scatter(x, index=subgraph.subgraph_node_idx, dim=0, reduce=self.aggr_to_reduce)
                predictions.append(self.final_layers(torch.log_softmax(x_nodes, dim=-1)))
        return torch.cat(predictions, dim=0), 0

    def reset_parameters(self):
        for l in self.gnn_list:
            l.reset_parameters()
        for bn in self.bn_list:
            bn.reset_parameters()
        for l in self.gnn_sum_list:
            l.reset_parameters()
        for bn in self.bn_sum_list:
            bn.reset_parameters()
        self.final_layers[0].reset_parameters()
        self.final_layers[2].reset_parameters()


class EgoEncoder(torch.nn.Module):
    def __init__(self, encoder):
        super(EgoEncoder, self).__init__()
        self.num_added = 2
        self.enc = encoder

    def forward(self, x):
        return torch.hstack((x[:, :self.num_added], self.enc(x[:, self.num_added:])))


class ZincAtomEncoder(torch.nn.Module):
    def __init__(self, policy, emb_dim):
        super(ZincAtomEncoder, self).__init__()
        self.policy = policy
        self.num_added = 2
        self.enc = torch.nn.Embedding(21, emb_dim)

    def forward(self, x):
        if self.policy == 'ego_nets_plus':
            return torch.hstack((x[:, :self.num_added], self.enc(x[:, self.num_added:].squeeze())))
        else:
            return self.enc(x.squeeze())