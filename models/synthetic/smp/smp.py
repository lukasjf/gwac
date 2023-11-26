import torch.nn.functional as F
from models.synthetic.smp.smp_helpers import create_batch_info, GraphExtractor, EdgeCounter, NodeExtractor
from models.synthetic.smp.smp_helpers import BatchNorm, XtoX, EntryWiseX, EntrywiseU, UtoU, map_x_to_u

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class FastSMPLayer(MessagePassing):
    def __init__(self, in_features: int, num_towers: int, out_features: int, use_x: bool, aggr:str):
        super().__init__(aggr=aggr, node_dim=-2 if use_x else -3)
        self.use_x = use_x
        self.in_u, self.out_u = in_features, out_features
        if use_x:
            self.message_nn = XtoX(in_features, out_features, bias=True)
            self.linu_i = EntryWiseX(out_features, out_features)
            self.linu_j = EntryWiseX(out_features, out_features)
        else:
            self.message_nn = UtoU(in_features, out_features, n_groups=num_towers, residual=False, aggr=aggr)
            self.linu_i = EntrywiseU(out_features, out_features, num_towers=out_features)
            self.linu_j = EntrywiseU(out_features, out_features, num_towers=out_features)

    def forward(self, u, edge_index, batch_info):
        n = batch_info['num_nodes']
        u = self.message_nn(u, batch_info)
        new_u = self.propagate(edge_index, size=(n, n), u=u)
        return new_u

    def message(self, u_j):
        return u_j

    def update(self, aggr_u, u):
        a_i = self.linu_i(u)
        a_j = self.linu_j(aggr_u)
        return aggr_u + u + a_i * a_j


    def reset_parameters(self):
        self.message_nn.reset_parameters()
        self.linu_i.reset_parameters()
        self.linu_j.reset_parameters()


class SMP(torch.nn.Module):
    def __init__(self, num_input_features: int, num_classes: int, num_layers: int, hidden: int, layer_type: str,
                 hidden_final: int, dropout_prob: float, use_batch_norm: bool, use_x: bool, map_x_to_u: bool,
                 num_towers: int, simplified: bool, graph_class:bool, aggr:str):
        """ num_input_features: number of node features
            layer_type: 'SMP', 'FastSMP' or 'SimplifiedFastSMP'
            hidden_final: size of the feature map after pooling
            use_x: for ablation study, run a MPNN instead of SMP
            map_x_to_u: map the node features to the local context
            num_towers: inside each SMP layers, use towers to reduce the number of parameters
            simplified: less layers in the feature extractor.
        """
        super().__init__()
        self.map_x_to_u, self.use_x = map_x_to_u, use_x
        self.dropout_prob = dropout_prob
        self.use_batch_norm = use_batch_norm
        self.edge_counter = EdgeCounter()
        self.num_classes = num_classes
        self.graph_class = graph_class
        self.aggr = aggr

        if self.graph_class:
            self.no_prop = GraphExtractor(in_features=num_input_features, out_features=hidden_final, use_x=use_x,
                                          aggr=aggr)
        else:
            self.no_prop = NodeExtractor(in_features_u=num_input_features, out_features_u=hidden_final,
                                         aggr=aggr)
        self.initial_lin = nn.Linear(num_input_features, hidden)

        conv_layer = FastSMPLayer

        self.convs = nn.ModuleList()
        self.batch_norm_list = nn.ModuleList()
        self.feature_extractors = torch.nn.ModuleList([])
        for i in range(0, num_layers):
            self.convs.append(conv_layer(in_features=hidden, num_towers=num_towers, out_features=hidden, use_x=use_x,
                                         aggr=aggr))
            self.batch_norm_list.append(BatchNorm(hidden, use_x))
            if self.graph_class:
                self.feature_extractors.append(GraphExtractor(in_features=hidden, out_features=hidden_final,
                                                              use_x=use_x, simplified=simplified, aggr=self.aggr))
            else:
                self.feature_extractors.append(NodeExtractor(in_features_u=hidden, out_features_u=hidden_final,
                                                             aggr=aggr))

        # Last layers
        self.simplified = simplified
        self.after_conv = nn.Linear(hidden_final, hidden_final)
        self.final_lin = nn.Linear(hidden_final, num_classes)

    def forward(self, data):
        """ data.x: (num_nodes, num_features)"""
        x, edge_index = data.x, data.edge_index
        batch_info = create_batch_info(data, self.edge_counter)

        # Create the context matrix
        if self.use_x:
            assert x is not None
            u = x
        elif self.map_x_to_u:
            u = map_x_to_u(data, batch_info)
        else:
            u = data.x.new_zeros((data.num_nodes, batch_info['n_colors']))
            u.scatter_(1, data.coloring, 1)
            u = u[..., None]

        # Forward pass
        out = self.no_prop("ignored", u, batch_info)
        u = self.initial_lin(u)
        for i, (conv, bn, extractor) in enumerate(zip(self.convs, self.batch_norm_list, self.feature_extractors)):
            if self.use_batch_norm and i > 0:
                u = bn(u)
            u = conv(u, edge_index, batch_info)
            global_features = extractor.forward("ignored", u, batch_info)
            out += global_features / len(self.convs)

        # Two layer MLP with dropout and residual connections:
        if not self.simplified:
            out = torch.relu(self.after_conv(out)) + out
        out = F.dropout(out, p=self.dropout_prob, training=self.training)
        out = self.final_lin(out)
        if self.num_classes > 1:
            # Classification
            return F.log_softmax(out, dim=-1), 0
        else:
            # Regression
            assert out.shape[1] == 1
            return out[:, 0], 0

    def reset_parameters(self):
        for layer in [self.no_prop, self.initial_lin, *self.convs, *self.batch_norm_list, *self.feature_extractors,
                      self.after_conv, self.final_lin]:
            layer.reset_parameters()

    def __repr__(self):
        return self.__class__.__name__