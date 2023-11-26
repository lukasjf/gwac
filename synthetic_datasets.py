# Synthetic datasets
import torch
from torch_geometric.utils import degree, from_networkx, to_networkx
from torch_geometric.data import DataLoader, Data
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class SymmetrySet:
    def __init__(self):
        self.hidden_units = 0
        self.num_classes = 0
        self.num_features = 0
        self.num_nodes = 0

    def addports(self, data):
        data.ports = torch.zeros(data.num_edges, 1)
        degs = degree(data.edge_index[0], data.num_nodes, dtype=torch.long) # out degree of all nodes
        for n in range(data.num_nodes):
            deg = degs[n]
            ports = np.random.permutation(int(deg))
            for i, neighbor in enumerate(data.edge_index[1][data.edge_index[0]==n]):
                nb = int(neighbor)
                data.ports[torch.logical_and(data.edge_index[0]==n, data.edge_index[1]==nb), 0] = float(ports[i])
        return data

    def makefeatures(self, data):
        data.x = torch.ones((data.num_nodes, 1))
        data.id = torch.tensor(np.random.permutation(np.arange(data.num_nodes))).unsqueeze(1)
        return data

    def makedata(self):
        pass


class LimitsOne(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.hidden_units = 16
        self.num_classes = 2
        self.num_features = 4
        self.num_nodes = 8
        self.graph_class = False
        self.first_message = torch.ones(1, 3)

    def makedata(self):
        n_nodes = 16  # There are two connected components, each with 8 nodes

        ports = [1, 1, 2, 2] * 8
        colors = [0, 1, 2, 3] * 4

        y = torch.tensor([0] * 8 + [1] * 8)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
                                    13, 14, 14, 15, 15, 8],
                                   [1, 0, 2, 1, 3, 2, 0, 3, 5, 4, 6, 5, 7, 6, 4, 7, 9, 8, 10, 9, 11, 10, 12, 11, 13, 12,
                                    14, 13, 15, 14, 8, 15]],
                                  dtype=torch.long)
        x = torch.zeros((n_nodes, 4))
        x[range(n_nodes), colors] = 1

        data = Data(x=x, edge_index=edge_index, y=y)
        data.id = torch.tensor(np.random.permutation(np.arange(n_nodes))).unsqueeze(1)
        data.ports = torch.tensor(ports).unsqueeze(1)
        data.starts = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        return [data]


class LimitsTwo(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.hidden_units = 16
        self.num_classes = 2
        self.num_features = 4
        self.num_nodes = 8
        self.graph_class = False
        self.first_message = torch.ones(1, 3)

    def makedata(self):
        n_nodes = 16 # There are two connected components, each with 8 nodes

        ports = ([1,1,2,2,1,1,2,2] * 2 + [3,3,3,3]) * 2
        colors = [0, 1, 2, 3] * 4
        y = torch.tensor([0] * 8 + [1] * 8)
        edge_index = torch.tensor([[0,1,1,2,2,3,3,0, 4,5,5,6,6,7,7,4, 1,3,5,7, 8,9,9,10,10,11,11,8, 12,13,13,14,14,15,15,12, 9,15,11,13],
                                   [1,0,2,1,3,2,0,3, 5,4,6,5,7,6,4,7, 3,1,7,5, 9,8,10,9,11,10,8,11, 13,12,14,13,15,14,12,15, 15,9,13,11]], dtype=torch.long)
        x = torch.zeros((n_nodes, 4))
        x[range(n_nodes), colors] = 1

        data = Data(x=x, edge_index=edge_index, y=y)
        data.id = torch.tensor(np.random.permutation(np.arange(n_nodes))).unsqueeze(1)
        data.ports = torch.tensor(ports).unsqueeze(1)
        data.starts = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        return [data]


class Triangles(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.hidden_units = 16
        self.num_classes = 2
        self.num_features = 1
        self.num_nodes = 60
        self.graph_class = False
        self.first_message = torch.ones(1, 3)

    def makedata(self):
        size = self.num_nodes
        generated = False
        while not generated:
            nx_g = nx.random_degree_sequence_graph([3] * size)
            data = from_networkx(nx_g)
            labels = [0] * size
            for n in range(size):
                for nb1 in data.edge_index[1][data.edge_index[0]==n]:
                    for nb2 in data.edge_index[1][data.edge_index[0]==n]:
                        if torch.logical_and(data.edge_index[0]==nb1, data.edge_index[1]==nb2).any():
                            labels[n] = 1
            generated = labels.count(0) >= 20 and labels.count(1) >= 20
        data.y = torch.tensor(labels)
        data = self.addports(data)
        data = self.makefeatures(data)
        data.starts = torch.tensor([1] + 15 * [0])
        return [data]


class LCC(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.hidden_units = 16
        self.num_classes = 3
        self.num_features = 1
        self.num_nodes = 10
        self.graph_class = False
        self.first_message = torch.ones(1, 3)

    def makedata(self):
        generated = False
        while not generated:
            graphs = []
            labels = []
            i = 0
            while i < 6:
                size = 10
                nx_g = nx.random_degree_sequence_graph([3] * size)
                if nx.is_connected(nx_g):
                    i += 1
                    data = from_networkx(nx_g)
                    lbls = [0] * size
                    for n in range(size):
                        edges = 0
                        nbs = [int(nb) for nb in data.edge_index[1][data.edge_index[0]==n]]
                        for nb1 in nbs:
                            for nb2 in nbs:
                                if torch.logical_and(data.edge_index[0]==nb1, data.edge_index[1]==nb2).any():
                                    edges += 1
                        lbls[n] = int(edges/2)
                    data.y = torch.tensor(lbls)
                    labels.extend(lbls)
                    data = self.addports(data)
                    data = self.makefeatures(data)
                    data.starts = torch.tensor([1] + [0] * (size - 1))
                    graphs.append(data)
            # Ensure the dataset is somewhat balanced
            generated = labels.count(0) >= 10 and labels.count(1) >= 10 and labels.count(2) >= 10
        return graphs


class MaxGraphs(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.num_nodes = 21
        self.num_classes = self.num_nodes - 1
        self.hidden_units = 32
        self.num_features = self.hidden_units
        self.graph_class = True
        self.first_message = torch.ones(1, 3)

    def makedata(self):
        neighbors = list(range(1, self.num_nodes))
        dataset = []
        for num_nb in neighbors:
            colors = [0] + [1] * num_nb
            y = num_nb - 1
            edge_index = torch.tensor([[0] * num_nb + list(range(1, num_nb+1)),
                                       list(range(1, num_nb+1)) + [0] * num_nb])
            x = torch.zeros((num_nb+1, self.hidden_units))
            x[range(num_nb+1), colors] = 1
            data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nb+1)
            data.id = torch.tensor(np.random.permutation(np.arange(num_nb+1))).unsqueeze(1)
            data = self.addports(data)
            data.starts = torch.tensor([1] + [0] * num_nb)
            dataset.append(data)
        return dataset


class MeanGraphs(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.num_nodes = 21
        self.num_classes = (self.num_nodes - 1) // 2
        self.hidden_units = 32
        self.num_features = self.hidden_units
        self.graph_class = True
        self.first_message = torch.ones(1, 3)

    def makedata(self):
        neighbors = list(range(1, self.num_classes + 1))
        dataset = []
        for num_nb in neighbors:
            num_nodes = 2 * num_nb + 1
            colors = [0] +  [1] * num_nb + [2] * num_nb
            y = num_nb - 1
            sources = [0] * (2*num_nb) + list(range(1, 2 * num_nb + 1))
            dest = list(range(1, 2 * num_nb + 1)) + [0] * (2*num_nb)
            edge_index = torch.tensor([sources, dest])
            x = torch.zeros((num_nodes, self.hidden_units))
            x[range(num_nodes), colors] = 1
            data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)
            data.id = torch.tensor(np.random.permutation(np.arange(num_nodes))).unsqueeze(1)
            data.starts = torch.tensor([1] + [0] * 2 * num_nb)
            data = self.addports(data)
            g = to_networkx(data)
            #plt.show()
            dataset.append(data)
        return dataset


class FourCycles(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.p = 4
        self.hidden_units = 16
        self.num_classes = 2
        self.num_features = 1
        self.num_nodes = 4 * self.p
        self.graph_class = True
        self.first_message = torch.ones(1, 3)

    def gen_graph(self, p):
        edge_index = None
        for i in range(p):
            e = torch.tensor([[i, p + i, 2 * p + i, 3 * p + i], [2 * p + i, 3 * p + i, i, p + i]], dtype=torch.long)
            if edge_index is None:
                edge_index = e
            else:
                edge_index = torch.cat([edge_index, e], dim=-1)
        top = np.zeros((p * p,))
        perm = np.random.permutation(range(p))
        for i, t in enumerate(perm):
            top[i * p + t] = 1
        bottom = np.zeros((p * p,))
        perm = np.random.permutation(range(p))
        for i, t in enumerate(perm):
            bottom[i * p + t] = 1
        for i, bit in enumerate(top):
            if bit:
                e = torch.tensor([[i // p, p + i % p], [p + i % p, i // p]], dtype=torch.long)
                edge_index = torch.cat([edge_index, e], dim=-1)
        for i, bit in enumerate(bottom):
            if bit:
                e = torch.tensor([[2 * p + i // p, 3 * p + i % p], [3 * p + i % p, 2 * p + i // p]], dtype=torch.long)
                edge_index = torch.cat([edge_index, e], dim=-1)
        return Data(edge_index=edge_index, num_nodes=self.num_nodes), any(np.logical_and(top, bottom))

    def makedata(self):
        size = 25
        p = self.p
        trues = []
        falses = []
        while len(trues) < size or len(falses) < size:
            data, label = self.gen_graph(p)
            data = self.makefeatures(data)
            data = self.addports(data)
            data.starts = torch.tensor([1] + [0] * 15)
            data.y = label
            if label and len(trues) < size:
                trues.append(data)
            elif not label and len(falses) < size:
                falses.append(data)
        return trues + falses


class SkipCircles(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.hidden_units = 32
        self.num_classes = 10  # num skips
        self.num_features = 1
        self.num_nodes = 41
        self.graph_class = True
        self.first_message = torch.ones(1, 3)
        self.makedata()

    def makedata(self):
        size=self.num_nodes
        skips = [2, 3, 4, 5, 6, 9, 11, 12, 13, 16]
        graphs = []
        for s, skip in enumerate(skips):
            edge_index = torch.tensor([[0, size-1], [size-1, 0]], dtype=torch.long)
            for i in range(size - 1):
                e = torch.tensor([[i, i+1], [i+1, i]], dtype=torch.long)
                edge_index = torch.cat([edge_index, e], dim=-1)
            for i in range(size):
                e = torch.tensor([[i, i], [(i - skip) % size, (i + skip) % size]], dtype=torch.long)
                edge_index = torch.cat([edge_index, e], dim=-1)
            data = Data(edge_index=edge_index, num_nodes=self.num_nodes)
            data = self.makefeatures(data)
            data = self.addports(data)
            data.starts = torch.tensor([1] + [0] * 40)
            data.y = torch.tensor(s)
            graphs.append(data)

        return graphs


class RookShrikande(SymmetrySet):

    def __init__(self):
        super().__init__()
        self.hidden_units = 32
        self.num_classes = 2
        self.num_features = 4
        self.num_nodes = 16
        self.graph_class = True
        self.first_message = torch.ones(1, 3)
        self.makedata()

    def makedata(self):
        n = 16
        edges1 = [[i]* 6 for i in range(n)]
        edges1 = [elem for lst in edges1 for elem in lst]
        x = torch.zeros((n, 4))

        rook_edges =   [1,2,3,4,8,12,   2,3,0,5,9,13,   3,0,1,6,10,14,   0,1,2,7,11,15,
                        5,6,7,8,12,0,   6,7,4,9,13,1,   7,4,5,10,14,2,   4,5,6,11,15,3,
                        9,10,11,12,0,4, 10,11,8,13,1,5, 11,8,9,14,2,6,   8,9,10,15,3,7,
                        13,14,15,0,4,8, 14,15,12,1,5,9, 15,12,13,2,6,10, 12,13,14,3,7,11]

        shirkande_edges = [1,3,4,12,5,15,   0,2,5,13,6,12,   1,3,6,14,7,13,   0,2,7,15,4,14,
                           5,7,8,0,9,3,     4,6,9,1,10,0,    5,7,10,2,11,1,   4,6,11,3,8,2,
                           9,11,12,4,13,7,  8,10,13,5,14,4,  9,11,14,6,15,5,  8,10,15,7,12,6,
                           13,15,0,8,1,11,  12,14,1,9,2,8,   13,15,2,10,3,9,  12,14,3,11,0,10]

        rook = Data(x=x, edge_index=torch.tensor([edges1, rook_edges]), y=torch.tensor(0))
        shirkande = Data(x=x, edge_index=torch.tensor([edges1, shirkande_edges]), y=torch.tensor(1))
        return [rook, shirkande]