# Synthetic datasets
import torch
from torch_geometric.utils import degree, from_networkx
from torch_geometric.data import DataLoader, Data
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random


def one_hot(size, pos=0):
    vector = torch.zeros(1, size)
    vector[0, pos] = 1
    return vector


def randomgraph(n, **args):
    g = nx.Graph()
    g.add_nodes_from(range(n))
    tree = set()
    nodes = list(range(n))
    current = random.choice(nodes)
    tree.add(current)
    while(len(tree) < n):
        nxt = random.choice(nodes)
        if not nxt in tree:
            tree.add(nxt)
            g.add_edge(current, nxt)
            g.add_edge(nxt, current)
        current = nxt
    for _ in range(n//5):
        i, j = np.random.permutation(n)[:2]
        while g.has_edge(i,j):
            i, j = np.random.permutation(n)[:2]
        g.add_edge(i, j)
        g.add_edge(j, i)
    return g

def get_localized_distances(g, n):
    seen = set()
    distances = {}
    queue = [(n, 0)]
    while queue:
        node, distance = queue.pop(0)
        if node in distances and distances[node] < distance:
            continue
        distances[node] = distance
        for nb in g.neighbors(node):
            if nb not in seen:
                seen.add(node)
                queue.append((nb, distance + 1))
    return [distances[i] for i in range(g.number_of_nodes())]


class OddEvenTask:
    def __init__(self, num_graphs, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_graphs = num_graphs
        self.graph_class = False
        self.message_size = 3
        self.num_features = 5
        self.hidden_units = 32
        self.num_classes = 2
        self.num_predictions = 1

    def makedata(self):
        graphs = []
        for _ in range(self.num_graphs):
            g = randomgraph(self.num_nodes)

            origin = random.randint(0, self.num_nodes - 1)
            queue = [(origin, 0)]
            seen = {origin}
            even = set()

            while queue:
                node, distance = queue.pop(0)
                if distance % 2 == 0:
                    even.add(node)
                for nb in g.neighbors(node):
                    if nb not in seen:
                        seen.add(nb)
                        queue.append((nb, distance + 1))
            data = from_networkx(g)
            data.x = torch.zeros(self.num_nodes, self.num_features).float()
            data.xa = torch.clone(data.x)
            data.x[origin:origin+1,:] = torch.ones(1, self.num_features).float()

            data.first_message = torch.zeros(self.num_nodes, 3)
            data.first_message[origin, 0] = 1
            data.starts = torch.zeros(self.num_nodes).int()
            data.starts[origin] = 1


            distances = get_localized_distances(g, origin)
            data.diameter = max(distances)
            data.distances = torch.tensor(distances).unsqueeze(1)
            data.edge_attr = torch.ones(g.number_of_edges()*2, 1)

            data.y = torch.tensor([[1] if n in even else [0] for n in range(self.num_nodes)])

            graphs.append(data)
        return graphs


class MultisourceOddeven:

    def __init__(self, num_graphs, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_graphs = num_graphs
        self.graph_class = False
        self.message_size = 6
        self.num_features = 6
        self.hidden_units = 30
        self.num_classes = 2
        self.num_origins = 2
        self.num_predictions = self.num_origins

    def makedata(self):
        graphs = []
        for _ in range(self.num_graphs):
            g = randomgraph(self.num_nodes)
            data = from_networkx(g)
            data.x = torch.zeros(self.num_nodes, self.num_features).float()
            data.xa = torch.clone(data.x)

            data.first_message = torch.zeros(self.num_nodes, self.message_size)
            data.starts = torch.zeros(self.num_nodes).int()

            y = torch.zeros(self.num_nodes, self.num_origins)
            origins = np.random.permutation(range(self.num_nodes))[:self.num_origins]
            data.edge_attr = torch.ones(g.number_of_edges()*2, 1)
            for i, origin in enumerate(origins):
                data.x[origin, i] = 1
                queue = [(origin, 0)]
                seen = {origin}
                even = set()

                while queue:
                    node, distance = queue.pop(0)
                    if distance % 2 == 0:
                        even.add(node)
                    for nb in g.neighbors(node):
                        if nb not in seen:
                            seen.add(nb)
                            queue.append((nb, distance + 1))
                for n in range(self.num_nodes):
                    y[n, i] = 1 if n in even else 0

                data.first_message[origin, i] = 1
                data.starts[origin] = 1

            distances0 = get_localized_distances(g, origins[0])
            distances1 = get_localized_distances(g, origins[1])
            #distances2 = get_localized_distances(g, origins[2])
            distances = torch.stack([torch.tensor(distances0), torch.tensor(distances1)], dim=1)#, torch.tensor(distances2)], dim=1)
            data.diameter = max([max(distances0), max(distances1)])#, max(distances2)])
            data.distances = distances
            data.y = y
            graphs.append(data)
            if False:
                print("__________")
                print(origins)
                print(data.y)
                print(data.starts)
                print(data.first_message)
                print(data.x)
        return graphs