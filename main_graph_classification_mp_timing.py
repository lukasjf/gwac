# This implementation is based on https://github.com/weihua916/powerful-gnns and https://github.com/chrsmrrs/k-gnn/tree/master/examples
import os.path as osp
import numpy as np
import time
import torch
import networkx as nx
import os
import shutil
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.data.dataloader import Collater
from torch_geometric.utils import degree
from torch_geometric.data import download_url, extract_zip, InMemoryDataset, Data
from sklearn.model_selection import StratifiedKFold
from test_tube import HyperOptArgumentParser
from test_tube.hpc import SlurmCluster
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
import torch.multiprocessing as mp
import copy
from collections import defaultdict
import math

from models.graph.gwac_graph import GwACGraph

if __name__ == "__main__":
    mp.set_start_method("spawn")


def run_on_split(sub, model, num_graphs, data):
    model = copy.deepcopy(model)
    while True:
        weights = sub.recv()
        model.load_state_dict(weights)
        logs, aux_logs = model(data)
        loss = F.nll_loss(logs, data.y)
        loss.backward()
        sub.send((model.encoder.bias.grad.detach() * num_graphs,
                  model.encoder.weight.grad.detach() * num_graphs,
                  model.newstate.bias.grad.detach() * num_graphs,
                  model.newstate.weight.grad.detach() * num_graphs,
                  model.new_message.bias.grad.detach() * num_graphs,
                  model.new_message.weight.grad.detach() * num_graphs,
                  model.decoder.bias.grad.detach() * num_graphs,
                  model.decoder.weight.grad.detach() * num_graphs,
                  #model.skipdecoder.bias.grad.detach() * num_graphs if model.skipdecoder.bias.grad is not None else 0,
                  #model.skipdecoder.weight.grad.detach() * num_graphs if model.skipdecoder.weight.grad is not None else 0,
                  loss.detach() * num_graphs))

def test_on_split(sub, model, data):
    print(data)
    while True:
        correct = 0
        weights = sub.recv()
        model.load_state_dict(weights)
        logs, aux_logs = model(data)
        pred = logs.max(1)[1]
        correct += pred.eq(data.y).sum().item()
        sub.send((correct))


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0

def S2V_to_PyG(data):
    new_data = Data()
    setattr(new_data, 'edge_index', data.edge_mat)
    setattr(new_data, 'x', data.node_features)
    setattr(new_data, 'num_nodes', data.node_features.shape[0])
    setattr(new_data, 'y', torch.tensor(data.label).unsqueeze(0).long())

    return new_data

def load_data(dataset, degree_as_tag, folder):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('%s/%s.txt' % (folder, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

    # add labels and edge_mat
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0, 1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    # Extracting unique tag labels
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]: i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1

    return [S2V_to_PyG(datum) for datum in g_list]


class PTCDataset(InMemoryDataset):
    def __init__(
            self,
            root,
            name,
            transform=None,
            pre_transform=None,
    ):
        self.name = name
        self.url = 'https://github.com/weihua916/powerful-gnns/raw/master/dataset.zip'

        super(PTCDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        name = 'raw'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed'
        return osp.join(self.root, self.name, name)

    @property
    def num_tasks(self):
        return 1  # it is always binary classification for the datasets we consider

    @property
    def eval_metric(self):
        return 'acc'

    @property
    def task_type(self):
        return 'classification'

    @property
    def raw_file_names(self):
        return ['PTC.mat', 'PTC.txt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        folder = osp.join(self.root, self.name)
        path = download_url(self.url, folder)
        extract_zip(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)

        shutil.move(osp.join(folder, f'dataset/{self.name}'), osp.join(folder, self.name))
        shutil.rmtree(osp.join(folder, 'dataset'))

        os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
        # Read data into huge `Data` list.
        data_list = load_data('PTC', degree_as_tag=False, folder=self.raw_dir)
        print(sum([data.num_nodes for data in data_list]))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def separate_data(self, seed, fold_idx):
        # code taken from GIN and adapted
        # since we only consider train and valid, use valid as test

        assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

        labels = self.data.y.numpy()
        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):
            idx_list.append(idx)
        train_idx, test_idx = idx_list[fold_idx]

        return {'train': torch.tensor(train_idx), 'valid': torch.tensor(test_idx), 'test': torch.tensor(test_idx)}


def main(args, cluster=None):
    print(args, flush=True)

    BATCH = args.batch_size

    if 'IMDB' in args.dataset: #IMDB-BINARY or #IMDB-MULTI
        class MyFilter(object):
            def __call__(self, data):
                return data.num_nodes <= 70
        class MyPreTransform(object):
            def __call__(self, data):
                data.x = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
                data.x = F.one_hot(data.x, num_classes=69).to(torch.float)#136 in k-gnn?
                return data
        path = osp.join(
            osp.dirname(osp.realpath(__file__)), 'data', f'{args.dataset}')
        dataset = TUDataset(
            path,
            name=args.dataset,
            pre_transform=MyPreTransform(),
            pre_filter=MyFilter())
    elif 'MUTAG' in args.dataset:
        class MyFilter(object):
            def __call__(self, data):
                return True
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'MUTAG')
        dataset = TUDataset(path, name='MUTAG', pre_filter=MyFilter())
    elif 'PROTEINS' in args.dataset:
        class MyFilter(object):
            def __call__(self, data):
                return not (data.num_nodes == 7 and data.num_edges == 12) and data.num_nodes < 450
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'PROTEINS')
        dataset = TUDataset(path, name='PROTEINS', pre_filter=MyFilter())
    elif 'PTC' == args.dataset:
        class MyFilter(object):
            def __call__(self, data):
                return True        
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'PTC')
        dataset = TUDataset(path, name='PTC_MR', pre_filter=MyFilter())
    elif 'GINPTC' == args.dataset:
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'GINPTC')
        dataset = PTCDataset(path, "PTC")
    else:
        raise ValueError

    print(dataset)

    # Set the sampling probability and number of runs/samples for the DropGIN
    n = []
    degs = []
    for g in dataset:
        num_nodes = g.num_nodes
        deg = degree(g.edge_index[0], g.num_nodes, dtype=torch.long)
        n.append(g.num_nodes)
        degs.append(deg.max())
    print(f'Mean Degree: {torch.stack(degs).float().mean()}')
    print(f'Max Degree: {torch.stack(degs).max()}')
    print(f'Min Degree: {torch.stack(degs).min()}')
    mean_n = torch.tensor(n).float().mean().round().long().item()
    print(f'Mean number of nodes: {mean_n}')
    print(f'Max number of nodes: {torch.tensor(n).float().max().round().long().item()}')
    print(f'Min number of nodes: {torch.tensor(n).float().min().round().long().item()}')
    print(f'Number of graphs: {len(dataset)}')
    gamma = mean_n
    p = 2 * 1 /(1+gamma)
    num_runs = gamma
    print(f'Number of runs: {num_runs}')
    print(f'Sampling probability: {p}')

    def separate_data(dataset_len, seed=0):
        # Use same splitting/10-fold as GIN paper
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        idx_list = []
        for idx in skf.split(np.zeros(dataset_len), np.zeros(dataset_len)):
            idx_list.append(idx)
        return idx_list

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    def train(epoch, num_graphs, train_processes, model, optimizer):
        optimizer.zero_grad()
        print("Epoch", epoch)
        start = time.time()
        for process, queue in train_processes:
            queue.send(model.state_dict())
        encoder_biases = []
        encoder_weights = []
        newstate_biases = []
        newstate_weights = []
        newmessage_biases = []
        newmessage_weights = []
        decoder_biases = []
        decoder_weights = []
        skipdecoder_biases = []
        skipdecoder_weights = []
        losses = []
        for process, queue in train_processes:
            values = queue.recv()
            encoder_biases.append(values[0])
            encoder_weights.append(values[1])
            newstate_biases.append(values[2])
            newstate_weights.append(values[3])
            newmessage_biases.append(values[4])
            newmessage_weights.append(values[5])
            decoder_biases.append(values[6])
            decoder_weights.append(values[7])
            #skipdecoder_biases.append(values[8])
            #skipdecoder_weights.append(values[9])
            losses.append(values[8])

        model.encoder.bias.grad = torch.sum(torch.stack(encoder_biases), dim=0) / num_graphs
        model.encoder.weight.grad = torch.sum(torch.stack(encoder_weights), dim=0) / num_graphs
        model.newstate.bias.grad = torch.sum(torch.stack(newstate_biases), dim=0) / num_graphs
        model.newstate.weight.grad = torch.sum(torch.stack(newstate_weights), dim=0) / num_graphs
        model.new_message.bias.grad = torch.sum(torch.stack(newmessage_biases), dim=0) / num_graphs
        model.new_message.weight.grad = torch.sum(torch.stack(newmessage_weights), dim=0) / num_graphs
        model.decoder.bias.grad = torch.sum(torch.stack(decoder_biases), dim=0) / num_graphs
        model.decoder.weight.grad = torch.sum(torch.stack(decoder_weights), dim=0) / num_graphs
        #model.skipdecoder.bias.grad = torch.sum(torch.stack(skipdecoder_biases), dim=0) / num_graphs
        #model.skipdecoder.weight.grad = torch.sum(torch.stack(skipdecoder_weights), dim=0) / num_graphs
        optimizer.step()
        end = time.time() - start
        return sum(losses) / num_graphs, end

    def test(num_graphs, test_processes, model):
        with torch.no_grad():
            correct = 0
            start = time.time()
            for process, queue in test_processes:
                queue.send(model.state_dict())
            for process, queue in test_processes:
                correct += queue.recv()
            print("Test time", time.time() - start)
        return correct / num_graphs

    splits = separate_data(len(dataset), seed=0)
    with open("result/timing{}_{}.csv".format(args.dataset, args.processes), 'w') as f:
        for i, (train_idx, test_idx) in enumerate(splits):
            if i != 0:
                continue
            lr = 0.01

            train_dataset = dataset[train_idx.tolist()]

            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=len(train_dataset),
                                                       collate_fn=Collater(follow_batch=[],exclude_keys=[]))
            train_data = None
            for data in train_loader:
                train_data = data.to_data_list()
            train_graph_split = []
            for _ in range(args.processes):
                train_graph_split.append([])
            for data_index, d in enumerate(train_data):
                train_graph_split[data_index % args.processes].append(d)

            torch.manual_seed(0)
            np.random.seed(0)
            model = GwACGraph(dataset.num_features, args.hidden_units, dataset.num_classes, args.hidden_units // 2,
                              graph_class=True, num_starts=mean_n, num_messages=args.num_messages,
                              use_skip=args.use_skip)
            model.share_memory()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5) # in GIN code 50 itters per epoch were used

            train_processes = []
            for process_number in range(args.processes):
                main, sub = mp.Pipe()
                p = mp.Process(target=run_on_split,
                               args=(sub, model, len(train_graph_split[process_number]),
                                     Batch.from_data_list(train_graph_split[process_number])))
                train_processes.append((p, main))
                p.start()


            print('---------------- Split {} ----------------'.format(i), flush=True)

            for epoch in range(1, 26+1):
                train_loss, tr_time = train(epoch, len(train_data), train_processes, model, optimizer)
                scheduler.step()
                if epoch%1 == 0:
                    print('Epoch: {:03d}, Time: {:7f}'.format(epoch, tr_time), flush=True)
                f.write("{},{},{}\n".format(args.dataset, args.processes, tr_time))
            for p, _ in train_processes:
                p.kill()


if __name__ == '__main__':
    parser = HyperOptArgumentParser(strategy='grid_search')
    parser.opt_list('--dropout', type=float, default=0.0, tunable=True, options=[0.5, 0.0])
    parser.opt_list('--batch_size', type=int, default=32, tunable=False, options=[32, 128])
    parser.opt_list('--aggr', type=str, default='max', tunable=False, options=['mean', 'max', 'add']) # for sage only
    parser.opt_list('--hidden_units', type=int, default=64, tunable=True, options=[16, 32])
    # 64 is used for social datasets (IMDB) and 16 or 32 for bio datasest (MUTAG, PTC, PROTEINS).
    # Set tunable=False to not grid search over this (for social datasets)
    parser.opt_list('--use_skip', type=int, default=0, tunable=True, options=[0, 1])
    parser.opt_list('--seed', type=int, default=0, tunable=False, options=[0,1,2,3,4,5,6,7,8,9])
    parser.add_argument('--num_messages', type=int, default=15)
    parser.add_argument('--use_aux_loss', action='store_true', default=False)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--slurm', action='store_true', default=False)
    parser.add_argument('--grid_search', action='store_true', default=False)
    parser.add_argument('--gpu_jobs', action='store_true', default=False)
    parser.add_argument('--dataset', type=str, default='MUTAG', help="Options are ['MUTAG', 'PTC', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI']")
    parser.add_argument('--processes', type=int, default=3)

    args = parser.parse_args()

    if args.slurm:
        print('Launching SLURM jobs')
        cluster = SlurmCluster(
            hyperparam_optimizer=args,
            log_path='slurm_log/',
            python_cmd='python'
        )
        cluster.job_time = '24:00:00'
        
        cluster.memory_mb_per_node = '8G'
        job_name = args.dataset
        if args.gpu_jobs:
            cluster.per_experiment_nb_cpus = 2
            cluster.per_experiment_nb_gpus = 1
            cluster.optimize_parallel_cluster_gpu(main, nb_trials=None, job_name=job_name, job_display_name=args.dataset)
        else:
            cluster.per_experiment_nb_cpus = 8
            cluster.optimize_parallel_cluster_cpu(main, nb_trials=None, job_name=job_name, job_display_name=args.dataset)
    elif args.grid_search:
        for hparam_trial in args.trials(None):
            main(hparam_trial)
    else:
        main(args)

    print('Finished', flush=True)
