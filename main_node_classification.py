# This implementation is based on https://github.com/weihua916/powerful-gnns and https://github.com/chrsmrrs/k-gnn/tree/master/examples
import os.path as osp
import numpy as np
import time
from collections import defaultdict
import torch
import networkx as nx
import os
import shutil
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch_geometric.data.dataloader import Collater
from torch_geometric.utils import degree
from torch_geometric.data import download_url, extract_zip, InMemoryDataset, Data
from sklearn.model_selection import StratifiedKFold
from test_tube import HyperOptArgumentParser
from test_tube.hpc import SlurmCluster

from models.graph.gwac_graph import GwACGraph
from models.graph.gwac_node import GwACNode
from models.graph.gin import GIN
from models.graph.gcn import GCN
from models.graph.gat import GAT
from models.graph.sage import SAGE
from models.graph.dropgnn import DropGIN

from ptc import PTCDataset


def main(args, cluster=None):
    print(args, flush=True)

    BATCH = args.batch_size
    if "Cora" == args.dataset:
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'Cora')
        dataset = Planetoid(path, "Cora", split='full')
    elif "CiteSeer" == args.dataset:
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'CiteSeer')
        dataset = Planetoid(path, "CiteSeer", split='full')
    elif "PubMed" == args.dataset:
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'PubMed')
        dataset = Planetoid(path, "PubMed", split='full')
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
    p = 2 * 1 / (1 + gamma)
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
    torch.manual_seed(0)
    np.random.seed(0)
    if args.model == 'gwac':
        model = GwACNode(dataset.num_features, args.hidden_units, dataset.num_classes, args.hidden_units // 2,
                          graph_class=False, num_starts=mean_n, num_messages=args.num_messages,
                          use_skip=args.use_skip)
    elif args.model == 'gin':
        model = GIN(dataset.num_features, args.hidden_units, dataset.num_classes, args.dropout)
    elif args.model == 'gcn':
        model = GCN(dataset.num_features, args.hidden_units, dataset.num_classes, args.dropout, False)
    elif args.model == 'gat':
        model = GAT(dataset.num_features, args.hidden_units, dataset.num_classes, args.dropout)
    elif args.model == 'sage':
        model = SAGE(dataset.num_features, args.hidden_units, dataset.num_classes, args.dropout, aggr=args.aggr)
    elif args.model == 'dropgnn':
        model = DropGIN(dataset.num_features, args.hidden_units, dataset.num_classes, args.dropout, args.use_aux_loss,
                        num_runs, p)

    def neighbors(data):
        nbs = defaultdict(set)
        source, dest = data.edge_index
        for node in range(data.num_nodes):
            for i in range(len(source)):
                if int(source[i]) == node:
                    nbs[node].add(int(dest[i]))
                if int(dest[i]) == node:
                    nbs[node].add(int(source[i]))
        return nbs

    def train(dataset, epoch, train_idx, optimizer, nbs):
        print("Epoch", epoch)
        start = time.time()
        model.train()
        optimizer.zero_grad()
        data = dataset.to(device)
        logs, aux_logs = model(data, p_starts=train_idx, nbs=nbs)
        loss = F.nll_loss(logs[train_idx, :], data.y[train_idx])
        loss.backward()
        optimizer.step()
        print(loss, time.time() - start)
        # with torch.no_grad():
        #    logs, aux_logs = model(data, optimizer)
        #    print(F.nll_loss(logs, data.y))
        return loss / train_idx.sum()

    def test(dataset, test_idx, nbs):
        model.eval()
        with torch.no_grad():
            correct = 0
            data = dataset.to(device)
            logs, aux_logs = model(data, p_starts=test_idx, nbs=nbs)
            pred = logs[test_idx, :].max(1)[1]
            correct += pred.eq(data.y[test_idx]).sum().item()
        return correct / test_idx.sum()

    acc = []
    print(model.__class__.__name__)
    data = dataset[0]
    nbs = neighbors(data)
    for i in range(10):
        if i != args.seed:
            continue
        # model.reset_parameters()
        lr = 0.01
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        print(data)
        train_idx = data.train_mask
        test_idx = data.test_mask

        print('---------------- Split {} ----------------'.format(i), flush=True)

        test_acc = 0.0
        acc_temp = []
        for epoch in range(1, 350 + 1):
            start = time.time()
            train_loss = train(data, epoch, train_idx, optimizer, nbs)
            print(train_loss)
            if epoch >= 300:
                test_acc = test(data, test_idx, nbs)
                acc_temp.append(test_acc)
                print('Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, '
                      'Val Loss: {:.7f}, Test Acc: {:.7f}, Time: {:7f}'.format(
                    epoch, lr, train_loss, 0, test_acc, time.time() - start), flush=True)
        skipstr = "skip" if args.use_skip else "noskip"
        with open("result/aaa{}_{}_{}_{}_{}_{}".format(args.dataset, args.model, args.hidden_units, args.num_messages,
                                                       skipstr, args.seed), 'w') as f:
            f.write(str(acc_temp))
        acc.append(torch.tensor(acc_temp))
    acc = torch.stack(acc, dim=0)
    acc_mean = acc.mean(dim=0)
    best_epoch = acc_mean.argmax().item()
    print('-----------------Results for seed {}----------------------'.format(args.seed))
    print('---------------- Final Epoch Result ----------------')
    print('Mean: {:7f}, Std: {:7f}'.format(acc[:, -1].mean(), acc[:, -1].std()))
    print(f'---------------- Best Epoch: {best_epoch} ----------------')
    print('Mean: {:7f}, Std: {:7f}'.format(acc[:, best_epoch].mean(), acc[:, best_epoch].std()), flush=True)
    print('-----------------------------Per epoch accuracy----------------------------------')
    print(acc)


if __name__ == '__main__':
    parser = HyperOptArgumentParser(strategy='grid_search')
    parser.opt_list('--dropout', type=float, default=0.0, tunable=True, options=[0.5, 0.0])
    parser.opt_list('--batch_size', type=int, default=32, tunable=False, options=[32, 128])
    parser.opt_list('--aggr', type=str, default='max', tunable=False, options=['mean', 'max', 'add'])  # for sage only
    parser.opt_list('--hidden_units', type=int, default=64, tunable=True, options=[16, 32])
    # 64 is used for social datasets (IMDB) and 16 or 32 for bio datasest (MUTAG, PTC, PROTEINS).
    # Set tunable=False to not grid search over this (for social datasets)
    parser.opt_list('--use_skip', type=int, default=0, tunable=True, options=[0, 1])
    parser.opt_list('--seed', type=int, default=0, tunable=True, options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--num_messages', type=int, default=15)
    parser.add_argument('--use_aux_loss', action='store_true', default=False)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--slurm', action='store_true', default=False)
    parser.add_argument('--grid_search', action='store_true', default=False)
    parser.add_argument('--gpu_jobs', action='store_true', default=False)
    parser.add_argument('--dataset', type=str, default='MUTAG',
                        help="Options are ['MUTAG', 'PTC', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI']")

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
            cluster.optimize_parallel_cluster_gpu(main, nb_trials=None, job_name=job_name,
                                                  job_display_name=args.dataset)
        else:
            cluster.per_experiment_nb_cpus = 8
            cluster.optimize_parallel_cluster_cpu(main, nb_trials=None, job_name=job_name,
                                                  job_display_name=args.dataset)
    elif args.grid_search:
        for hparam_trial in args.trials(None):
            main(hparam_trial)
    else:
        main(args)

    print('Finished', flush=True)
