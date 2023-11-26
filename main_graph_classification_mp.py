# This implementation is based on https://github.com/weihua916/powerful-gnns and https://github.com/chrsmrrs/k-gnn/tree/master/examples
import os.path as osp
import numpy as np
import time
import torch
import networkx as nx
import os
import shutil
import torch.nn.functional as F
import torch_geometric.transforms as T
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
from ptc import PTCDataset

if __name__ == "__main__":
    mp.set_start_method("spawn")


def run_on_split(sub, model, num_graphs, data):
    model = copy.deepcopy(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.001)
    while True:
        weights = sub.recv()
        model.load_state_dict(weights)
        opt.zero_grad(set_to_none=True)
        logs, aux_logs = model(data)
        loss = F.nll_loss(logs, data.y)
        loss.backward()
        if model.decoder.bias.grad is None:
            decoder_bias = model.skipdecoder.bias.grad
            decoder_weight = model.skipdecoder.weight.grad
        else:
            decoder_bias = model.decoder.bias.grad
            decoder_weight = model.decoder.weight.grad
        sub.send((model.encoder.bias.grad.detach() * num_graphs,
                  model.encoder.weight.grad.detach() * num_graphs,
                  model.newstate.bias.grad.detach() * num_graphs,
                  model.newstate.weight.grad.detach() * num_graphs,
                  model.new_message.bias.grad.detach() * num_graphs,
                  model.new_message.weight.grad.detach() * num_graphs,
                  decoder_bias * num_graphs,
                  decoder_weight * num_graphs,
                  loss.detach() * num_graphs))

def test_on_split(sub, model, data):
    while True:
        correct = 0
        weights = sub.recv()
        model.load_state_dict(weights)
        logs, aux_logs = model(data)
        pred = logs.max(1)[1]
        correct += pred.eq(data.y).sum().item()
        sub.send(correct)


def main(args, cluster=None):
    print(args, flush=True)

    if 'IMDB' in args.dataset: #IMDB-BINARY or #IMDB-MULTI
        class MyFilter(object):
            def __call__(self, data):
                return data.num_nodes <= 70
        class MyPreTransform(object):
            def __call__(self, data):
                data.x = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
                data.x = F.one_hot(data.x, num_classes=69).to(torch.float)
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
    elif 'DD' == args.dataset:
        class MyFilter(object):
            def __call__(self, data):
                return True

        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'DD')
        dataset = TUDataset(path, name='DD', pre_filter=MyFilter())
    elif 'REDDIT' in args.dataset:  # REDDIT-BINARY or REDDIT-MULTI-5K
        class MyFilter(object):
            def __call__(self, data):
                return True

        if args.one_hot_degree:
            path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', f'{args.dataset}')
            max_degree = 3062 if args.dataset == 'REDDIT-BINARY' else 2011
            dataset = TUDataset(
                path,
                name=args.dataset,
                pre_transform=T.OneHotDegree(max_degree),
                pre_filter=MyFilter())
        else:
            path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', f'{args.dataset}_constant')
            dataset = TUDataset(
                path,
                name=args.dataset,
                pre_transform=T.Constant(),
                pre_filter=MyFilter())
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
            losses.append(values[8])

        model.encoder.bias.grad = torch.sum(torch.stack(encoder_biases), dim=0) / num_graphs
        model.encoder.weight.grad = torch.sum(torch.stack(encoder_weights), dim=0) / num_graphs
        model.newstate.bias.grad = torch.sum(torch.stack(newstate_biases), dim=0) / num_graphs
        model.newstate.weight.grad = torch.sum(torch.stack(newstate_weights), dim=0) / num_graphs
        model.new_message.bias.grad = torch.sum(torch.stack(newmessage_biases), dim=0) / num_graphs
        model.new_message.weight.grad = torch.sum(torch.stack(newmessage_weights), dim=0) / num_graphs
        if args.use_skip:
            model.skipdecoder.bias.grad = torch.sum(torch.stack(decoder_biases), dim=0) / num_graphs
            model.skipdecoder.weight.grad = torch.sum(torch.stack(decoder_weights), dim=0) / num_graphs
        else:
            model.decoder.bias.grad = torch.sum(torch.stack(decoder_biases), dim=0) / num_graphs
            model.decoder.weight.grad = torch.sum(torch.stack(decoder_weights), dim=0) / num_graphs
        optimizer.step()
        print(sum(losses) / num_graphs, time.time() - start)
        return sum(losses) / num_graphs

    def test(num_graphs, test_processes, model):
        print(num_graphs)
        with torch.no_grad():
            correct = 0
            start = time.time()
            for process, queue in test_processes:
                queue.send(model.state_dict())
            for process, queue in test_processes:
                correct += queue.recv()
            print("Test time", time.time() - start)
        return correct / num_graphs

    acc = []
    splits = separate_data(len(dataset), seed=0)
    experimentstring = args.dataset + "hidden" + str(args.hidden_units) + "dropout" + str(args.dropout) + \
                       "skip" + str(args.use_skip) + "messages" + str(args.num_messages)
    with open("result/graph/{}_{}".format(experimentstring, args.seed), 'w') as f:
        for i, (train_idx, test_idx) in enumerate(splits):
            if i != args.seed:
                continue
            lr = args.lr

            train_dataset = dataset[train_idx.tolist()]
            test_dataset = dataset[test_idx.tolist()]

            train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
            train_data = None
            for data in train_loader:
                train_data = data.to_data_list()
            train_graph_split = []
            for _ in range(args.processes):
                train_graph_split.append([])
            for data_index, d in enumerate(train_data):
                train_graph_split[data_index % args.processes].append(d)

            test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
            test_data = None
            for data in test_loader:
                test_data = data.to_data_list()
            num_processes = min(len(test_data), args.processes)
            test_graph_split = []
            for _ in range(num_processes):
                test_graph_split.append([])
            for data_index, d in enumerate(test_data):
                test_graph_split[data_index % num_processes].append(d)


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

            test_processes = []
            for process_number in range(min(args.processes, len(test_data))):
                main, sub = mp.Pipe()
                p = mp.Process(target=test_on_split,
                               args=(sub, model, Batch.from_data_list(test_graph_split[process_number])))
                test_processes.append((p, main))
                p.start()


            print('---------------- Split {} ----------------'.format(i), flush=True)

            acc_temp = []
            for epoch in range(1, 350+1):
                start = time.time()
                train_loss = train(epoch, len(train_data), train_processes, model, optimizer)
                scheduler.step()
                test_acc = test(len(test_data), test_processes, model)
                if epoch%1 == 0:
                    print('Epoch: {:03d}, Train Loss: {:.7f}, '
                        'Val Loss: {:.7f}, Test Acc: {:.7f}, Time: {:7f}'.format(
                            epoch, train_loss, 0, test_acc, time.time() - start), flush=True)
                acc_temp.append(test_acc)
            for p, _ in train_processes:
                p.kill()
            for p, _ in test_processes:
                p.kill()
            f.write(str(acc_temp)+"\n")
            f.close()


if __name__ == '__main__':
    parser = HyperOptArgumentParser(strategy='grid_search')
    parser.opt_list('--dropout', type=float, default=0.0, tunable=True, options=[0.5, 0.0])
    parser.opt_list('--aggr', type=str, default='max', tunable=False, options=['mean', 'max', 'add'])
    parser.opt_list('--hidden_units', type=int, default=64, tunable=True, options=[16, 32])
    parser.opt_list('--use_skip', type=int, default=0, tunable=True, options=[0, 1])
    parser.opt_list('--seed', type=int, default=0, tunable=True, options=[0,1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.opt_list('--num_messages', type=int, tunable=True, default=15, options=[15, 25])
    parser.add_argument('--use_aux_loss', action='store_true', default=False)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--slurm', action='store_true', default=False)
    parser.add_argument('--grid_search', action='store_true', default=False)
    parser.add_argument('--gpu_jobs', action='store_true', default=False)
    parser.add_argument('--dataset', type=str, default='MUTAG', help="Options are ['MUTAG', 'PTC', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI']")
    parser.add_argument('--processes', type=int, default=8)

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
