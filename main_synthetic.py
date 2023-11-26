# This implementation is based on https://github.com/weihua916/powerful-gnns and https://github.com/chrsmrrs/k-gnn/tree/master/examples
# Datasets are implemented based on the description in the corresonding papers (see the paper for references)
import argparse
from synthetic_datasets import *
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree
from torch_geometric.nn import GINConv, GINEConv

from models.synthetic.gin import GIN
from models.synthetic.gcn import GCN
from models.synthetic.gat import GAT
from models.synthetic.dropgnn import DropGIN
from models.synthetic.smp.smp import SMP
from models.synthetic.ppgn import Powerful
from models.synthetic.esan.esan import DSnetwork, DSSnetwork
from models.synthetic.gwac_constant import ConstantDelayGwAC
from models.synthetic.gwac_random import RandomDelayGwAC


torch.set_printoptions(profile="full")

def main(args, cluster=None):
    print(args, flush=True)

    if args.dataset == "skipcircles":
        dataset = SkipCircles()
    elif args.dataset == "triangles":
        dataset = Triangles()
    elif args.dataset == "lcc":
        dataset = LCC()
    elif args.dataset == "limitsone":
        dataset = LimitsOne()
    elif args.dataset == "limitstwo":
        dataset = LimitsTwo()
    elif args.dataset == "fourcycles":
        dataset = FourCycles()
    elif args.dataset == "maxlimit":
        dataset = MaxGraphs()
    elif args.dataset == "meanlimit":
        dataset = MeanGraphs()
    elif args.dataset == "rookshrikande":
        dataset = RookShrikande()
    else:
        1/0

    print(dataset.__class__.__name__)

    # Set the sampling probability and number of runs/samples for the DropGIN
    n = dataset.num_nodes
    print(f'Number of nodes: {n}')
    gamma = n
    p_opt = 2 * 1 /(1+gamma)
    if args.prob >= 0:
        p = args.prob
    else:
        p = p_opt
    if args.num_runs > 0:
        num_runs = args.num_runs
    else:
        num_runs = gamma
    print(f'Number of runs: {num_runs}')
    print(f'Sampling probability: {p}')

    degs = []
    for g in dataset.makedata():
        deg = degree(g.edge_index[0], g.num_nodes, dtype=torch.long)
        degs.append(deg.max())
    print(f'Mean Degree: {torch.stack(degs).float().mean()}')
    print(f'Max Degree: {torch.stack(degs).max()}')
    print(f'Min Degree: {torch.stack(degs).min()}')
    print(f'Number of graphs: {len(dataset.makedata())}')

    graph_classification = dataset.graph_class
    if graph_classification:
        print('Graph Clasification Task')
    else:
        print('Node Clasification Task')
    
    num_features = dataset.num_features
    if args.model == 'ids' or args.model == 'random':
        num_features += 1

    use_aux_loss = args.use_aux_loss

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if args.dataset == "maxlimit":
        aggr = "max"
    elif args.dataset == "meanlimit":
        aggr = "mean"
    else:
        aggr = "add"
    if args.model == 'dropout':
        model = DropGIN(num_features=num_features, num_hidden=dataset.hidden_units, num_classes=dataset.num_classes,
                        num_layers=args.num_layers, runs=num_runs, p=p, use_aux_loss=use_aux_loss,
                        graph_class=dataset.graph_class, aggr=aggr).to(device)
    elif args.model == 'smp':
        model = SMP(num_input_features=num_features+1, num_classes=dataset.num_classes, num_layers=args.num_layers,
                    hidden=dataset.hidden_units, layer_type="FastSMP", hidden_final=dataset.hidden_units,
                    dropout_prob=0.5, use_batch_norm=False, use_x=False, map_x_to_u=True, num_towers=1,
                    simplified=False, graph_class=dataset.graph_class, aggr=aggr)
    elif args.model == 'ppgn':
        model = Powerful(num_classes=dataset.num_classes, num_layers=args.num_layers, num_features=num_features,
                         hidden=dataset.hidden_units, hidden_final=dataset.hidden_units, dropout_prob=0.5,
                         simplified=False, graph_class=dataset.graph_class, aggr=aggr)
    elif args.model == 'dss':
        model = DSSnetwork(num_layers=args.num_layers, in_dim=num_features, emb_dim=dataset.hidden_units,
                           num_tasks=dataset.num_classes, num_nodes=dataset.num_nodes, graph_class=dataset.graph_class,
                           permute=args.permutation, aggr=aggr)
    elif args.model == 'ds':
        model = DSnetwork(num_layers=args.num_layers, in_dim=num_features, emb_dim=dataset.hidden_units,
                          num_tasks=dataset.num_classes, num_nodes=dataset.num_nodes, graph_class=dataset.graph_class,
                          permute=args.permutation, aggr=aggr)
    elif args.model == 'gwacc':
        model = ConstantDelayGwAC(in_features=num_features, hidden=dataset.hidden_units, out_features=dataset.num_classes,
                                  message_size=dataset.first_message.size()[1], graph_class=dataset.graph_class,
                                  first_message=dataset.first_message, aggr=aggr)
    elif args.model == 'gwacr':
        model = RandomDelayGwAC(in_features=num_features, hidden=dataset.hidden_units, out_features=dataset.num_classes,
                                message_size=dataset.first_message.size()[1], graph_class=dataset.graph_class,
                                first_message=dataset.first_message, aggr=aggr)
    elif args.model == 'gcn':
        model = GCN(num_features=num_features, num_hidden=dataset.hidden_units,
                    num_classes=dataset.num_classes,
                    num_layers=args.num_layers, graph_class=dataset.graph_class)
    elif args.model == 'gat':
        model = GAT(num_features=num_features, num_hidden=dataset.hidden_units,
                                num_classes=dataset.num_classes,
                                num_layers=args.num_layers, graph_class=dataset.graph_class)
    else:
        conv = GINEConv if args.model == "ports" else GINConv
        model = GIN(num_features=num_features, num_hidden=dataset.hidden_units, num_classes=dataset.num_classes,
                    num_layers=args.num_layers, graph_class=dataset.graph_class, conv=conv,
                    augmentation=args.model, aggr=aggr).to(device)
        use_aux_loss = False

    def train(epoch, loader, optimizer):
        model.train()
        loss_all = 0
        n = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            logs, aux_logs = model(data)
            loss = F.nll_loss(logs, data.y)
            n += len(data.y)
            if use_aux_loss:
                aux_loss = F.nll_loss(aux_logs.view(-1, aux_logs.size(-1)), data.y.unsqueeze(0).expand(aux_logs.size(0),-1).clone().view(-1))
                loss = 0.75*loss + 0.25*aux_loss
            loss.backward()
            loss_all += data.num_graphs * loss.item()
            if epoch % 100 == 99:
                print(epoch, loss.item())
            optimizer.step()
        return loss_all / len(loader.dataset)

    def test(loader):
        model.eval()
        n = 0
        with torch.no_grad():
            correct = 0
            for data in loader:
                data = data.to(device)
                logs, aux_logs = model(data)
                pred = logs.max(1)[1]
                n += len(pred)
                correct += pred.eq(data.y).sum().item()
        return correct / n

    def train_and_test(multiple_tests=False, test_over_runs=None):
        train_accs = []
        test_accs = []
        nonlocal num_runs # access global num_runs variable inside this function
        print(model.__class__.__name__)
        for seed in range(10):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model.reset_parameters()
            lr = 0.01
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            test_dataset = dataset.makedata()
            train_dataset = dataset.makedata()

            test_loader = DataLoader(test_dataset, batch_size=len(train_dataset))
            train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

            print('---------------- Seed {} ----------------'.format(seed))
            for epoch in range(1, 1001):
                if args.verbose:
                    start = time.time()
                train_loss = train(epoch, train_loader, optimizer)
                if args.verbose:
                    print('Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, Time: {:7f}'.format(epoch, lr, train_loss, time.time() - start), flush=True)
            train_acc = test(train_loader)
            train_accs.append(train_acc)
            if not test_over_runs is None:
                if multiple_tests:
                    for i in range(10):
                        old_num_runs = num_runs
                        for r in test_over_runs:
                            num_runs = r
                            test_acc = test(test_loader)
                            test_accs.append(test_acc)
                        num_runs = old_num_runs
                else:
                    old_num_runs = num_runs
                    for r in test_over_runs:
                        num_runs = r
                        test_acc = test(test_loader)
                        test_accs.append(test_acc)
                    num_runs = old_num_runs
            elif multiple_tests:
                for i in range(10):
                    test_acc = test(test_loader)
                    test_accs.append(test_acc)
                test_acc =  torch.tensor(test_accs[-10:]).mean().item()
            else:
                test_acc = test(test_loader)
                test_accs.append(test_acc)
            print('Train Acc: {:.7f}, Test Acc: {:7f}'.format(train_acc, test_acc), flush=True)            
        train_acc = torch.tensor(train_accs)
        test_acc = torch.tensor(test_accs)
        if not test_over_runs is None:
            test_acc = test_acc.view(-1, len(test_over_runs))
        print('---------------- Final Result ----------------')
        print('Train Mean: {:7f}, Train Std: {:7f}, Test Mean: {}, Test Std: {}'.format(train_acc.mean(), train_acc.std(), test_acc.mean(dim=0), test_acc.std(dim=0)), flush=True)
        return test_acc.mean(dim=0), test_acc.std(dim=0)

    train_and_test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='none',
                        help="Options 'none', 'ports', 'ids', 'random', 'dropout', 'smp', 'ppgn', 'ds', 'dss', 'amp'")
    parser.add_argument('--permutation', type=str, default='',
                        help='Permutation for ESAN (ds and dss): node, edge or egonets')
    parser.add_argument('--prob', type=int, default=-1)
    parser.add_argument('--num_runs', type=int, default=50)
    parser.add_argument('--num_layers', type=int, default=4) # 9 layers were used for skipcircles dataset
    parser.add_argument('--use_aux_loss', action='store_true', default=False)

    parser.add_argument('--verbose', action='store_true', default=False)

    parser.add_argument('--prob_ablation', action='store_true', default=False, help="Run probability ablation study")
    parser.add_argument('--num_runs_ablation', action='store_true', default=False, help="Run number of runs ablation study")

    parser.add_argument('--dataset', type=str, default='limitsone',
                        help="Options are ['limitsone', 'limitstwo', 'triangles', 'lcc', "
                             "'maxlimit', 'meanlimit', 'fourcycles', 'skipcircles']")
    args = parser.parse_args()

    main(args)

    print('Finished', flush=True)
