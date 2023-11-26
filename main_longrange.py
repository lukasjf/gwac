# This implementation is based on https://github.com/weihua916/powerful-gnns and https://github.com/chrsmrrs/k-gnn/tree/master/examples
# Datasets are implemented based on the description in the corresonding papers (see the paper for references)
import argparse
from longrange_datasets import *
import numpy as np
import time
import random
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree
from models.longrange.neural_execution_of_graph_algorithms import NEG
from models.longrange.gin import GIN
from models.longrange.gat import GAT
from models.longrange.gcn import GCN
from models.longrange.gwac_iter import GwACIter
from models.longrange.gwac_act import GwACACT
from models.longrange.gwacs import GwACS
from models.longrange.gwac_gru import GwACGRU
from models.longrange.gwac_lstm import GwACLSTM
from models.longrange.gwac_att import GwACAttention
from models.longrange.universal_transformers import UniversalTransformers
from models.longrange.itergnn import GraphGNNModels, NodeGNNModels
from collections import defaultdict

torch.set_printoptions(profile="full")

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def main(args, cluster=None):
    print(args, flush=True)
    size = 10
    graphs = 100

    if args.dataset == "oddeven":
        dataset = OddEvenTask(num_graphs=graphs, num_nodes=size)
    elif args.dataset == "multioddeven":
        dataset = MultisourceOddeven(num_graphs=graphs, num_nodes=size)
    else:
        1/0  # invalid dataset

    print(dataset.__class__.__name__)

    # Set the sampling probability and number of runs/samples for the DropGIN
    n = dataset.num_nodes
    print(f'Number of nodes: {n}')
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
    use_aux_loss = args.use_aux_loss

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(args.model)
    if args.model == 'gwacs':
        model = GwACS(in_features=num_features, hidden=args.num_hidden,
                      out_features=dataset.num_classes, num_predictions=dataset.num_predictions,
                      message_size=dataset.message_size, graph_class=dataset.graph_class)
    elif args.model == 'gwacgru':
        model = GwACGRU(in_features=num_features, hidden=args.num_hidden,
                        out_features=dataset.num_classes, num_predictions=dataset.num_predictions,
                        message_size=dataset.message_size, graph_class=dataset.graph_class)
    elif args.model == 'gwaclstm':
        model = GwACLSTM(in_features=num_features, hidden=args.num_hidden,
                         out_features=dataset.num_classes, num_predictions=dataset.num_predictions,
                         message_size=dataset.message_size, graph_class=dataset.graph_class)
    elif args.model == 'gwaciter':
        model = GwACIter(in_features=num_features, hidden=args.num_hidden,
                         out_features=dataset.num_classes, num_predictions = dataset.num_predictions,
                         message_size=dataset.message_size, graph_class=dataset.graph_class)
    elif args.model == 'gwacact':
        model = GwACACT(in_features=num_features, hidden=args.num_hidden,
                        out_features=dataset.num_classes, num_predictions=dataset.num_predictions,
                        message_size=dataset.message_size, graph_class=dataset.graph_class)
    elif args.model == "gwacatt":
        model = GwACAttention(in_features=num_features, hidden=args.num_hidden,
                              out_features=dataset.num_classes, num_predictions=dataset.num_predictions,
                              message_size=dataset.message_size, graph_class=dataset.graph_class)
    elif args.model == 'universal':
        model = UniversalTransformers(num_features=num_features, num_hidden=args.num_hidden,
                                      num_predictions=dataset.num_predictions,
                                      num_classes=dataset.num_classes, graph_class=dataset.graph_class).to(device)
    elif args.model == 'itergnn':
        if dataset.graph_class:
            model = GraphGNNModels(in_channel=dataset.num_features, edge_channel=1, hidden_size=args.num_hidden,
                                   out_channel=dataset.num_classes, num_predictions = dataset.num_predictions,
                                   embedding_layer_num=2, architecture_name='IterGNN',
                                   layer_num=50, module_num=1, layer_name='PathGNN', input_feat_flag=True,
                                   homogeneous_flag=1, readout_name='Max', confidence_layer_num=1, head_layer_num=1)
        else:
            model = NodeGNNModels(in_channel=dataset.num_features, edge_channel=1, hidden_size=args.num_hidden,
                                  num_predictions=dataset.num_predictions,
                                  out_channel=dataset.num_classes, embedding_layer_num=2, architecture_name='IterGNN',
                                  layer_num=10, module_num=1, layer_name='PathGNN', input_feat_flag=True,
                                  homogeneous_flag=1, readout_name='Max', confidence_layer_num=1, head_layer_num=1)
    elif args.model == 'neg':
        model = NEG(num_features=num_features, num_hidden=args.num_hidden,
                    num_predictions=dataset.num_predictions,
                    num_classes=dataset.num_classes, graph_class=dataset.graph_class).to(device)
    elif args.model == 'gin':
        model = GIN(num_features=num_features, num_hidden=args.num_hidden,
                    num_predictions=dataset.num_predictions,
                    num_classes=dataset.num_classes, graph_class=dataset.graph_class).to(device)
    elif args.model == 'gat':
        model = GAT(num_features=num_features, num_hidden=args.num_hidden,
                    num_predictions=dataset.num_predictions,
                    num_classes=dataset.num_classes, graph_class=dataset.graph_class).to(device)
    elif args.model == 'gcn':
        model = GCN(num_features=num_features, num_hidden=args.num_hidden,
                    num_predictions=dataset.num_predictions,
                    num_classes=dataset.num_classes, graph_class=dataset.graph_class).to(device)
    else:
        1/0  # invalid model

    def train(epoch, loader, optimizer):
        if epoch % 100 == 99:
            print(epoch)
        model.train()
        loss_all = 0
        n = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            pred, aux_loss = model(data)
            total_loss = 0.0
            for i in range(dataset.num_predictions):
                logs = pred[i]
                loss = F.nll_loss(logs, data.y.long()[:, i])
                n += len(data.y)
                total_loss += loss
            loss = loss + aux_loss
            total_loss.backward()
            loss_all += data.num_graphs * total_loss.item()
            optimizer.step()
        return loss_all / len(loader.dataset)

    def test(loader):
        model.eval()
        n = 0
        with torch.no_grad():
            correct = 0
            acc_by_distance = defaultdict(list)
            for data in loader:
                data = data.to(device)
                pred, aux_logs = model(data)
                for i in range(dataset.num_predictions):
                    logs = pred[i]
                    pred_class = logs.max(1)[1]
                    for p in range(len(pred_class)):
                        acc_by_distance[data.distances[p][i].item()].append(
                            1 if pred_class[p].item() == data.y[p][i].item() else 0)
                    n += len(pred_class)
                    correct += pred_class.eq(data.y[:, i]).sum().item()
                    print(correct, n)
        return correct / n, acc_by_distance

    def train_and_test(multiple_tests=False, test_over_runs=None):
        train_accs = []
        all_test_accs = defaultdict(list)
        print(model.__class__.__name__)
        for seed in range(10):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model.reset_parameters()
            lr = 0.01
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            train_dataset = dataset.makedata()

            train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

            print('---------------- Seed {} ----------------'.format(seed))
            for epoch in range(1, 1001):
                if args.verbose:
                    start = time.time()
                train_loss = train(epoch, train_loader, optimizer)
                if args.verbose:
                    print('Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, Time: {:7f}'.format(
                        epoch, lr, train_loss, time.time() - start), flush=True)
            train_acc, train_accs_by_distance = test(train_loader)
            train_accs.append(train_acc)

            if args.dataset == "oddeven":
                test_dataset = OddEvenTask
            elif args.dataset == "multioddeven":
                test_dataset = MultisourceOddeven
            else:
                1/0  # not a valid dataset

            test_accs = []
            total_acc_per_disc = defaultdict(list)
            total_acc_per_size = defaultdict(list)
            seen_dists = train_accs_by_distance.keys()
            seen_accs_per_size = {}

            for factor in [1, 2.5, 5, 10, 25, 50, 100]:
                scaled_size = int(factor * size)
                test_acc, acc_by_distance = test(DataLoader(test_dataset(10, scaled_size).makedata(), batch_size=10))
                test_accs.append(test_acc)
                all_test_accs[scaled_size].append(test_acc)
                accs_in_training_range = []
                for dist in sorted(acc_by_distance.keys()):
                    total_acc_per_disc[dist].extend(acc_by_distance[dist])
                    total_acc_per_size[scaled_size].extend(acc_by_distance[dist])
                    if dist in seen_dists:
                        accs_in_training_range.append(torch.tensor(acc_by_distance[dist]).float().mean())
                seen_accs_per_size[scaled_size]=torch.stack(accs_in_training_range).mean()

            print('Accuracies per distance')
            with open("longrange_results/distance_wise_acc{}_{}".format(args.dataset, args.model), "a") as f:
                for dist in total_acc_per_disc.keys():
                    r = '{},{},{},{},{},{}\n'.format(args.model, args.dataset, seed, dist,
                                               torch.tensor(total_acc_per_disc[dist]).float().mean(), len(total_acc_per_disc[dist]))
                    f.write(r)
                    print(r, end='')
            print('Accuracies for graph sizes')
            with open("longrange_results/size_wise_acc{}_{}".format(args.dataset, args.model), "a") as f:
                for scale in total_acc_per_size.keys():
                    r = '{},{},{},{},{},{}\n'.format(args.model, args.dataset, seed, scale,
                                                   torch.tensor(total_acc_per_size[scale]).float().mean(),
                                                   len(total_acc_per_size[scale]))
                    f.write(r)
                    print(r, end='')
            print('Accuracy of known distances to training set')
            with open('longrange_results/training_range_accs{}_{}'.format(args.dataset, args.model), "a") as f:
                for scale, acc_in_train_range in seen_accs_per_size.items():
                    r = '{},{},{},{},{}\n'.format(args.model, args.dataset, seed, scale, acc_in_train_range)
                    f.write(r)
                    print(r, end='')
        train_acc = torch.tensor(train_accs)
        test_acc = torch.tensor(test_accs)
        if not test_over_runs is None:
            test_acc = test_acc.view(-1, len(test_over_runs))
        print('---------------- Final Result ----------------')
        print(
            'Train Mean: {:7f}, Train Std: {:7f}, Test Mean: {}, Test Std: {}'.format(train_acc.mean(), train_acc.std(),
                                                                                      test_acc.mean(dim=0),
                                                                                      test_acc.std(dim=0)), flush=True)
        for scale, scores in all_test_accs.items():
            print('{},{},{},{}'.format(args.model, args.dataset, torch.tensor(scores).mean(),
                                       torch.tensor(scores).std()))
        return test_acc.mean(dim=0), test_acc.std(dim=0)

    train_and_test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='none',
                        help="Options are amprnn, ampgru, amplstm, ampact, ampiter, itergnn, universal, neg")
    parser.add_argument('--prob', type=int, default=-1)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_hidden', type=int, default=30)
    parser.add_argument('--use_aux_loss', action='store_true', default=False)

    parser.add_argument('--verbose', action='store_true', default=False)

    parser.add_argument('--prob_ablation', action='store_true', default=False, help="Run probability ablation study")

    parser.add_argument('--dataset', type=str, default='oddeven', help="Options are oddeven, multioddeven")
    args = parser.parse_args()

    main(args)

    print('Finished', flush=True)
