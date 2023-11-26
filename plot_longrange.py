from os import listdir
from collections import defaultdict
from ast import literal_eval
import torch
import numpy as np
import matplotlib.pyplot as plt

from labellines import labelLine, labelLines

results = {'oddeven':defaultdict(list), 'multioddeven':defaultdict(list)}

for f in listdir('longrange_results'):
    if not f.startswith('size_wise_acc'):
        continue
    print(f)
    with open('longrange_results/' + f, "r") as res:
        for line in res.readlines():
            model, dataset, seed, size, accuracy, nodes = line.split(',')
            size, accuracy, nodes = int(size), float(accuracy), int(nodes)
            if not dataset in results:
                results[dataset] = {}
            if not model in results[dataset]:
                results[dataset][model] = {}
            if not size in results[dataset][model]:
                results[dataset][model][size] = []
            results[dataset][model][size].append(accuracy)

model_order = ['gin', 'gcn', 'gat', 'neg', 'universal', 'itergnn', 'gwacs', 'gwaciter', 'gwacact']
for dataset in results:
    size_order = sorted(results[dataset][model_order[0]])
    break

print(model_order)
print(size_order)

for model in model_order:
    for dataset in results:
        print(model)
        if not model in results[dataset]:
            continue
        if dataset != "oddeven":
            continue
        accs = [np.mean(results[dataset][model][size]) for size in size_order]
        stds = [np.std(results[dataset][model][size]) for size in size_order]
        accs = [str(acc)[:4] for acc in accs]
        stds = [str(std)[:4] for std in stds]
        accs = ['\makebox{{{}\\rpm{}}}'.format(accs[i], stds[i]) for i in range(len(accs))]
        print(model, dataset)
        print(' & '.join(accs))

"""##############################################################################"""

str_for_model = {"gin": "GIN", "gat": "GAT", "gcn": "GCN", "neg":"NEG", "universal":"UT",
                 "itergnn":"IterGNN", "gwacs":"GwAC-S", "gwacgru":"GwAC-GRU",
                 "gwaclstm":"GwAC-LSTM", "gwacact": "GwAC-UT", "gwaciter": "GwAC-Iter",
                 "gwacatt": "GwAC-ATT"}

results = {}

for f in listdir('longrange_results'):
    if not f.startswith('distance_wise_acc'):
        continue
    with open('longrange_results/' + f, "r") as res:
        for line in res.readlines():
            model, dataset, seed, distance, accuracy, nodes = line.split(',')
            distance, accuracy, nodes = int(distance), float(accuracy), int(nodes)
            if not dataset in results:
                results[dataset] = {}
            if not model in results[dataset]:
                results[dataset][model] = {}
            if not distance in results[dataset][model]:
                results[dataset][model][distance] = []
            results[dataset][model][distance].append(accuracy)

dataset = 'oddeven'
plt.figure(figsize=(9, 4.5))
for m in model_order:
    xs = [d for d in results[dataset][m]]
    xs = xs[1::2]
    xs = [x for x in xs if x < 19]
    print(m, results[dataset][m])
    ys = [np.array(results[dataset][m][d-1]) + np.array(results[dataset][m][d]) for d in xs]
    stds = np.array([np.std(y) for y in ys])
    ys = np.array([np.mean(y)/2 for y in ys])
    xs = xs
    plt.plot(xs, ys, label="{}".format(str_for_model[m]))
plt.xticks([0] + list(range(1, 19, 2)), [0]+["{}-{}".format(i, i+1) for i in range(1, 19,2)])
plt.xlabel("Node distance")
plt.ylabel("Accuracy")
    #plt.fill_between(xs, ys-stds, ys+stds, alpha=0.3)
labelLines(plt.gca().get_lines(), align=True, xvals=[2, 4.5, 1.1, 2., 3.5, 4.5, 7.5, 9.5, 8.5], fontsize=14)
#plt.legend()
plt.savefig('underreaching.pdf')
#plt.show()
plt.clf()

print("###################UNDERREACHING########")

print(xs)
for d in xs:
    numbers= []
    for m in model_order:
        ys = np.array(results[dataset][m][d - 1]) + np.array(results[dataset][m][d])
        std = np.std(ys/4)
        ys = np.mean(ys/2)
        numbers.append('\makebox{{{}\\rpm{}}}'.format(str(ys)[:4], str(std)[:4]))
    print("{}-{} & ".format(d-1, d) + " & ".join(numbers) + "\\\\")



"""##############################################################################"""

results = {}

for f in listdir('longrange_results'):
    if not f.startswith('training_range_accs'):
        continue
    with open('longrange_results/' + f, "r") as res:
        for line in res.readlines():
            model, dataset, seed, size, accuracy = line.split(',')
            size, accuracy, nodes = int(size), float(accuracy), int(nodes)
            if not dataset in results:
                results[dataset] = {}
            if not model in results[dataset]:
                results[dataset][model] = {}
            if not size in results[dataset][model]:
                results[dataset][model][size] = []
            results[dataset][model][size].append(accuracy)

dataset = 'oddeven'
plt.figure(figsize=(9, 4.5))
for m in model_order:
    xs = [size for size in results[dataset][m]]
    xvalues = range(len(xs))
    ys = np.array([np.mean(results[dataset][m][d]) for d in xs])
    stds = np.array([np.std(results[dataset][m][d]) for d in xs])
    plt.plot(xvalues, ys, label="{}".format(str_for_model[m]))
    #plt.fill_between(xvalues, ys -stds, ys + stds,alpha=0.3)
    plt.xticks(xvalues, xs)
plt.xlabel("Graph size")
plt.ylabel("Accuracy")
labelLines(plt.gca().get_lines(), align=True, xvals=[1.6, 1.1, 0.1, 0.5, 2.5, 4.5, 6.2, 4.5, 6, 0.3], fontsize=14)
plt.savefig('oversmoothing.pdf')
#plt.show()
plt.clf()

print("###################OVERSMOOTHING########")

for model in model_order:
    for dataset in results:
        if dataset != "oddeven":
            continue
        print(size_order)
        accs = [np.mean(results[dataset][model][size]) for size in size_order]
        stds = [np.std(results[dataset][model][size]) for size in size_order]
        accs = [str(acc)[:4] for acc in accs]
        stds = [str(std)[:4] for std in stds]
        accs = ['\makebox{{{}\\rpm{}}}'.format(accs[i], stds[i]) for i in range(len(accs))]
        print(model, dataset)
        print(' & '.join(accs))


"""##############################################################################"""