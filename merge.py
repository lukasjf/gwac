from os import listdir
from collections import defaultdict
from ast import literal_eval
import torch
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.size"] = 14


results = defaultdict(list)

for f in listdir('result/graph/'):
    experiment = f[:-2]
    print(f)
    with open("result/graph/" + f, "r") as res:
        line = res.readline()
        print(line)
        if not line:
            continue
        results[experiment].append(literal_eval(line))

for dataset in sorted(results.keys()):
    tensors = [torch.tensor(r) for r in results[dataset]]
    combined = torch.stack(tensors, dim=0)
    best = torch.max(combined, dim=1)
    acc_mean = combined.mean(dim=0)
    best_epoch = acc_mean.argmax().item()
    print(dataset, 'Mean: {:1f}, Std: {:1f}'.format(combined[:,best_epoch].mean()*100,
                                           combined[:,best_epoch].std()*100), flush=True)

