from os import listdir
from collections import defaultdict
from ast import literal_eval
import torch
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.size"] = 14


results = defaultdict(list)
new_or_old = True

datasets = ["PROTEINS", "MUTAG", "GINPTC", "PTC", "IMDB-BINARY", "IMDB-MULTI"]
num_processes = list(range(1,15))
times = {dataset:{i:[] for i in num_processes} for dataset in datasets}

for f in listdir('result/timing/'):
    if not f.startswith("timing"):
        continue
    for line in open("result/timing/" + f, "r").readlines():
        dataset, processes, seconds = line.split(',')
        if int(processes) > 13:
            continue
        times[dataset][int(processes)].append(float(seconds))

plt.figure(figsize=(6,3))
for ds in datasets:
    if ds == "PTC":# or ds == "PROTEINS":
        continue
    print(ds, [np.mean(times[ds][i]) for i in num_processes])
    plt.plot(num_processes, [np.mean(times[ds][i]) for i in num_processes], label=ds)
plt.legend()
#plt.yscale('log')
plt.xlabel("Processes")
plt.ylabel("Time(s) per Epoch")
plt.tight_layout()
plt.savefig("timing.pdf")
plt.show()
