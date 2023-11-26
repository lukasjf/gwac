This is the reference implementation for the paper "Asynchronous Neural Networks for Learning in Graphs"
This paper and the repository contains three sets of experiments. The core of the codebase is copied from
DropGNN: https://github.com/KarolisMart/DropGNN

You can install the required packages with conda from the environment.yml file with the command
`conda env create -f environment.yml`

1. The first set of experiments experiment on synthetic graphs that beyond 1-WL expressiveness. You can run these
datasets with the following command
`python main_synthetic.py --dataset=<dataset> --model=<model>`
dataset can be one of (limitsone, limitstwo, triangles, lcc, maxlimit, meanlimit, fourcycles, skipcircles, rookshrikande)
model can be one of
    * none, ports, ids, random, dropout, (from https://github.com/KarolisMart/DropGNN)
    * smp, ppgn, from https://github.com/cvignac/SMP
    * ds, dss, from https://github.com/beabevi/ESAN . Additionally you need to seed another flag
        --permutation=node|edge|egonets   (this defines the permutation performed on the subgraphs)
    * gwacc, gwacr (for constant and random delays)


2. The second set of experiments targets the long-range information propagation on shortest path parity graphs. They can
be run with the following command:
`python main_longrange.py --dataset=<dataset> --model=<model>`
dataset can be one of (oddeven, multioddeven) --- multioddeven does three executions in parallel
model can be one of gwacs, gwacgru, gwaclstm, gwacatt, gwacact, gwactiter, itergnn, universal, neg
After running all experiments `python plot_longrange` generates all table bodys and plots.


3. The third set of experiments runs models on several graph classification benchmarks with the following command:
`python main_graph_classification.py --dataset=<dataset> --model=<model> --seed=seed`
dataset can be on of (MUTAG, PTC, PROTEINS, IMDB-BINARY, IMDB-MULTI)
A wrapper for GwAC multiprocessing exists in main_graph_classification_mp.py with the same interface
Runs are for one single seed and write their results to result/graph (make sure folder exists) and can be merged with `python merge.py`. For GNNs, we would recommend running all seeds at once (comment out the "continue" in line 381 in the main file) - the final results is available via printing

The third main file main_graph_classification_mp_timing.py contains a subset just for measuring times for different multiprocessing levels.
Results are written to result/timing (ensure folder exists). merge_timing.py creates a plot from results in this folder