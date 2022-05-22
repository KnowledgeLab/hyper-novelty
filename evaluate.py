import random
from functools import partial
from collections import Counter
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd


def get_hyperedges(G):
    # G: a edge-by-node incidence matrix
    f = partial(get_nodes,G=G)
    with Pool(cpu_count()-1) as pool:
        E=Counter(pool.map(f, (G.sum(axis=1).A.flatten()>1).nonzero()[0]))
    return E

def get_nodes(x,G):
    return tuple(sorted(G[x,:].nonzero()[1]))

def ph(h,theta):
    # Propensity of a group of nodes (h)
    return max(theta[h,:].prod(axis=0).sum(), 1e-8)

def estimate_auc(theta, E, N, iterations=10, batchsize=1000):
    auc=np.zeros(iterations)
    batchsize2=min(batchsize,len(E))
    for i in range(iterations):
        E0=random.sample(E.keys(),batchsize2)
        l_pos=np.zeros(batchsize2)
        l_neg=np.zeros(batchsize2)
        for j,h in enumerate(E0):
            l_pos[j]=ph(h,theta)
            while True:
                e=tuple(sorted(random.sample(range(N),len(h))))
                if e not in E:
                    break
            l_neg[j]=ph(e,theta)
        auc[i] = (l_pos>=l_neg).sum()/batchsize2
    return (auc.mean(), auc.std())

def evaluate(G, edge_time, thetas):
    time=[]
    num_nodes=[]
    num_edges=[]
    mean_degree=[]
    mean_edge_size=[]
    max_edge_size=[]
    repeated_edges=[]
    prev_E=None
    AUC=[]
    std_AUC=[]
    for i,t in enumerate(np.unique(edge_time)[:-1]):
        print("Evaluating time {}...".format(t+1))
        # Get hyperedges at t
        G0=G[(edge_time==t+1).nonzero()[0],:]
        E0=get_hyperedges(G0)
        if len(E0)==0:
            print("No data found.")
            continue
        # Basic statistics
        degree=G0.sum(axis=0).A.squeeze()
        degree=degree[degree>0]
        time.append(t+1)
        num_nodes.append(len(degree))
        num_edges.append(len(E0))
        mean_edge_size.append(np.mean([len(h) for h in E0]))
        max_edge_size.append(max(len(h) for h in E0))
        mean_degree.append(degree.mean())
        # Calculate AUC
        auc,std=estimate_auc(thetas[i], 
                             E0, 
                             G0.shape[1],
                             iterations=10, 
                             batchsize=1000)
        AUC.append(auc)
        std_AUC.append(std)
        # Count repeated edges from previous years
        if not prev_E:
            G0=G[(edge_time==t).nonzero()[0],:]
            prev_E=set(list(get_hyperedges(G0).keys()))
        curr_E = set(list(E0.keys()))
        repeated_edges.append(len(curr_E & prev_E))
        prev_E = prev_E | curr_E

    res = pd.DataFrame({"time": time,
                        "num_nodes": num_nodes,
                        "num_edges": num_edges,
                        "mean_edge_size": mean_edge_size,
                        "max_edge_size": max_edge_size,
                        "mean_degree": mean_degree,
                        "auc": AUC,
                        "auc_std": std_AUC,
                        "repeated_edges": repeated_edges})
    return res

