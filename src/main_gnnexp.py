# reduce the number of cores to 
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"



import torch

import pandas as pd
import models as models 
import datasets as datasets 

from explainers import  gnn_explainer
from collections import namedtuple

import warnings

from experiment import run_experiment

warnings.filterwarnings("ignore")

import argparse



def main():
    # take two arguments as input: homophily and labelled_fraction
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--homophily', '-h', type=float, default=0.33,)
    parser.add_argument('--labelled_fraction', '-lf', type=float, default=0.8)
    args = parser.parse_args()

    homophily = args.homophily
    labelled_fraction = args.labelled_fraction

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    Params = namedtuple("Params", "dataset_type, num_nodes,  p_edge, homophily, labelled_fraction")

    params = [
        Params("ER", 1000, 0.005, homophily, labelled_fraction),
        Params("ER", 1000, 0.01, homophily, labelled_fraction),
        Params("ER", 1000, 0.05, homophily, labelled_fraction),

    ]

    params_seed = [(p, 3030 + seed) for seed in range(1) for p in params]

    WHITEBOX_TYPES = [
       models.LPModel#, models.ECModel
    ]

    path = 'results/gnnexplainer_exps/'

    explainers_and_kwargs = (
        (gnn_explainer, {'epochs': 10}),
        (gnn_explainer, {'epochs': 30}),
        (gnn_explainer, {'epochs': 100}),
        (gnn_explainer, {'epochs': 300}),
        (gnn_explainer, {'epochs': 1000}),
        (gnn_explainer, {'epochs': 3000}),
        (gnn_explainer, {'epochs': 10000}),
        (gnn_explainer, {'edge_size': 0.0001}),
        (gnn_explainer, {'edge_size': 0.001}),
        (gnn_explainer, {'edge_size': 0.01}),
        (gnn_explainer, {'edge_size': 0.1}),
        (gnn_explainer, {'edge_size': 1}),

        )
    full_results = []
    for param, seed in params_seed:
        print(param)
        for whitebox_type in WHITEBOX_TYPES:
                whitebox = whitebox_type(**param._asdict())
                print(whitebox.__class__.__name__) 
                results = run_experiment(
                    model=whitebox,
                    seed=seed,
                    device=device,
                    explainers_and_kwargs=explainers_and_kwargs,
                    path=path,
                    **param._asdict(),
                )
                run_results = param._asdict().copy()
                run_results['seed']  = seed
                run_results['whitebox'] = whitebox.__class__.__name__

                for key, value in run_results.items():
                    results[key] = value
                
                # if param.dataset_type == 'ER':
                #     results.to_csv(f'results/partial/{param.dataset_type}__p_{param.p_edge}__h_{param.homophily:.2f}__lf_{param.labelled_fraction:.2f}__{whitebox.__class__.__name__}__s_{seed}.csv')
                # else:
                #     results.to_csv(f'results/partial/{param.dataset_type}__lf_{param.labelled_fraction:.2f}__{whitebox.__class__.__name__}__s_{seed}.csv')
                full_results.append(results.copy())

    df_full_results = pd.concat(full_results)
    df_full_results.to_csv(f'{path}ER__h_{homophily:.2f}__lf_{labelled_fraction:.2f}.csv')

    return

if __name__=='__main__':
    main()
