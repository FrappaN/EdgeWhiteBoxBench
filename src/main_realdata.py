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

from collections import namedtuple

import warnings

warnings.filterwarnings("ignore")

import argparse

from experiment import run_experiment


def main():
    # take two arguments as input: homophily and labelled_fraction
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--homophily', '-h', type=float, default=0.33,)
    parser.add_argument('--labelled_fraction', '-lf', type=float, default=0.8)
    args = parser.parse_args()

    homophily = args.homophily
    labelled_fraction = args.labelled_fraction

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Params = namedtuple("Params", "dataset_type, num_nodes,  p_edge, homophily, labelled_fraction, num_classes")

    params = [
        Params("Cora", None, None, None, labelled_fraction, 7),
        Params("PubMed", None, None, None, labelled_fraction, 3),
        Params("Arxiv", None, None, None, labelled_fraction, 40)
    ]

    params_seed = [(p, 3030 + seed) for seed in range(1) for p in params]

    WHITEBOX_TYPES = [
       models.GeneralLPModel#, models.ECModel
    ]


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
    df_full_results.to_csv(f'results/full/real_datasets__lf_{labelled_fraction:.2f}.csv')

    return

if __name__=='__main__':
    main()


