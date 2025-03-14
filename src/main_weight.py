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

from explainers import ig_explainer, LRP_explainer, deconvolution_explainer
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

    path = 'results/weight_scale_experiment/'
    os.makedirs(path, exist_ok=True)
    os.makedirs(path + 'full/', exist_ok=True)
    os.makedirs(path +'explanations/', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    Params = namedtuple("Params", "dataset_type, num_nodes,  p_edge, homophily, labelled_fraction, num_classes, weight_scale")

    params = [
        Params("Cora", None, None, None, labelled_fraction, 7, 1.0),
        Params("Cora", None, None, None, labelled_fraction, 7, 0.75),
        Params("Cora", None, None, None, labelled_fraction, 7, 0.5),
        Params("Cora", None, None, None, labelled_fraction, 7, 0.25),
        Params("PubMed", None, None, None, labelled_fraction, 3, 1.0),
        Params("PubMed", None, None, None, labelled_fraction, 3, 0.75),
        Params("PubMed", None, None, None, labelled_fraction, 3, 0.5),
        Params("PubMed", None, None, None, labelled_fraction, 3, 0.25),
        Params("Arxiv", None, None, None, labelled_fraction, 40, 1.0),
        Params("Arxiv", None, None, None, labelled_fraction, 40, 0.75),
        Params("Arxiv", None, None, None, labelled_fraction, 40, 0.5),
        Params("Arxiv", None, None, None, labelled_fraction, 40, 0.25),
        Params("ER", 1000, 0.05, homophily, labelled_fraction, 2, 1.0),
        Params("ER", 1000, 0.05, homophily, labelled_fraction, 2, 0.75),
        Params("ER", 1000, 0.05, homophily, labelled_fraction, 2, 0.5),
        Params("ER", 1000, 0.05, homophily, labelled_fraction, 2, 0.25),
    ]

    params_seed = [(p, 3030 + seed) for seed in range(1) for p in params]

    WHITEBOX_TYPES = [
       models.GeneralLPModel, models.LPModel,
    ]
    explainers_and_kwargs = (
            (ig_explainer, {}),
            (ig_explainer, {"post_process": True}),
            (deconvolution_explainer, {}),
            (deconvolution_explainer, {"post_process": True}),
            (LRP_explainer, {}),
            (LRP_explainer, {"post_process": True}),
        )

    full_results = []
    for param, seed in params_seed:
        print(param)
        if param.dataset_type in ['Cora', 'PubMed', 'Arxiv']:
            whitebox_type = WHITEBOX_TYPES[0]
        elif param.dataset_type in ['ER']:
            whitebox_type = WHITEBOX_TYPES[1]
        else:
            raise ValueError(f"Unknown dataset type {param.dataset_type}")
        whitebox = whitebox_type(**param._asdict())
        print(whitebox.__class__.__name__) 
        results = run_experiment(
            model=whitebox,
            seed=seed,
            device=device,
            path=path,
            explainers_and_kwargs=explainers_and_kwargs,
            additional_str=f'__ws_{param.weight_scale}',
            **param._asdict(),
        )
        run_results = param._asdict().copy()
        run_results['seed']  = seed
        run_results['whitebox'] = whitebox.__class__.__name__
        run_results["weight_scale"] = param.weight_scale

        for key, value in run_results.items():
            results[key] = value

        full_results.append(results.copy())

    df_full_results = pd.concat(full_results)
    df_full_results.to_csv(f'{path}full/weight_exp__h_{homophily:.2f}__lf_{labelled_fraction:.2f}.csv')

    return

if __name__=='__main__':
    main()
