import numpy
import torch
import numpy as np
import pandas as pd
import models as models 
import datasets as datasets 
import ground_truths as gt
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, f1_score
from explainers import  gnn_explainer, ig_explainer, LRP_explainer, deconvolution_explainer, random_explainer #, guided_bp_explainer, SubgraphX_explainer, saliency_explainer,

from tqdm import tqdm

from torch_geometric import seed_everything
import torch_geometric.utils as tgu

def run_experiment(
    dataset_type,
    model,
    num_nodes,
    p_edge,
    homophily,
    labelled_fraction,
    seed=0,
    device='cpu',
    path=None,
    explainers_and_kwargs=None,
    additional_str = '',
    **kwargs
):
    seed_everything(seed)
    if path is None:
        path = 'results/'
    if explainers_and_kwargs is None:
        explainers_and_kwargs = (
            (deconvolution_explainer, {}),
            (deconvolution_explainer, {"post_process": True}),
            (ig_explainer, {}),
            (ig_explainer, {"post_process": True}),
            (LRP_explainer, {}),
            (LRP_explainer, {"post_process": True}),
            (gnn_explainer, {}),
            (gnn_explainer, {"post_process": True, "threshold": 0.5,}),
            (random_explainer, {}),
        )

    model_name = model.__class__.__name__

    if dataset_type == "ER":
        dataset = datasets.ERGraphDataset(
            seed, num_nodes=num_nodes, p=p_edge,
        )
        data = dataset.generate(model_name=model_name, labelled_fraction=labelled_fraction, homophily=homophily)
    elif dataset_type == "Cora":
        dataset = datasets.CoraDataset(seed)
        data = dataset.generate(model_name=model_name, labelled_fraction=labelled_fraction)
    elif dataset_type == "PubMed":
        dataset = datasets.PubMedDataset(seed)
        data = dataset.generate(model_name=model_name, labelled_fraction=labelled_fraction)
    elif dataset_type == "CiteSeer":
        dataset = datasets.CiteSeerDataset(seed)
        data = dataset.generate(model_name=model_name, labelled_fraction=labelled_fraction)
    elif dataset_type == "Arxiv":
        dataset = datasets.ArxivDataset(seed)
        data = dataset.generate(model_name=model_name, labelled_fraction=labelled_fraction)
    else:
        raise NotImplementedError

    data.to(device)
    model.to(device)
    output = model(data.x, data.edge_index)
    if len(output.shape) == 1:
        relevant_node_ids = torch.where(output != 0.5)[0].cpu().tolist()
    else:
        output = torch.where(output == output.max(dim=1, keepdim=True)[0], 1, 0)
        relevant_node_ids = torch.where(output.sum(dim=1) == 1)[0].cpu().tolist()
    if len(relevant_node_ids) > 100:
        # random sample 100 nodes
        relevant_node_ids = np.random.choice(relevant_node_ids, 100, replace=False).tolist()
        
    gt_importances = []

    if model_name == 'LPModel' or model_name == 'GeneralLPModel':
        gt_importances = gt.compute_importances_LPModel(relevant_node_ids, data)
    elif model_name == 'ECModel':
        gt_importances = gt.compute_importances_ECModel(relevant_node_ids, data)
    else:
        raise NotImplementedError
    results = []

    ground_truth_importances = []
    difficult_unimportant_edges = []
    masks = []
    for k in range(len(relevant_node_ids)):
        curr_mask = gt_importances[k]['mask']
        masks.append(curr_mask)
        ground_truth_importances.append(gt_importances[k]['positive'][curr_mask])
        difficult_unimportant_edges.append(gt_importances[k]['difficult_negative'][curr_mask])

    ground_truth_importances = np.array(ground_truth_importances, dtype='object')
    difficult_unimportant_edges = np.array(difficult_unimportant_edges, dtype='object')
    masks = np.array(masks)
    if dataset_type == 'ER':
        np.save(f'{path}explanations/ground_truth__{dataset_type}__p_{p_edge}__h_{homophily:.2f}__lf_{labelled_fraction:.2f}__{model_name}{additional_str}__s_{seed}.npy', ground_truth_importances)
        np.save(f'{path}explanations/difficult_negatives__{dataset_type}__p_{p_edge}__h_{homophily:.2f}__lf_{labelled_fraction:.2f}__{model_name}{additional_str}__s_{seed}.npy', difficult_unimportant_edges)
        np.save(f'{path}explanations/masks__{dataset_type}__p_{p_edge}__h_{homophily:.2f}__lf_{labelled_fraction:.2f}__{model_name}{additional_str}__s_{seed}.npy', masks)
    else:
        np.save(f'{path}explanations/ground_truth__{dataset_type}__lf_{labelled_fraction:.2f}__{model_name}{additional_str}__s_{seed}.npy', ground_truth_importances)
        np.save(f'{path}explanations/difficult_negatives__{dataset_type}__lf_{labelled_fraction:.2f}__{model_name}{additional_str}__s_{seed}.npy', difficult_unimportant_edges)
        np.save(f'{path}explanations/masks__{dataset_type}__lf_{labelled_fraction:.2f}__{model_name}{additional_str}__s_{seed}.npy', masks)

    for explainer, kwargs_explainer in explainers_and_kwargs:

        seed_everything(seed) # a second seed for a reproducible explanation
        explainer_instance = explainer(model, data, **kwargs_explainer)
        explainer_name = str(explainer_instance)
        if explainer_instance.apply_postproc:
            explainer_name = explainer_name + '_postproc'
        print(explainer_name)

        explainer_results =  []
        explanations = []

        for k, node_idx in enumerate(tqdm(relevant_node_ids)):
            explanation_results = {}
            explanation_results["node_idx"] = node_idx
            explanation_results["explainer"] = explainer_name

            curr_mask = gt_importances[k]['mask']

            curr_gt_explanation = gt_importances[k]['positive'][curr_mask]

            curr_gt_neg = gt_importances[k]['negative'][curr_mask]
            curr_gt_difficult_neg = gt_importances[k]['difficult_negative'][curr_mask]

            explanation = explainer_instance.explain(node_idx).detach().cpu().numpy()[curr_mask]
            # if the explanation contains nan values, replace them with 0
            explanation = np.nan_to_num(explanation)
            explanations.append(explanation)

            pr_curve = precision_recall_curve(curr_gt_explanation, explanation)
            explanation_results["pr_auc"] = auc(pr_curve[1], pr_curve[0])

            binarized_exp = np.where(explanation > 0, 1, 0)
            if np.unique(curr_gt_explanation).shape[0] == 1:
                explanation_results["roc_auc"] = None
            else:
                explanation_results["roc_auc"] = roc_auc_score(curr_gt_explanation, explanation)


            
            explanation_results["f1"] = f1_score(curr_gt_explanation, binarized_exp)

            false_negatives = np.sum((binarized_exp == 0) & (curr_gt_explanation))
            explanation_results["false_negative_rate"] = false_negatives / np.sum(curr_gt_explanation)
            
            false_positives = np.sum((binarized_exp == 1) & (curr_gt_neg))
            explanation_results["false_positive_rate"] = false_positives / np.sum(curr_gt_neg)

            difficult_false_positives = np.sum((binarized_exp == 1) & (curr_gt_difficult_neg))
            explanation_results["difficult_false_positives"] = difficult_false_positives / np.sum(curr_gt_difficult_neg)

            precision = np.sum((binarized_exp == 1) & (curr_gt_explanation)) / np.sum(binarized_exp)
            explanation_results["precision"] = precision

            explainer_results.append(explanation_results)

            torch.cuda.empty_cache()
        
        explanations = np.array(explanations, dtype='object')
        # save explanations
        if explainer_name != 'Random':
            if dataset_type == 'ER':
                np.save(f'{path}explanations/{explainer_name}__{dataset_type}__p_{p_edge}__h_{homophily:.2f}__lf_{labelled_fraction:.2f}__{model_name}{additional_str}__s_{seed}.npy', explanations)
            else:
                np.save(f'{path}explanations/{explainer_name}__{dataset_type}__lf_{labelled_fraction:.2f}__{model_name}{additional_str}__s_{seed}.npy', explanations)

        results.extend(explainer_results)


        del explainer_instance
        torch.cuda.empty_cache()
    
    del data
    torch.cuda.empty_cache()

    results = pd.DataFrame(results)
    return results