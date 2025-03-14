import numpy as np

from sklearn.metrics import accuracy_score
#from scipy.stats import spearmanr
import torch




def evaluate_exp(data, explanation):
    results = {}

    #spearman_node_wise = []

    true_edge_imp = data.edge_imp
    explainer_edge_imp = torch.zeros(data.num_edges)
    # for k, explanation in enumerate(explanations):

    #     true_edge_imp_node_wise = torch.zeros(data.num_edges)
    #     indx = torch.where(data.edge_index[1] == k)[0]
    #     true_edge_imp_node_wise[indx] = true_edge_imp[indx]
        
    #     true_edge_imp_node_wise = true_edge_imp_node_wise.detach().cpu().numpy()
    #     node_wise_edge_imp = explanation.detach().cpu().numpy()

    #     if np.sum(node_wise_edge_imp)==0:
    #         if np.sum(true_edge_imp_node_wise)==0:
    #             correlation = 1
    #         else:
    #             correlation = 0
    #     else:
    #         correlation = np.abs(spearmanr(true_edge_imp_node_wise, node_wise_edge_imp).statistic)

    #     spearman_node_wise.append(correlation)
    #     explainer_edge_imp += explanation

    # results["avg_spearman_node_wise"] = np.mean(spearman_node_wise)

    true_edge_imp = true_edge_imp.detach().cpu().numpy()
    # explainer_edge_imp = explainer_edge_imp.detach().cpu().numpy()
    explainer_edge_imp = explanations.detach().cpu().numpy()
    

    results["spearman__fractional_importances"] = np.abs(spearmanr(true_edge_imp, explainer_edge_imp).statistic)

    true_edge_imp_integer = np.zeros(len(true_edge_imp))
    true_edge_imp_integer[true_edge_imp>0] = 1
    true_edge_imp_integer[true_edge_imp<0] = -1
    explainer_edge_imp_integer = np.zeros(len(explainer_edge_imp))
    explainer_edge_imp_integer[explainer_edge_imp>0] = 1
    explainer_edge_imp_integer[explainer_edge_imp<0] = -1

    results["f1__integer_importances"] = f1_score(true_edge_imp_integer, explainer_edge_imp_integer, average='micro')
    results['spearman__integer_importances'] = np.abs(spearmanr(true_edge_imp_integer, explainer_edge_imp_integer).statistic)   

    return results