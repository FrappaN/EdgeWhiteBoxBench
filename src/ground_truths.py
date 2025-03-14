import networkx as nx
import torch
import torch_geometric.utils as tgu


# def compute_importances_LPModel_OLD(node_ids, graph):
#     node_features = graph.x
#     node_features = (node_features[:, 0]-node_features[:, 1])
#     edge_index = graph.edge_index

#     ground_truth_importances = []

#     initial_labels = node_features[edge_index[0]]
#     first_round_labels = tgu.scatter(initial_labels, edge_index[1], dim=0, reduce='sum', dim_size=graph.num_nodes)
#     first_round_labels[first_round_labels!=0] = first_round_labels[first_round_labels!=0] / torch.abs(first_round_labels[first_round_labels!=0])

#     second_round_labels = tgu.scatter(first_round_labels[edge_index[0]], edge_index[1], dim=0, reduce='sum', dim_size=graph.num_nodes)
#     second_round_labels[second_round_labels!=0] = second_round_labels[second_round_labels!=0] / torch.abs(second_round_labels[second_round_labels!=0])

#     for node_idx in node_ids:
#         _, _, _, edge_mask = tgu.k_hop_subgraph(node_idx, 2, edge_index, relabel_nodes=False)

#         label_node = second_round_labels[node_idx]
#         if label_node == 0:
#             ground_truth_importances.append(torch.zeros(len(edge_index[0]), dtype=torch.bool).cpu().numpy())
#             continue

#         curr_importances = torch.zeros_like(edge_mask, dtype=torch.bool)
#         curr_difficult_negatives = curr_importances.clone()
#         curr_negatives = curr_importances.clone()

#         first_order_importances = (first_round_labels[edge_index[0]][edge_mask] == label_node) & (edge_index[1][edge_mask]==node_idx)
#         first_order_negatives = (first_round_labels[edge_index[0]][edge_mask] != label_node) & (edge_index[1][edge_mask]==node_idx)

#         neighs = edge_index[0][edge_mask][first_order_importances]
#         negative_neighs = edge_index[0][edge_mask][first_order_negatives]

#         second_order_mask = torch.tensor(torch.isin(edge_index[1][edge_mask], neighs), dtype=torch.bool)
#         negative_second_order_mask = torch.tensor(torch.isin(edge_index[1][edge_mask], negative_neighs), dtype=torch.bool)

#         second_order_importances = ((initial_labels == label_node)[edge_mask] & second_order_mask)
#         difficult_second_negatives = ((initial_labels == label_node)[edge_mask] & negative_second_order_mask)

#         curr_importances[edge_mask] = first_order_importances | second_order_importances
#         curr_negatives[edge_mask] = ~curr_importances[edge_mask]
#         curr_difficult_negatives[edge_mask] = difficult_second_negatives

#         curr_gt_dict = {
#             'positive': curr_importances.cpu().numpy(),
#             'negative': curr_negatives.cpu().numpy(),
#             'difficult_negative': curr_difficult_negatives.cpu().numpy()
#         }

#         ground_truth_importances.append(curr_gt_dict)

#     return ground_truth_importances


def compute_importances_LPModel(node_ids, graph):
    node_features = graph.x
    num_classes = node_features.shape[1]
    #node_features = torch.argmax(node_features, dim=1)
    edge_index = graph.edge_index
    weights = torch.ones(num_classes, num_classes, device=node_features.device) * (-1)
    weights.fill_diagonal_(1)
    epsilon = 1e-15

    ground_truth_importances = []

    initial_labels = torch.argmax(node_features, dim=1)[edge_index[0]]
    initial_labels[node_features[edge_index[0]].sum(dim=1) == 0] = -1
    first_layer_features = tgu.scatter(node_features[edge_index[0]], edge_index[1], dim=0, reduce='sum', dim_size=graph.num_nodes)
    # turn everything to 1 where first_layer_labels row has a max value or 0 otherwise
    first_layer_features = first_layer_features@weights
    first_layer_features[first_layer_features<0] = 0
    first_layer_features = first_layer_features.div(first_layer_features.norm(dim=1, keepdim=True).detach()+epsilon)
    # set to 0 all rows that have sum > 1

    first_layer_labels = torch.argmax(first_layer_features, dim=1)
    first_layer_labels[first_layer_features.sum(dim=1) == 0] = -1

    second_layer_features = tgu.scatter(first_layer_features[edge_index[0]], edge_index[1], dim=0, reduce='sum', dim_size=graph.num_nodes)
    second_layer_features = second_layer_features@weights
    second_layer_features[second_layer_features<0] = 0
    second_layer_features = second_layer_features.div(second_layer_features.norm(dim=1, keepdim=True).detach()+epsilon)
    
    second_layer_labels = torch.argmax(second_layer_features, dim=1)
    second_layer_labels[second_layer_features.sum(dim=1) == 0] = -1

    
    for node_idx in node_ids:
        _, _, _, edge_mask = tgu.k_hop_subgraph(node_idx, 2, edge_index, relabel_nodes=False)

        label_node = second_layer_labels[node_idx]

        curr_importances = torch.zeros_like(edge_mask, dtype=torch.bool)
        curr_difficult_negatives = curr_importances.clone()
        curr_negatives = curr_importances.clone()

        first_order_importances = (first_layer_labels[edge_index[0]][edge_mask] == label_node) & (edge_index[1][edge_mask]==node_idx)
        first_order_negatives = (first_layer_labels[edge_index[0]][edge_mask] != label_node) & (edge_index[1][edge_mask]==node_idx)

        neighs = edge_index[0][edge_mask][first_order_importances]
        negative_neighs = edge_index[0][edge_mask][first_order_negatives]

        second_order_mask = torch.tensor(torch.isin(edge_index[1][edge_mask], neighs), dtype=torch.bool)
        negative_second_order_mask = torch.tensor(torch.isin(edge_index[1][edge_mask], negative_neighs), dtype=torch.bool)

        second_order_importances = ((initial_labels == label_node)[edge_mask] & second_order_mask)
        difficult_second_negatives = ((initial_labels == label_node)[edge_mask] & negative_second_order_mask)

        curr_importances[edge_mask] = first_order_importances | second_order_importances
        curr_negatives[edge_mask] = ~curr_importances[edge_mask]
        curr_difficult_negatives[edge_mask] = difficult_second_negatives

        curr_gt_dict = {
            'mask': edge_mask.cpu().numpy(),
            'positive': curr_importances.cpu().numpy(),
            'negative': curr_negatives.cpu().numpy(),
            'difficult_negative': curr_difficult_negatives.cpu().numpy()
        }

        ground_truth_importances.append(curr_gt_dict)

    return ground_truth_importances


def compute_Z_importances_LPModel(node_ids, graph):
    node_features = graph.x
    node_features = node_features[:, 0]-node_features[:, 1]
    edge_index = graph.edge_index

    ground_truth_importances = []


    initial_labels = node_features[edge_index[0]]
    first_round_labels = tgu.scatter(initial_labels, edge_index[1], dim=0, reduce='sum', dim_size=graph.num_nodes)
    first_round_labels[first_round_labels!=0] = first_round_labels[first_round_labels!=0] / torch.abs(first_round_labels[first_round_labels!=0])

    second_round_labels = tgu.scatter(first_round_labels[edge_index[0]], edge_index[1], dim=0, reduce='sum', dim_size=graph.num_nodes)
    second_round_labels[second_round_labels!=0] = second_round_labels[second_round_labels!=0] / torch.abs(second_round_labels[second_round_labels!=0])

    for node_idx in node_ids:
        _, _, _, edge_mask = tgu.k_hop_subgraph(node_idx, 2, edge_index, relabel_nodes=False)

        label_node = second_round_labels[node_idx]
        if label_node == 0:
            ground_truth_importances.append(torch.zeros(len(edge_index[0]), dtype=torch.bool).cpu().numpy())
            continue

        curr_importances = edge_mask.clone().to(torch.long)
        first_order_importances = (first_round_labels[edge_index[0]][edge_mask] == label_node) & (edge_index[1][edge_mask]==node_idx)
        negative_first_order_importances = (first_round_labels[edge_index[0]][edge_mask] == -label_node) & (edge_index[1][edge_mask]==node_idx)

        neighs = edge_index[0][edge_mask][first_order_importances]
        negative_neighs = edge_index[0][edge_mask][negative_first_order_importances]

        second_order_mask = torch.tensor(torch.isin(edge_index[1][edge_mask], neighs), dtype=torch.bool)
        negative_second_order_mask = torch.tensor(torch.isin(edge_index[1][edge_mask], negative_neighs), dtype=torch.bool)

        second_order_importances = ((initial_labels == label_node)[edge_mask] & second_order_mask)
        negative_second_order_importances = ((initial_labels == -label_node)[edge_mask] & negative_second_order_mask)

        curr_importances[edge_mask] = (first_order_importances | second_order_importances).to(torch.long) - (negative_first_order_importances | negative_second_order_importances).to(torch.long)
        ground_truth_importances.append(curr_importances.cpu().numpy())

    return ground_truth_importances



def compute_importances_UnbalancedLPModel(node_ids, graph, weights):
    node_features = graph.x
    node_features = node_features[:, 0]-node_features[:, 1]
    edge_index = graph.edge_index

    ground_truth_importances = []


    initial_labels = node_features[edge_index[0]]
    initial_labels_scaled = initial_labels.clone()
    # multiply the positive class by weights[0] and the negative class by weights[1]
    initial_labels_scaled[initial_labels > 0] = initial_labels[initial_labels > 0] * weights[0]
    initial_labels_scaled[initial_labels < 0] = initial_labels[initial_labels < 0] * weights[1]

    first_round_labels = tgu.scatter(initial_labels_scaled, edge_index[1], dim=0, reduce='sum', dim_size=graph.num_nodes)
    first_round_labels[first_round_labels!=0] = first_round_labels[first_round_labels!=0] / torch.abs(first_round_labels[first_round_labels!=0])
    

    first_round_labels_scaled = first_round_labels.clone()
    # multiply the positive class by weights[0] and the negative class by weights[1]
    first_round_labels_scaled[first_round_labels > 0] = first_round_labels[first_round_labels > 0] * weights[0]
    first_round_labels_scaled[first_round_labels < 0] = first_round_labels[first_round_labels < 0] * weights[1]

    second_round_labels = tgu.scatter(first_round_labels_scaled[edge_index[0]], edge_index[1], dim=0, reduce='sum', dim_size=graph.num_nodes)
    second_round_labels[second_round_labels!=0] = second_round_labels[second_round_labels!=0] / torch.abs(second_round_labels[second_round_labels!=0])
    
    for node_idx in node_ids:
        _, _, _, edge_mask = tgu.k_hop_subgraph(node_idx, 2, edge_index, relabel_nodes=False)

        label_node = second_round_labels[node_idx]
        if label_node == 0:
            ground_truth_importances.append(torch.zeros(len(edge_index[0]), dtype=torch.bool).cpu().numpy())
            continue

        curr_importances = torch.zeros_like(edge_mask, dtype=torch.bool)
        curr_difficult_negatives = curr_importances.clone()
        curr_negatives = curr_importances.clone()

        first_order_importances = (first_round_labels[edge_index[0]][edge_mask] == label_node) & (edge_index[1][edge_mask]==node_idx)
        first_order_negatives = (first_round_labels[edge_index[0]][edge_mask] != label_node) & (edge_index[1][edge_mask]==node_idx)

        neighs = edge_index[0][edge_mask][first_order_importances]
        negative_neighs = edge_index[0][edge_mask][first_order_negatives]

        second_order_mask = torch.tensor(torch.isin(edge_index[1][edge_mask], neighs), dtype=torch.bool)
        negative_second_order_mask = torch.tensor(torch.isin(edge_index[1][edge_mask], negative_neighs), dtype=torch.bool)

        second_order_importances = ((initial_labels == label_node)[edge_mask] & second_order_mask)
        difficult_second_negatives = ((initial_labels == label_node)[edge_mask] & negative_second_order_mask)

        curr_importances[edge_mask] = first_order_importances | second_order_importances
        curr_negatives[edge_mask] = ~curr_importances[edge_mask]
        curr_difficult_negatives[edge_mask] = difficult_second_negatives

        curr_gt_dict = {
            'positive': curr_importances.cpu().numpy(),
            'negative': curr_negatives.cpu().numpy(),
            'difficult_negative': curr_difficult_negatives.cpu().numpy()
        }

        ground_truth_importances.append(curr_gt_dict)

    return ground_truth_importances


def compute_Z_importances_UnbalancedLPModel(node_ids, graph, weights):
    node_features = graph.x
    node_features = node_features[:, 0]-node_features[:, 1]
    edge_index = graph.edge_index

    ground_truth_importances = []


    initial_labels = node_features[edge_index[0]]
    initial_labels_scaled = initial_labels.clone()
    # multiply the positive class by weights[0] and the negative class by weights[1]
    initial_labels_scaled[initial_labels > 0] = initial_labels[initial_labels > 0] * weights[0]
    initial_labels_scaled[initial_labels < 0] = initial_labels[initial_labels < 0] * weights[1]

    first_round_labels = tgu.scatter(initial_labels_scaled, edge_index[1], dim=0, reduce='sum', dim_size=graph.num_nodes)
    first_round_labels[first_round_labels!=0] = first_round_labels[first_round_labels!=0] / torch.abs(first_round_labels[first_round_labels!=0])
    

    first_round_labels_scaled = first_round_labels.clone()
    # multiply the positive class by weights[0] and the negative class by weights[1]
    first_round_labels_scaled[first_round_labels > 0] = first_round_labels[first_round_labels > 0] * weights[0]
    first_round_labels_scaled[first_round_labels < 0] = first_round_labels[first_round_labels < 0] * weights[1]

    second_round_labels = tgu.scatter(first_round_labels_scaled[edge_index[0]], edge_index[1], dim=0, reduce='sum', dim_size=graph.num_nodes)
    second_round_labels[second_round_labels!=0] = second_round_labels[second_round_labels!=0] / torch.abs(second_round_labels[second_round_labels!=0])
    for node_idx in node_ids:
        _, _, _, edge_mask = tgu.k_hop_subgraph(node_idx, 2, edge_index, relabel_nodes=False)

        label_node = second_round_labels[node_idx]
        if label_node == 0:
            ground_truth_importances.append(torch.zeros(len(edge_index[0]), dtype=torch.bool).cpu().numpy())
            continue

        curr_importances = edge_mask.clone().to(torch.long)
        first_order_importances = (first_round_labels[edge_index[0]][edge_mask] == label_node) & (edge_index[1][edge_mask]==node_idx)
        negative_first_order_importances = (first_round_labels[edge_index[0]][edge_mask] == -label_node) & (edge_index[1][edge_mask]==node_idx)

        neighs = edge_index[0][edge_mask][first_order_importances]
        negative_neighs = edge_index[0][edge_mask][negative_first_order_importances]

        second_order_mask = torch.tensor(torch.isin(edge_index[1][edge_mask], neighs), dtype=torch.bool)
        negative_second_order_mask = torch.tensor(torch.isin(edge_index[1][edge_mask], negative_neighs), dtype=torch.bool)

        second_order_importances = ((initial_labels == label_node)[edge_mask] & second_order_mask)
        negative_second_order_importances = ((initial_labels == -label_node)[edge_mask] & negative_second_order_mask)

        curr_importances[edge_mask] = (first_order_importances | second_order_importances).to(torch.long) - (negative_first_order_importances | negative_second_order_importances).to(torch.long)

        ground_truth_importances.append(curr_importances.cpu().numpy())


    return ground_truth_importances
            

def compute_importances_OneLabelLPModel(node_ids, graph, n_layers):
    node_features = graph.x
    edge_index = graph.edge_index

    ground_truth_importances = []
    
    for node_idx in node_ids:

        initial_labels = node_features[edge_index[0]].squeeze()
        node_neighs_mask = edge_index[1] == node_idx
        node_neighs = edge_index[0, node_neighs_mask]
        if n_layers == 1:
            important_edges = (initial_labels == 1) & (node_neighs_mask)
        else:

            second_neighs_mask = torch.isin(edge_index[1], node_neighs)
            first_layer_labels = tgu.scatter(initial_labels, edge_index[1], dim=0, reduce='sum')

            important_edges = ((initial_labels == 1) & (second_neighs_mask)) | ((first_layer_labels[edge_index[0]] > 0) & node_neighs_mask)

        curr_importances = important_edges.squeeze()

        ground_truth_importances.append(curr_importances.cpu().numpy())


    return ground_truth_importances

def compute_importances_ECModel(node_ids, graph):
    node_features = graph.x
    edge_index = graph.edge_index

    ground_truth_importances = []

    for node_idx in node_ids:
        importances = torch.zeros(len(edge_index[0]))
        G = tgu.to_networkx(graph, to_undirected=False)
        relevant_nodes = torch.where(node_features[:, 0] == 1)[0]
        neigh, _, _, _ = tgu.k_hop_subgraph(node_idx, 2, edge_index, relabel_nodes=False)

        relevant_nodes = set(relevant_nodes.tolist())
        neigh = set(neigh.tolist())

        relevant_nodes = relevant_nodes.intersection(neigh)

        if node_idx in relevant_nodes:
            relevant_nodes.remove(node_idx)
            cycles = nx.simple_cycles(G, 2)
            for cycle in cycles:
                if len(cycle)==2 and node_idx in cycle:
                    for k in range(len(cycle)):
                        edge = (cycle[k], cycle[(k+1)%2])
                        idx = torch.where((edge_index[0] == edge[0]) & (edge_index[1] == edge[1]))[0]
                        importances[idx] += 1


        for curr_relevant_node in relevant_nodes:
            paths = nx.all_simple_paths(G, source=curr_relevant_node, target=node_idx, cutoff=2)
            for path in paths:
                if len(path) == 3:
                    for k in range(len(path)-1):
                        edge = (path[k], path[k+1])
                        idx = torch.where((edge_index[0] == edge[0]) & (edge_index[1] == edge[1]))[0]
                        importances[idx] += 1
        ground_truth_importances.append(importances > 0)
    return ground_truth_importances
