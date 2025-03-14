from torch_geometric.datasets import Planetoid
from torch_geometric.datasets.graph_generator import ERGraph
from torch_geometric.loader import DataLoader
import torch_geometric.utils as tgu
from torch_geometric.transforms import to_undirected
from torch_geometric import seed_everything
import networkx as nx
import torch
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset


class DatasetLoader:
    def __init__(self, seed) -> None:
        seed_everything(seed)
        pass

    def generate(self):
        return NotImplementedError()


class ERGraphDataset(DatasetLoader):
    def __init__(self, seed, num_nodes, p) -> None:
        super().__init__(seed)
        self._num_nodes = num_nodes
        self._p = p
        self._graph_generator = ERGraph(num_nodes=self._num_nodes, edge_prob=self._p)
        return

    def generate(self, model_name, labelled_fraction=0.4, homophily=0.5):
        
        graph = self._graph_generator()

        if model_name == 'LPModel' or model_name == 'GeneralLPModel':
            x = torch.zeros(graph.num_nodes, 2)

            # randomly assign two communities to  labelled_nodes*nodes nodes
            labelled_nodes = int(labelled_fraction*graph.num_nodes)
            labelled_nodes = np.random.choice(graph.num_nodes, labelled_nodes, replace=False)
            commA_nodes = labelled_nodes[:len(labelled_nodes)//2]
            x[commA_nodes, 0] = 1
            commB_nodes = labelled_nodes[len(labelled_nodes)//2:]
            x[commB_nodes, 1] = 1

            # rewire a fraction of edges according to homophily
            edge_index = graph.edge_index

            commA_nodes = torch.tensor(commA_nodes)
            commB_nodes = torch.tensor(commB_nodes)

            hetero_edges_A_to_B_mask = torch.isin(edge_index[0], commA_nodes) & torch.isin(edge_index[1], commB_nodes)
            hetero_edges_B_to_A_mask = torch.isin(edge_index[0], commB_nodes) & torch.isin(edge_index[1], commA_nodes)
            hetero_edges_A_to_B = edge_index[:, hetero_edges_A_to_B_mask]
            hetero_edges_B_to_A = hetero_edges_A_to_B.flip(0)

            num_rewire = int(homophily*len(hetero_edges_A_to_B[0]))
            rewire_edges_A_to_B = np.random.choice(len(hetero_edges_A_to_B[0]), num_rewire, replace=False)
            destination_nodes_A = np.random.choice(commA_nodes, num_rewire, replace=True)
            destination_nodes_B = np.random.choice(commB_nodes, num_rewire, replace=True)
            hetero_edges_A_to_B[1, rewire_edges_A_to_B] = torch.tensor(destination_nodes_A)
            hetero_edges_B_to_A[1, rewire_edges_A_to_B] = torch.tensor(destination_nodes_B)

            edge_index[:, hetero_edges_A_to_B_mask] = hetero_edges_A_to_B
            edge_index[:, hetero_edges_B_to_A_mask] = hetero_edges_B_to_A

            graph.edge_index = tgu.to_undirected(tgu.coalesce(edge_index))

            graph.edge_index = tgu.remove_self_loops(graph.edge_index)[0]

            graph.x = x
        elif model_name == 'ECModel':
            x = torch.ones(graph.num_nodes, 1)
            graph.x = x
        elif model_name == 'OneLabelLPModel':
            x = torch.zeros(graph.num_nodes, 1)
            labelled_nodes = int(labelled_fraction*graph.num_nodes)
            labelled_nodes = np.random.choice(graph.num_nodes, labelled_nodes, replace=False)
            x[labelled_nodes] = 1
            graph.x = x

        else:
            raise NotImplementedError

        return graph

class PlanetoidDataset(DatasetLoader):
    def __init__(self, seed, dataset_name) -> None:
        super().__init__(seed)
        dataset = Planetoid(root=f'dataset/{dataset_name}', name=dataset_name)
        self.dataset = dataset[0]
        edge_index = self.dataset.edge_index
        edge_index = tgu.to_undirected(tgu.coalesce(edge_index))
        edge_index = tgu.remove_self_loops(edge_index)[0]
        self.dataset.edge_index = edge_index
        self.num_classes = dataset.num_classes
        return

    def generate(self, model_name, labelled_fraction=0.4):
        data = self.dataset
        if model_name == 'LPModel' or model_name == 'GeneralLPModel':
            x = torch.zeros(data.num_nodes, self.num_classes)

            # pick labelled_fraction of nodes to be labelled
            labelled_nodes = int(labelled_fraction*data.num_nodes)
            labelled_nodes = np.random.choice(data.num_nodes, labelled_nodes, replace=False)
            # assign as x the one-hot encoding of the binary label
            labels = data.y

            x[labelled_nodes, labels[labelled_nodes]] = 1
            assert x.sum(dim=1).max() == 1

            data.x = x

        else:
            raise NotImplementedError

        return data
    
class CoraDataset(PlanetoidDataset):
    def __init__(self, seed) -> None:
        super().__init__(seed, 'Cora')
        # old_y = self.dataset.y
        # new_y = old_y.clone()
        # # set 0 to y==0 and y==3 else 1
        # new_y[old_y == 3] = 0
        # new_y[(old_y != 0) & (old_y != 3)] = 1
        # self.dataset.y = new_y
        return
    

class PubMedDataset(PlanetoidDataset):
    def __init__(self, seed) -> None:
        super().__init__(seed, 'PubMed')
        # old_y = self.dataset.y
        # new_y = old_y.clone()
        # new_y[old_y == 2] = 0
        # new_y[(old_y != 0) & (old_y != 2)] = 1
        # self.dataset.y = new_y
        return
    
class CiteSeerDataset(PlanetoidDataset):
    def __init__(self, seed) -> None:
        super().__init__(seed, 'CiteSeer')
        # old_y = self.dataset.y
        # new_y = old_y.clone()
        # new_y[(old_y == 2) | (old_y == 3)] = 0
        # new_y[(old_y != 2) & (old_y != 3)] = 1
        # self.dataset.y = new_y
        return


# 

class ArxivDataset(DatasetLoader):
    def __init__(self, seed) -> None:
        super().__init__(seed)
        dataset = PygNodePropPredDataset(name='ogbn-arxiv')
        self.dataset = dataset[0]
        # old_y = self.dataset.y
        # new_y = old_y.clone()
        # new_y[old_y < 20] = 0
        # new_y[old_y >= 20] = 1
        # self.dataset.y = new_y.reshape(-1)
        edge_index = self.dataset.edge_index
        edge_index = tgu.to_undirected(tgu.coalesce(edge_index))
        edge_index = tgu.remove_self_loops(edge_index)[0]
        self.dataset.edge_index = edge_index
        self.num_classes = dataset.num_classes
        return
    
    def generate(self, model_name, labelled_fraction=0.4):
        data = self.dataset
        if model_name == 'LPModel' or model_name == 'GeneralLPModel':
            x = torch.zeros(data.num_nodes, self.num_classes)

            # pick labelled_fraction of nodes to be labelled
            labelled_nodes = int(labelled_fraction*data.num_nodes)
            labelled_nodes = np.random.choice(data.num_nodes, labelled_nodes, replace=False)
            # assign as x the one-hot encoding of the binary label
            labels = data.y.reshape(-1)

            x[labelled_nodes, labels[labelled_nodes]] = 1
            assert x.sum(dim=1).max() == 1
            data.x = x

        else:
            raise NotImplementedError

        return data