import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing

from torch import Tensor
from torch_geometric.typing import OptTensor


from torch import Tensor
from torch_geometric.typing import OptTensor


class DummyConv(MessagePassing):

    def __init__(self, hidden_channels):
        super(DummyConv, self).__init__(aggr='add')
        self.reset_parameters()
        self.weight = nn.Parameter(torch.eye(hidden_channels))
        self.in_channels = hidden_channels
        self.out_channels = hidden_channels

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: OptTensor = None) -> Tensor:
        out = self.propagate(edge_index=edge_index, x=x, edge_weight=edge_weight, aggr='add')
        return out@self.weight
    
    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

class DummyConvWithWeights(MessagePassing):

    def __init__(self, weight_matrix, bias = None):
        super(DummyConvWithWeights, self).__init__(aggr='add')
        self.reset_parameters()
        self.weights = nn.Parameter(weight_matrix)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        self.in_channels = weight_matrix.shape[0]
        self.out_channels = weight_matrix.shape[1]

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: OptTensor = None) -> Tensor:
        out = self.propagate(edge_index=edge_index, x=x, edge_weight=edge_weight, aggr='add')
        out = out@self.weights
        if hasattr(self, 'bias'):
            out += self.bias
        return out
    
    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
    

class OneLabelLPModel(torch.nn.Module):
    def __init__(self, n_layers=2, bias=-2, **kwargs) -> None:
        super().__init__()
        self.num_iter = n_layers

        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))
        self.convs = nn.ModuleList()
        self.relus = nn.ModuleList()
        for i in range(self.num_iter):
            conv = DummyConv(1)
            relu = nn.ReLU()
            self.convs.append(conv)
            self.relus.append(relu)
        #self.final_weight = nn.Parameter(torch.tensor([1, -1], dtype=torch.float32))

    def forward(self, x, edge_index):
        epsilon = 1e-15

        for conv, relu in zip(self.convs, self.relus):
            #x = x.div(x.norm(dim=-1, keepdim=True).detach()+epsilon)
            x = conv(x, edge_index)#+x
            x = relu(x)

        #x = (x-x.mean())/x.std()
        x = x.squeeze()
        x = x+self.bias
        #pred = torch.stack([x, -x], dim=1)
        return F.sigmoid(x)

class LPModel(torch.nn.Module):
    def __init__(self, num_iter=2, weight_scale=1, **kwargs) -> None:
        super().__init__()
        self.num_iter = num_iter

        self.convs = nn.ModuleList()
        self.relus = nn.ModuleList()
        self.weight_scale = nn.Parameter(torch.tensor(weight_scale, dtype=torch.float32))
        for i in range(self.num_iter):
            weight_matrix = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32)
            conv = DummyConvWithWeights(weight_matrix*weight_scale)
            relu = nn.ReLU()
            self.convs.append(conv)
            self.relus.append(relu)
        self.final_weight = nn.Parameter(torch.tensor([1, -1], dtype=torch.float32))
        self.num_classes = 2

    def forward(self, x, edge_index):
        epsilon = 1e-15

        for conv, relu in zip(self.convs, self.relus):
            x = x.div(x.norm(dim=1, keepdim=True).detach()+epsilon)
            x = conv(x, edge_index)
            x = relu(x)

        #x = x.div(self.weight_scale)
        
        x = x@self.final_weight
        # pred = torch.stack([x, -x], dim=1)
        return F.sigmoid(x)
    
    def subgraphx_model(self):
            
        def subgraphx_forward(x, edge_index):
            epsilon = 1e-15

            for conv, relu in zip(self.convs, self.relus):
                x = x.div(x.norm(dim=1, keepdim=True).detach()+epsilon)
                x = conv(x, edge_index)
                x = relu(x)
            x = x@self.final_weight
            pred = torch.stack([x, -x], dim=1)
            return pred
        
        sub_model = LPModel(num_iter=self.num_iter)
        sub_model.forward = subgraphx_forward
        return sub_model
    
class GeneralLPModel(torch.nn.Module):
    def __init__(self,  num_classes,  num_iter=2, weight_scale=1, **kwargs) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_iter = num_iter

        self.convs = nn.ModuleList()
        self.relus = nn.ModuleList()
        weight_temp = torch.ones((num_classes, num_classes))*(-1)
        # set the diagonal to 1
        weight_temp.fill_diagonal_(1)

        for i in range(self.num_iter):
            weight_matrix = torch.tensor(weight_temp, dtype=torch.float32)
            conv = DummyConvWithWeights(weight_matrix*weight_scale)
            relu = nn.ReLU()
            self.convs.append(conv)
            self.relus.append(relu)
        self.final_weight = nn.Parameter(torch.tensor(weight_temp, dtype=torch.float32))

    def forward(self, x, edge_index):
        epsilon = 1e-15

        for conv, relu in zip(self.convs, self.relus):
            x = x.div(x.norm(dim=1, keepdim=True).detach()+epsilon)
            x = conv(x, edge_index)
            x = relu(x)

        return F.softmax(x, dim=1)
    
    def subgraphx_model(self):
                
            return self


class UnbalancedLPModel(torch.nn.Module):
    def __init__(self, num_iter=2, weights=(1, 100), **kwargs) -> None:
        super().__init__()
        self.num_iter = num_iter
        self.weights = weights

        self.convs = nn.ModuleList()
        self.relus = nn.ModuleList()
        for i in range(self.num_iter):
            weight_matrix = torch.tensor([[weights[0], -weights[0]], [-weights[1], weights[1]]], dtype=torch.float32) 
            conv = DummyConvWithWeights(weight_matrix)
            relu = nn.ReLU()
            self.convs.append(conv)
            self.relus.append(relu)
        self.final_weight = nn.Parameter(torch.tensor([1, -1], dtype=torch.float32))

    def forward(self, x, edge_index):
        epsilon = 1e-15

        for conv, relu in zip(self.convs, self.relus):
            x = x.div(x.norm(dim=1, keepdim=True).detach()+epsilon)
            x = conv(x, edge_index)
            x = relu(x)

        x = x@self.final_weight

        return F.sigmoid(x)
    


class ECModel(torch.nn.Module):
    def __init__(self, min_centrality=0., *args, **kwargs) -> None:
        super().__init__()
        self.conv1 = DummyConv(1)
        self.conv2 = DummyConv(1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.min_centrality = min_centrality

    def forward(self, x, edge_index):
        #x = x.div(x.norm(dim=None, p=1).detach())
        x = self.conv1(x, edge_index)
        x = self.relu1(x)
        #x = x.div(x.norm(dim=None, p=1).detach())
        x = self.conv2(x, edge_index)
        x = self.relu2(x)
        # output two logits, one with x-self.min_centrality and one with -x+self.min_centrality
        #pred = torch.stack([x-self.min_centrality, -x+self.min_centrality], dim=1)
        return F.sigmoid(x-self.min_centrality)
    

    def subgraphx_model(self):
        sub_model = ECModel(min_centrality=self.min_centrality)

        def subgraphx_forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = self.relu1(x)
            x = self.conv2(x, edge_index)
            x = self.relu2(x)
            return torch.stack([x-self.min_centrality, -x+self.min_centrality], dim=1)
        
        sub_model.forward = subgraphx_forward
        return sub_model
    

class CNModel(torch.nn.Module):
    def __init__(self, num_nodes, *args, **kwargs) -> None:
        super().__init__()
        self.hidden_channels = num_nodes
        self.conv1 = DummyConv(self.hidden_channels)
        self.relu = nn.ReLU()
        # create a diagonal matrix with the elements in nodes as the diagonal
        #self.nodes_matrix = nn.Parameter(M)

    def forward(self, x, edge_index, edge_label_index=None):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        pred = (x.t())@x
        if edge_label_index is not None:
            pred = pred[edge_label_index]
        return F.sigmoid(pred)
