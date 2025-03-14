from torch_geometric.explain import Explainer, ModelConfig 
from torch_geometric.explain.algorithm import GNNExplainer, CaptumExplainer, PGExplainer, GraphMaskExplainer
import numpy as np
from models import DummyConv, DummyConvWithWeights
import torch_geometric as tg
from graphxai.explainers.subgraphx import SubgraphX
from torch_geometric import utils as tgu
from torch_geometric.nn import MessagePassing
import torch


def count_num_hops(model):
    k = 0
    for module in model.modules():
        if isinstance(module, MessagePassing):
            k += 1
    return k


class explainer_class():
    def __init__(self, model, data, post_process=False, threshold=0., **kwargs) -> None:
        self.model = model
        self.data = data
        self.apply_postproc = post_process
        self.threshold = threshold
        return
    
    def explain(self, node_idx):
        explanation = self.explain_node(node_idx)
        if self.apply_postproc:
            explanation = self.post_process(explanation, node_idx)
        return explanation.detach()
    
    def post_process(self, explanation, node_idx):

        model = self.model
        data = self.data
        threshold = self.threshold
        num_hops = count_num_hops(model)


        new_explanation = explanation.clone()

        row, col = data.edge_index
        num_nodes = data.num_nodes

        node_mask = row.new_empty(num_nodes, dtype=torch.bool)
        edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
        not_imp_node_mask = row.new_empty(num_nodes, dtype=torch.bool)

        if isinstance(node_idx, int):
            node_idx = torch.tensor([node_idx], device=row.device)
        elif isinstance(node_idx, (list, tuple)):
            node_idx = torch.tensor(node_idx, device=row.device)
        else:
            node_idx = node_idx.to(row.device)
        subsets = [node_idx]

        not_imp_node_mask.fill_(True)
        not_imp_node_mask[node_idx] = False

        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, col, out=edge_mask)

            incoherent_edge_mask = edge_mask & (explanation > threshold) & not_imp_node_mask[col]
            
            new_explanation[incoherent_edge_mask] = threshold

            not_imp_node_mask = tgu.scatter(new_explanation[edge_mask], row[edge_mask], dim_size=num_nodes, dim=0, reduce='max') <= threshold

            subsets.append(row[edge_mask])

        return new_explanation
    
    def explain_node(self, node_idx):
        raise NotImplementedError
    
    def __str__(self) -> str:
        return 'Explainer'


class saliency_explainer(explainer_class):
    def __init__(self, model, data, **kwargs) -> None:
        super().__init__(model, data, **kwargs)
        num_classes = self.model.num_classes
        if num_classes == 2:
            task = 'binary_classification'
        else:
            task = 'multiclass_classification'
        self.model_config = ModelConfig(mode=task, task_level='node', return_type='probs')
        return
        
    def explain_node(self, node_idx):
        explainer = Explainer(
            model=self.model,
            algorithm=CaptumExplainer(attribution_method='Saliency'),
            explanation_type='model',
            node_mask_type=None,
            edge_mask_type='object',
            model_config = self.model_config,
        )

        explanation = explainer(self.data.x, self.data.edge_index, index=node_idx)
        return explanation.edge_mask
    
    def __str__(self) -> str:
        return 'Saliency'
    

class deconvolution_explainer(explainer_class):
    def __init__(self, model, data, **kwargs) -> None:
        super().__init__(model, data, **kwargs)
        num_classes = self.model.num_classes
        if num_classes == 2:
            task = 'binary_classification'
        else:
            task = 'multiclass_classification'
        self.model_config = ModelConfig(mode=task, task_level='node', return_type='probs')
        return
        
    def explain_node(self, node_idx):
        explainer = Explainer(
            model=self.model,
            algorithm=CaptumExplainer(attribution_method='Deconvolution'),
            explanation_type='model',
            node_mask_type=None,
            edge_mask_type='object',
            model_config = self.model_config,
        )

        explanation = explainer(self.data.x, self.data.edge_index, index=node_idx)
        return explanation.edge_mask
    
    def __str__(self) -> str:
        return 'Deconvolution'
    
class guided_bp_explainer(explainer_class):
    def __init__(self, model, data, **kwargs) -> None:
        super().__init__(model, data, **kwargs)
        num_classes = self.model.num_classes
        if num_classes == 2:
            task = 'binary_classification'
        else:
            task = 'multiclass_classification'
        self.model_config = ModelConfig(mode=task, task_level='node', return_type='probs')
        return
        
    def explain_node(self, node_idx):
        explainer = Explainer(
            model=self.model,
            algorithm=CaptumExplainer(attribution_method='GuidedBackprop'),
            explanation_type='model',
            node_mask_type=None,
            edge_mask_type='object',
            model_config = self.model_config,
        )

        explanation = explainer(self.data.x, self.data.edge_index, index=node_idx)
        return explanation.edge_mask
    
    def __str__(self) -> str:
        return 'GuidedBackprop'
    
class ig_explainer(explainer_class):
    def __init__(self, model, data, **kwargs) -> None:
        super().__init__(model, data, **kwargs)
        num_classes = self.model.num_classes
        if num_classes == 2:
            task = 'binary_classification'
        else:
            task = 'multiclass_classification'
        self.model_config = ModelConfig(mode=task, task_level='node', return_type='probs')
        return
        
        
    def explain_node(self, node_idx):
        explainer = Explainer(
            model=self.model,
            algorithm=CaptumExplainer(attribution_method='IntegratedGradients'),
            explanation_type='model',
            node_mask_type=None,
            edge_mask_type='object',
            model_config = self.model_config,
        )

        explanation = explainer(self.data.x, self.data.edge_index, index=node_idx)
        return explanation.edge_mask
    
    def __str__(self) -> str:
        return 'IntegratedGradients'
    
class gnn_explainer(explainer_class):
    def __init__(self, model, data, **kwargs) -> None:
        super().__init__(model, data, **kwargs)
        num_classes = self.model.num_classes
        if num_classes == 2:
            task = 'binary_classification'
        else:
            task = 'multiclass_classification'
        self.model_config = ModelConfig(mode=task, task_level='node', return_type='probs')
        self.explainer = None
        self.kwargs = kwargs
        return
        
    def explain_node(self, node_idx):
        if self.explainer is None:
            epochs = self.kwargs.pop('epochs', 10000)
            lr = self.kwargs.pop('lr', 0.01)
            edge_size = self.kwargs.pop('edge_size', 0.0001)

            self.explainer = Explainer(
                model=self.model,
                algorithm=GNNExplainer(epochs=epochs, lr=lr, edge_size=edge_size, **self.kwargs),
                explanation_type='model',
                node_mask_type=None,
                edge_mask_type='object',
                model_config = self.model_config,
            )
        explanation = self.explainer(self.data.x, self.data.edge_index, index=node_idx)
        return explanation.edge_mask
     
    def __str__(self) -> str:
        if 'edge_size' in self.kwargs:
            edge_size = self.kwargs['edge_size']
            name = f'GNNExplainer__edge_size_{edge_size}'
            return name
        if 'epochs' in self.kwargs:
            epochs = self.kwargs['epochs']
            name = f'GNNExplainer__epochs_{epochs}'
            return name
        return 'GNNExplainer'
    

from torch_geometric.explain import GraphMaskExplainer

class graphmask_explainer(explainer_class):
    def __init__(self, model, data, **kwargs) -> None:
        super().__init__(model, data, **kwargs)
        num_classes = self.model.num_classes
        if num_classes == 2:
            task = 'binary_classification'
        else:
            task = 'multiclass_classification'
        self.model_config = ModelConfig(mode=task, task_level='node', return_type='probs')
        self.explainer = None
        self.kwargs = kwargs
        return
        
    def explain_node(self, node_idx):
        if self.explainer is None:
            epochs = self.kwargs.pop('epochs', 100)
            lr = self.kwargs.pop('lr', 0.01)

            self.explainer = Explainer(
                model=self.model,
                algorithm=GraphMaskExplainer(num_layers=2, epochs=1000, lr=0.0001, allow_multiple_explanations=True, log=False, **self.kwargs),
                explanation_type='model',
                node_mask_type=None,
                edge_mask_type='object',
                model_config = self.model_config,
            )
        explanation = self.explainer(self.data.x, self.data.edge_index, index=node_idx)
        return explanation.edge_mask
    
    def __str__(self) -> str:
        return 'GraphMask'
    

from captum.attr import LRP
from torch_geometric.explain.algorithm.captum import CaptumModel

class LRP_explainer(explainer_class):
    def __init__(self, model, data, **kwargs) -> None:
        super().__init__(model, data, **kwargs)
        num_classes = self.model.num_classes
        if num_classes == 2:
            task = 'binary_classification'
        else:
            task = 'multiclass_classification'
        self.model_config = ModelConfig(mode=task, task_level='node', return_type='probs')
        self.explainer = self.generate_lrp()
        return
    
    def generate_captum(self, CaptumExplainer, seed=None, attr_kwargs=None, **kwargs):
        attr_kwargs = dict() if attr_kwargs is None else attr_kwargs
        def captum_explainer(model, x, edge_index, node_idx):
            if seed is not None:
                tg.seed_everything(seed)
            captum_model = CaptumModel(model, mask_type='edge', output_idx=node_idx, model_config=self.model_config)
            inputs, additional_forward_args = tg.nn.to_captum_input(x=x, edge_index=edge_index, mask_type='edge')
            exp = CaptumExplainer(captum_model, **kwargs)
            output = model(x, edge_index)[node_idx].reshape(-1)

            if len(output) > 1:
                target = torch.argmax(output).item()
            else:
                target = 1 if output > 0.5 else 0
            exp_attr_edge =exp.attribute(
                inputs=inputs,
                target=target,
                additional_forward_args=additional_forward_args,
                **attr_kwargs
            )

            explanation = exp_attr_edge[0].squeeze(0).detach()
            return explanation
            #return exp_attr_edge[0].squeeze(0).detach()
        return captum_explainer

    def generate_lrp(self, seed=None, **kwargs):
        import captum
        import torch_geometric
        import torch

        captum.attr._core.lrp.SUPPORTED_LAYERS_WITH_RULES[
            DummyConv
        ] = captum.attr._utils.lrp_rules.EpsilonRule
        captum.attr._core.lrp.SUPPORTED_LAYERS_WITH_RULES[
            DummyConvWithWeights
        ] = captum.attr._utils.lrp_rules.EpsilonRule
        captum.attr._core.lrp.SUPPORTED_LAYERS_WITH_RULES[
            torch_geometric.nn.aggr.basic.SumAggregation
        ] = captum.attr._utils.lrp_rules.EpsilonRule
        captum.attr._core.lrp.SUPPORTED_LAYERS_WITH_RULES[
            torch.nn.modules.activation.Threshold
        ] = captum.attr._utils.lrp_rules.EpsilonRule
        
        return self.generate_captum(LRP, seed=seed, **kwargs)
    
    def explain_node(self, node_idx):
        return self.explainer(self.model, self.data.x, self.data.edge_index,node_idx=node_idx)
    
    def __str__(self) -> str:
        return 'LRP'
    

class random_explainer(explainer_class):
    def __init__(self, model, data, **kwargs) -> None:
        super().__init__(model, data, **kwargs)
        return
    
    def explain_node(self, node_idx):
        _, _, _, edge_mask = tgu.k_hop_subgraph(node_idx, 2, self.data.edge_index, relabel_nodes=False)
        explanation = torch.zeros(self.data.num_edges, dtype=torch.int64)
        explanation[edge_mask] = torch.randint(0, 2, (edge_mask.sum(),))

        return explanation
    
    def __str__(self) -> str:
        return 'Random'
    




class SubgraphX_explainer(explainer_class):
    def __init__(self, model, data, **kwargs) -> None:
        super().__init__(model, data, **kwargs)
        model_name = model.__class__.__name__
        if model_name == 'LPModel':
            self.model = model.subgraphx_model()
        else:
            self.model = model
        self.labels = self.model(data.x, data.edge_index).argmax(dim=1)
        self.explainer = SubgraphX(self.model, num_hops=2)
        return

    def explain_node(self, node_idx):
        
        explanation = self.explainer.get_explanation_node(x=self.data.x, edge_index=self.data.edge_index, node_idx=node_idx, label=self.labels[node_idx], max_nodes=1000)
        return explanation.edge_imp
    
    def __str__(self) -> str:
        return 'SubgraphX'
    


class flowx_explainer(explainer_class):
    def __init__(self, model, data, **kwargs) -> None:
        
        from dig.xgraph.method import FlowX

        super().__init__(model, data, **kwargs)
        self.model = model.subgraphx_model()
        self.labels = self.model(data.x, data.edge_index).argmax(dim=1)

        self.explainer = FlowX(self.model)
        self.num_classes = self.model.num_classes
        self.edge_index_with_self_loops = add_self_loops(data.edge_index)[0]
        self.edge_mask_of_not_loops = self.edge_index_with_self_loops[0] != self.edge_index_with_self_loops[1]
        return

    def explain_node(self, node_idx):
        _, masks, _ = self.explainer(self.data.x, self.data.edge_index, node_idx=node_idx, num_classes=self.num_classes)
        label = self.labels[node_idx]
        filtered_mask = masks[label][self.edge_mask_of_not_loops]
        return filtered_mask
    
    def __str__(self) -> str:
        return 'FlowX'