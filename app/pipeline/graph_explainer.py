# pipeline to run the GNNExplainer on the model and user data and return the image output. 
from torch_geometric.explain import GNNExplainer, Explainer
from torch_geometric.explain.config import ModelConfig
from model.gnn_model import GraphClassifier

class ModelExplainer(): 


    def __init__(self, model: GraphClassifier): 
        
        self.explainer = Explainer(
            model=model, 
            algorithm=GNNExplainer(epochs = 100), 
            explanation_type='model', 
            node_mask_type='attributes', 
            edge_mask_type='object', 
            model_config=ModelConfig(
                mode = 'binary_classification', 
                task_level = 'graph', 
                return_type = 'raw'
            )
        )
    

    def explain_graph(self, data): 

        explanation = self.explainer(
            x = data.x, 
            edge_index = data.edge_index, 
            edge_attrs = data.edge_attr, 
            batch = data.batch 
        )

        node_importance = explanation.node_mask 
        edge_importance = explanation.edge_mask 

        node_scores = node_importance.mean(dim = 1) 

        return node_scores, edge_importance 
