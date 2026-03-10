import torch 
import torch.nn as nn 
import torch_geometric.nn as gnn 


class GraphClassifier(nn.Module): 

  def __init__(self, *args, node_attributes_shape, hidden_dim, dropout_rate = 0.5,  **kwargs):
    super().__init__(*args, **kwargs)

    self.conv1 = gnn.SAGEConv(node_attributes_shape, hidden_dim) 
    self.norm = nn.LayerNorm(hidden_dim) 
    self.conv2 = gnn.SAGEConv(hidden_dim, hidden_dim) 

    self.dropout = nn.Dropout(dropout_rate) 

    self.classifier = nn.Sequential(
      nn.Linear(hidden_dim, int(hidden_dim/2)), 
      nn.ReLU(), 
      nn.Linear(int(hidden_dim/2), int(hidden_dim/2)),
      nn.ReLU(), 
      nn.Linear(int(hidden_dim/2), 1)
    )

    self.sigmoid = nn.Sigmoid() 
    self.activation = nn.ReLU() 

  def forward(self, x, edge_index, batch = None): 

    graph_embedding = self.activation(self.dropout(self.conv1(x, edge_index)))
    graph_embedding = self.norm(graph_embedding) 
    graph_embedding = self.activation(self.dropout(self.conv2(graph_embedding, edge_index)))  

    graph_embedding = gnn.global_add_pool(graph_embedding, batch) 

    prediction = self.classifier(graph_embedding) 

    return prediction 


  def predict(self, x, edge_index): 

    preds = self.forward(x, edge_index) 

    return self.sigmoid(preds) 
  
