import torch.nn as nn 
import torch_geometric.nn as gnn 
import torch 

class GraphClassifier(nn.Module): 

  def __init__(self, *args, node_attributes_shape, edge_attributes_shape, hidden_dim, dropout_rate = 0.1,  **kwargs):
    super().__init__(*args, **kwargs)

    self.node_encoder = nn.Linear(node_attributes_shape, hidden_dim) 
    self.edge_encoder = nn.Linear(edge_attributes_shape, hidden_dim) 

    self.conv1 = gnn.GINEConv(nn.Sequential(
      nn.Linear(hidden_dim, hidden_dim*2), 
      nn.BatchNorm1d(hidden_dim*2),
      nn.ReLU(), 
      nn.Linear(hidden_dim*2, hidden_dim) 
    )) 

    self.ln1 = gnn.LayerNorm(hidden_dim) 

    self.conv2 = gnn.GINEConv(nn.Sequential(
      nn.Linear(hidden_dim, hidden_dim*2), 
      nn.BatchNorm1d(hidden_dim*2),
      nn.ReLU(), 
      nn.Linear(hidden_dim*2, hidden_dim) 
    ))

    self.ln2 = gnn.LayerNorm(hidden_dim) 

    self.conv3 = gnn.GINEConv(nn.Sequential(
      nn.Linear(hidden_dim, hidden_dim), 
      nn.BatchNorm1d(hidden_dim),
      nn.ReLU(), 
      nn.Linear(hidden_dim, hidden_dim) 
    ))

    self.ln3 = gnn.LayerNorm(hidden_dim) 

    self.conv4 = gnn.GINEConv(nn.Sequential(
      nn.Linear(hidden_dim, hidden_dim), 
      nn.BatchNorm1d(hidden_dim),
      nn.ReLU(), 
      nn.Linear(hidden_dim, hidden_dim) 
    ))

    self.ln4 = gnn.LayerNorm(hidden_dim) 

    self.pool = gnn.GlobalAttention(nn.Sequential(
      nn.Linear(hidden_dim, hidden_dim//2),
      nn.ReLU(), 
      nn.Dropout(dropout_rate),
      nn.Linear(hidden_dim//2, 1) 
    )) 

    self.classifier = nn.Sequential(
      nn.Linear(hidden_dim*2, hidden_dim), 
      nn.ReLU(), 
      nn.Linear(hidden_dim, int(hidden_dim/2)),
      nn.ReLU(), 
      nn.Linear(int(hidden_dim/2), 1)
    )

    self.dropout = nn.Dropout(dropout_rate) 
    self.sigmoid = nn.Sigmoid() 
    self.activation = nn.ReLU() 

  def forward(self, x, edge_index, edge_attrs, batch = None): 

    x = self.node_encoder(x) 
    edge_attrs = self.edge_encoder(edge_attrs)   

    ge1 = self.conv1(x, edge_index, edge_attrs) 
    ge1 = self.dropout(self.activation(ge1)) 
    x = self.activation(x + ge1) 
    x = self.ln1(x) 
    x = self.dropout(x) 

    ge2 = self.conv2(x, edge_index, edge_attrs) 
    ge2 = self.dropout(self.activation(ge2)) 
    x = self.activation(x + ge2) 
    x = self.ln2(x) 
    x = self.dropout(x) 

    ge3 = self.conv3(x, edge_index, edge_attrs) 
    ge3 = self.dropout(self.activation(ge3)) 
    x = self.activation(x + ge3) 
    x = self.ln3(x) 
    x = self.dropout(x) 

    ge4 = self.conv4(x, edge_index, edge_attrs) 
    ge4 = self.dropout(self.activation(ge4)) 
    x = self.activation(x + ge4) 
    x = self.ln4(x) 
    x = self.dropout(x) 

    graph_embedding_att = self.pool(x, batch) 

    graph_embedding_max = gnn.global_add_pool(x, batch) 

    graph_embedding = torch.concat([graph_embedding_att, graph_embedding_max], dim = 1)

    prediction = self.classifier(graph_embedding) 

    return prediction 


  def predict(self, x, edge_index, edge_attrs, batch): 

    preds = self.forward(x, edge_index, edge_attrs, batch)  

    return self.sigmoid(preds) 
  
