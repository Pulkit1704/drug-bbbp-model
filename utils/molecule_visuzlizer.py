import networkx as nx
import matplotlib.pyplot as plt 

color_assignment = {
  'Carbon': 'grey',
  'Nitrogen': 'red',
  'Oxygen': 'blue',
  'Fluorine': 'yellow',
  'Chlorine': 'green',
  'sulphur': 'pink'
}


def visualize_networkx(data): 

  figure = plt.figure(figsize = (10, 10)) 

  G = nx.Graph() 

  num_nodes = data.x.shape[-1] 

  G.add_nodes_from(range(num_nodes))

  edges = data.edge_index.t().tolist() 

  G.add_edges_from(edges) 

  # node_colors = [color_assignment.get(data.node_labels[i], 'black') for i in range(num_nodes)]


  positions = nx.kamada_kawai_layout(G) 

  nx.draw(G, pos = positions, 
          with_labels = True, 
          labels = data.node_labels) 
  
  return figure 
