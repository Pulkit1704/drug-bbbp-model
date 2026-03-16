# convert the primary code into a class 
# move the utility functions to a seperate utils module 
# 

import logging 
from utils.graph_constructor import smiles_to_graph, scaffold_split
from utils.file_loader import load_csv_files 
from torch_geometric.loader import DataLoader
import pandas as pd 


def construct_graphs(data_frame: pd.DataFrame): 

  if 'smiles' not in data_frame.columns or 'label' not in data_frame.columns: 
    logging.error(f"did not find smiles or label entry in data: {data_frame.columns}")

    return 

  graphs = [] 
  for i in range(data_frame.shape[0]): 

    smiles_string = data_frame['smiles'].iloc[i] 
    target_label = data_frame['label'].iloc[i] 

    graph = smiles_to_graph(smiles_string, target_label) 

    if graph is None:
      print(f"smiles conversion error, skipping string {smiles_string}")
      continue 

    graphs.append(graph) 

  return graphs 


def construct_data_loader(graph_array: list, batch_size: int, shuffle: bool = True):


  return DataLoader(graph_array, batch_size = batch_size, shuffle = shuffle) 


def split_data(data_frame: pd.DataFrame, train_split_size = 0.8): 

  train_split, validation_split = scaffold_split(data_frame, train_frac= train_split_size)

  return train_split, validation_split 


def main(batch_size = 64, train_frac = 0.5): 

  raw_data = load_csv_files() 

  train_frame, test_frame = split_data(raw_data, train_frac) 

  train_graphs = construct_graphs(train_frame)
  test_graphs = construct_graphs(test_frame) 

  train_loader = construct_data_loader(train_graphs, batch_size, shuffle = True) 
  test_loader = construct_data_loader(test_graphs, batch_size, shuffle = False) 


  return train_loader, test_loader 

