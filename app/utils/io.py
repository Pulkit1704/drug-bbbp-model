import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from model.gnn_model import GraphClassifier


def load_csv_files(root=Path("./data/")):

    if not os.path.exists(os.path.join(root, "BBBP.csv")):
        print(f"BBBP.csv file not found")

    molecule_net_data = pd.read_csv(os.path.join(root, "BBBP.csv")).loc[
        :, ["smiles", "p_np"]
    ]
    molecule_net_data = molecule_net_data.rename(
        columns={"smiles": "smiles", "p_np": "label"}
    )

    if not os.path.exists(os.path.join(root, "B3DB_classification_extended.tsv.gz")):
        print("b3d8 file not found")

    b3d8_data = pd.read_csv(
        os.path.join(root, "B3DB_classification_extended.tsv.gz"), sep="\t"
    )

    b3d8_data["label"] = np.where(b3d8_data["BBB+/BBB-"] == "BBB+", 1, 0)

    b3d8_data = b3d8_data.loc[:, ["SMILES", "label"]]

    b3d8_data = b3d8_data.rename(columns={"SMILES": "smiles", "label": "label"})

    combined_data = pd.concat([molecule_net_data, b3d8_data], axis=0)

    combined_data = combined_data.drop_duplicates(["smiles"])
    combined_data = combined_data.dropna(axis=0)

    return combined_data


def save_model(model_checkpoint: dict, save_path: Path):

    if not os.path.exists(save_path):
        print("save path not found")
        return

    filepath = os.path.join(save_path, "trained_model.pth")

    torch.save(model_checkpoint, filepath)

    print(f"model saved to {filepath}")
    return


def load_model(load_path:Path) -> GraphClassifier: 

    if not Path(load_path).exists(): 
        print(f"{load_path} not found") 
    
    model_checkpoint = torch.load(load_path) 

    hyper_parameters = model_checkpoint['hyperparameters']
    model_state_dict = model_checkpoint['weights']

    model = GraphClassifier(
        node_attributes_shape= hyper_parameters['node_attributes_shape'], 
        edge_attributes_shape= hyper_parameters['edge_attributes_shape'], 
        hidden_dim= hyper_parameters['model_hidden_dim'], 
        dropout_rate= hyper_parameters['model_dropout']
    )
    
    model.load_state_dict(model_state_dict) 

    return model 