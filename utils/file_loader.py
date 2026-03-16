import os 
import pandas as pd 
import numpy as np 

def load_csv_files(root = "./data/"):

  if not os.path.exists(os.path.join(root, "BBBP.csv")): 
    print(f"BBBP.csv file not found")

  molecule_net_data = pd.read_csv(os.path.join(root, "BBBP.csv")).loc[:, ['smiles', 'p_np']]
  molecule_net_data = molecule_net_data.rename(columns = {
    'smiles': 'smiles', 
    'p_np': 'label'
  })

  if not os.path.exists(os.path.join(root, "B3DB_classification_extended.tsv.gz")): 
    print("b3d8 file not found") 

  b3d8_data = pd.read_csv(os.path.join(root, 'B3DB_classification_extended.tsv.gz'), sep = '\t')

  b3d8_data['label'] = np.where(b3d8_data['BBB+/BBB-'] == 'BBB+', 1, 0) 

  b3d8_data = b3d8_data.loc[:, ['SMILES', 'label']] 

  b3d8_data = b3d8_data.rename(columns = {
    'SMILES': 'smiles', 
    'label': 'label'
  })

  combined_data = pd.concat([molecule_net_data, b3d8_data],axis = 0)

  combined_data = combined_data.drop_duplicates(['smiles']) 
  combined_data = combined_data.dropna(axis = 0)

  return combined_data 