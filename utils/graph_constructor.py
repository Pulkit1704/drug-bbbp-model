import torch 
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import pandas as pd 
import datamol as dm 

def smiles_to_graph(smiles_str, label, add_node_labels = False):

  mol = dm.to_mol(smiles_str, sanitize= True)
  
  if mol is None: 
    print(f"smiles conversion returned an empty molecule: {smiles_str}") 
    return None 
  
  node_labels = {} 

  periodic_table = GetPeriodicTable()
  
  atoms = []
  for atom in mol.GetAtoms():
    atoms.append([
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetHybridization().real,
        atom.GetIsAromatic() * 1.0,
        atom.GetTotalNumHs(),
        atom.IsInRing() * 1.0
    ])

    if add_node_labels: 
       id = atom.GetIdx() 
       name = periodic_table.GetElementName(atom.GetAtomicNum()) 

       node_labels[id] = name 

  x = torch.tensor(atoms, dtype=torch.float)

  edge_indices = []
  edge_attrs = []
  

  bt_map = {Chem.rdchem.BondType.SINGLE: 1, Chem.rdchem.BondType.DOUBLE: 2, 
            Chem.rdchem.BondType.TRIPLE: 3, Chem.rdchem.BondType.AROMATIC: 1.5}

  for bond in mol.GetBonds():
    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    bt = bt_map.get(bond.GetBondType(), 0)
    
    edge_indices += [[i, j], [j, i]]
    edge_attrs += [[bt], [bt]]

  edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
  edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

  y = torch.tensor([label], dtype=torch.float)

  return Data(x=x, 
              edge_index=edge_index, 
              y=y, 
              node_labels = node_labels, 
              edge_attr = edge_attr)


def get_scaffold(smiles):
  # Returns the canonical SMILES of the Bemis-Murcko scaffold
  
  mol = dm.to_mol(smiles, sanitize= True)
  
  if mol is None:
      print(f"smiles conversion returned an empty molecule: {smiles}")
      return None
  
  return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_split(smiles_frame: pd.DataFrame, train_frac=0.8):

  indices = smiles_frame.index 
  smiles_list = smiles_frame['smiles'].tolist()
  
  scaffolds = defaultdict(list)
  for idx, smiles in zip(indices, smiles_list):
    scaffold = get_scaffold(smiles)

    if scaffold is None: 
        print(f"{smiles} could not be coverted to scaffold") 
        continue 

    scaffolds[scaffold].append(idx)
  

  sorted_scaffolds = sorted(scaffolds.values(), key=len, reverse=True)
  
  train_indices = []
  test_indices = []
  train_cutoff = train_frac * len(smiles_list)
  

  for group in sorted_scaffolds:
    if len(train_indices) + len(group) <= train_cutoff:
        train_indices.extend(group)
    else:
        test_indices.extend(group)
          
  return smiles_frame.iloc[train_indices, :], smiles_frame.iloc[test_indices, :]

