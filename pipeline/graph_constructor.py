from rdkit import Chem 
import torch 
from torch_geometric.data import Data


def smiles_to_graph(smiles_str, label):
    mol = Chem.MolFromSmiles(smiles_str)
    

    atomic_numbers = []
    is_in_ring = [] 
    atomic_sizes = [] 
    node_labels = {} 
    bond_types = [] 
    # for each edge, store the bond level information, there should be as many bond type values as there are edges. 
    # figure out a storage for ionic bonds. 

    pt = Chem.GetPeriodicTable() 
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atomic_number = atom.GetAtomicNum()
        ring_status = atom.IsInRing()
        atomic_size = pt.GetRvdw(atomic_number) 

        atomic_sizes.append(atomic_size)
        atomic_numbers.append(atomic_number) 
        is_in_ring.append(ring_status)
        node_labels[idx] = pt.GetElementName(atomic_number)

    x = torch.concat([
        torch.tensor([atomic_numbers], dtype = torch.float),
        torch.tensor([is_in_ring], dtype = torch.float),
        torch.tensor([atomic_sizes], dtype = torch.float)
        ], dim = 0)

    

    edge_indices = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondType() 
        bond_types.append(bond_type) 
        edge_indices.append([start, end])
        edge_indices.append([end, start])
    
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    edge_attributes = torch.tensor(bond_types, dtype = torch.float) 
    y = torch.tensor([label], dtype=torch.float)

    return Data(x=x.T, 
                edge_index=edge_index, 
                y=y, 
                # node_labels = node_labels, 
                edge_attr = edge_attributes)
