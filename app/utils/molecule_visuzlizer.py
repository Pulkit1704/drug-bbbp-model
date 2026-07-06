from rdkit import Chem 
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem
import py3Dmol


color_assignment = {
  'Carbon': 'grey',
  'Nitrogen': 'red',
  'Oxygen': 'blue',
  'Fluorine': 'yellow',
  'Chlorine': 'green',
  'sulphur': 'pink'
}


def visualize_molecule(mol, 
                        highlight_atoms = None, 
                        highlight_colors = None, 
                        width=500, height=500):

    mol_copy = Chem.Mol(mol) 

    Chem.Draw.rdMolDraw2D.PrepareMolForDrawing(mol_copy)

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().useBWAtomPalette() 

    drawer.DrawMolecule(
        mol,
        highlightAtoms = highlight_atoms, 
        highlightAtomColors = highlight_colors
    )

    drawer.FinishDrawing() 

    return drawer.GetDrawingText() 


def visualize_2d_quantitative_svg(smiles, 
                                  node_scores,
                                  width = 500,
                                  height = 500):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return "Invalid SMILES"
    
    # Chem.Draw.rdMolDraw2D.PrepareMolForDrawing(mol)
    
    # Normalize scores to [0, 1]
    scores = node_scores.detach().cpu().numpy()
    s_min, s_max = scores.min(), scores.max()
    
    # Prevent division by zero if all scores are identical
    if s_max - s_min > 1e-8:
        norm_scores = (scores - s_min) / (s_max - s_min)
    else:
        norm_scores = scores * 0.0 

    atom_colors = {}
    highlight_atoms = []

    for i in range(mol.GetNumAtoms()):
        s = float(norm_scores[i])

        if s < 0.5:
            # First half: Interpolate from Bright Red to Slate Grey
            local_t = s / 0.5
            r_val = 0.8 + local_t * (0.75 - 1.0)
            g_val = 0.2 + local_t * (0.76 - 0.0)
            b_val = 0.2 + local_t * (0.78 - 0.0)
        else:
            # Second half: Interpolate from Slate Grey to Bright Blue
            local_t = (s - 0.5) / 0.5
            r_val = 0.75 + local_t * (0.0 - 0.75)
            g_val = 0.76 + local_t * (0.0 - 0.76)
            b_val = 0.78 + local_t * (1.0 - 0.78)
            
        color = (r_val, g_val, b_val)
        atom_colors[i] = color
        highlight_atoms.append(i)

    drawer_txt = visualize_molecule(mol, 
                                    highlight_atoms, 
                                    atom_colors,
                                    width,
                                    height)

    return drawer_txt


def get_3d_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)

    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)

    return mol


def visualize_3d(smiles, node_scores):
    mol = get_3d_mol(smiles)

    mb = Chem.MolToMolBlock(mol)

    view = py3Dmol.view(width=500, height=500)
    view.setBackgroundColor('black')
    view.addModel(mb, "mol")
    view.setStyle({'stick': {}})

    scores = node_scores.detach().cpu().numpy()
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    for i, score in enumerate(scores):
        if score > 0.6:
            view.addStyle(
                {'serial': i + 1},
                {'sphere': {'color': 'red', 'radius': 0.5}}
            )

    view.zoomTo()
    return view