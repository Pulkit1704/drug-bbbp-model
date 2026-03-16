# Drug Blood-Brain Barrier Penetration (BBBP) Prediction
**A Graph Neural Network (GNN) approach to predict pharmacokinetic properties of small molecules.**

## 🚀 The Challenge
Predicting whether a molecule can cross the Blood-Brain Barrier (BBB) is a multi-million dollar question in neuro-drug discovery. This project implements an end-to-end graph generation and classification pipeline using Graph Isomorphism Network with Edge features (GINE) network to automate this prediction using the MoleculeNet BBBP dataset.

## 🛠️ Technical Stack
* **Architecture**: GINE (Graph Isomorphism Network with Edge Features). I chose this specifically because standard GINs ignore bond attributes, which are crucial for chemical identity.

* **Data Strategy**: Combined SMILES/labels from BBBP and B3DB datasets.

* **Splitting**: Used Bemis-Murcko Scaffold Splitting. This is much harder than a random split; the model is tested on chemical families it has never seen during training.

* **Featurization**: custom RDKit pipeline for node (atoms) and edge (bonds) features


## 📊 Performance
* **F1-score (macro average):** 0.82
* **Accuracy**: 84%
* **Classification report**: 
```
             precision    recall  f1-score   support

         0.0       0.82      0.73      0.77       713
         1.0       0.85      0.90      0.88      1218

    accuracy                           0.84      1931
   macro avg       0.83      0.82      0.82      1931
weighted avg       0.84      0.84      0.84      1931
```
## **Loss plot** 
![Loss vlaidation plot](./trained_model/training_plot.png)

## 📁 Highlights
* `model/`: Modular GNN layers.
* `pipeline/`: Robust training and evaluation logic.
* `molecule_visualizer.py`: Function to visualize a graph as a networkx plot. 