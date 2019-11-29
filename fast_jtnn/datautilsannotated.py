from torch.utils.data import Dataset
from mol_tree import MolTree
import numpy as np

class MoleculeDataset(Dataset):

    def __init__(self, data_file):
        with open(data_file) as f:
            self.data = [line.strip("\r\n ").split()[0] for line in f]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        #Convert from smiles string to mol_tree, which contains self.mol and self.smiles
        #as fields. It also has a set of nodes that I believe are the cliques. There is
        #a lot of extra computation if we don't use the junction tree representation.
        smiles = self.data[idx]
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        mol_tree.assemble()
        return mol_tree

class PropDataset(Dataset):

    def __init__(self, data_file, prop_file):
        self.prop_data = np.loadtxt(prop_file)
        with open(data_file) as f:
            self.data = [line.strip("\r\n ").split()[0] for line in f]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        smiles = self.data[idx]
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        mol_tree.assemble()
        return mol_tree, self.prop_data[idx]