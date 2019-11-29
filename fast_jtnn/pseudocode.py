import random
import torch
import torch.nn as nn
import rdkit.Chem as Chem
from rdkit.Chem.rdchem import *
import torch.nn.functional as F
from nnutils import *
from chemutils_used import get_mol
from mpn import atom_features, bond_features

from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os, random

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_FDIM = 5 + 6
n = 100

# -----------------------------------------------------   C L A S S E S   ------------------------------------------------------------------------- #

### TODO: Probably should include labels 
class WalkDataset(Dataset):
    def __init__(self, smiles_list):
        self.smiles_list = smiles_list

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, index):
        return tensorize(self.smiles_list[index])

# -----------------------------------------------------   U S E F U L  F U N C T I O N S   --------------------------------------------------------- #

def onek_encoding_unk(s, allowable_set):
    if s not in allowable_set:
        s = allowable_set[-1]
    return [int(x == s) for x in allowable_set]

def atom_features(atom):
    return np.array(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3])
            + [atom.GetIsAromatic()])

def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0,1,2,3,4,5])
    return np.array(fbond + fstereo)

# This function takes in a smiles_batch, then for each smiles, prepares arguments for the rand_walk algorithm
# and then executes it. Finally, outputs rand_walks for the entire batch.
def tensorize(smiles):
    neighbor_dict = dict()
    mol = get_mol(smiles)
    n_atoms = mol.GetNumAtoms()
    # Initialize neighbor set, as well as initialize and fill fatoms
    fatoms = np.zeros((n_atoms, ATOM_FDIM))
    for atom in mol.GetAtoms():
        fatoms[atom.GetIdx(), :] = atom_features(atom)
        neighbor_dict[atom.GetIdx()] = set()
    # Initialize and fill fbonds, fill neighbor_dict
    fbonds = np.zeros((n_atoms, n_atoms, BOND_FDIM))
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        x = a1.GetIdx()
        y = a2.GetIdx()
        neighbor_dict[x].add(y)
        neighbor_dict[y].add(x)
        fbonds[x, y, :] = bond_features(bond)
    # With prepared arguments, execute rand_walk
    return rand_walk(fatoms, fbonds, neighbor_dict, n, mol)[0]

def rand_walk(fatoms, fbonds, neighbor_dict, n, mol):
    grid = np.zeros((n, n, 2))
    grid.fill(None)
    init_atom = random.randint(0, fatoms.shape[0]-1)
    random_neighbor = random.sample(neighbor_dict[init_atom], 1)[0]
    grid[0, 0, 0] = init_atom
    grid[0, 0, 1] = random_neighbor
    to_expand = [(0, 0, init_atom)]
    while (len(to_expand) > 0):
        expand_triple = to_expand.pop(random.randint(0, len(to_expand)-1))
        i = expand_triple[0]
        j = expand_triple[1]
        current_atom = expand_triple[2]
        neighbors = neighbor_dict[current_atom]
        if (len(neighbors) >= 2):
            expansions = random.sample(neighbors, 2)
            lower_update_atom = expansions[0]
            right_update_atom = expansions[1]
        else:
            expansion = random.sample(neighbors, 1)[0]
            lower_update_atom = expansion
            right_update_atom = expansion
        if (i+1)<n and np.isnan(grid[i+1, j, 0]):
            grid[i+1, j, 0] = lower_update_atom
            grid[i+1, j, 1] = current_atom
            to_expand.append((i+1, j, lower_update_atom))
        if (j+1)<n and np.isnan(grid[i, j+1, 0]):
            grid[i, j+1, 0] = right_update_atom
            grid[i, j+1, 1] = current_atom
            to_expand.append((i, j+1, right_update_atom))
    rand_walk = np.zeros((n, n, ATOM_FDIM+BOND_FDIM))
    for a in range(n):
        for b in range(n):
            rand_walk[a, b, :ATOM_FDIM] = atom_features(mol.GetAtomWithIdx(int(grid[a, b, 0])))
            rand_walk[a, b, ATOM_FDIM:] = bond_features(mol.GetBondBetweenAtoms(int(grid[a, b, 1]), int(grid[a, b, 0])))
    return rand_walk, grid

# ---------------------------------------   T E S T I N G   ------------------------------------------------------------- #

# Takes in a smiles, and outputs a random walk with information for testing
def tensorize_test(smiles):
    neighbor_dict = dict()
    mol = get_mol(smiles)
    n_atoms = mol.GetNumAtoms()
    # Initialize neighbor set, as well as initialize and fill fatoms
    fatoms = np.zeros((n_atoms, ATOM_FDIM))
    for atom in mol.GetAtoms():
        fatoms[atom.GetIdx(), :] = atom_features(atom)
        neighbor_dict[atom.GetIdx()] = set()
    # Initialize and fill fbonds, fill neighbor_dict
    fbonds = np.zeros((n_atoms, n_atoms, BOND_FDIM))
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        x = a1.GetIdx()
        y = a2.GetIdx()
        neighbor_dict[x].add(y)
        neighbor_dict[y].add(x)
        fbonds[x, y, :] = bond_features(bond)
    # With prepared arguments, execute rand_walk
    app = rand_walk(fatoms, fbonds, neighbor_dict, n, mol)
    rand_walk_comp = app[0]
    rand_walk_incomp = app[1]
    return rand_walk_comp, rand_walk_incomp

# This is the tester function. Given a numpy array of complete (including atom features and bond features)
# and incomplete (including just atom idx and from-atom idx) rand walk, it tests them for two properties.
# First, it checks that for each (i, j) in the incomplete rand walk, one of its upper or left neighbor's
# atom is the (i, j) from-atom, and there exists a bond between the from-atom at (i, j) and the atom at (i, j).
# Second, it checks that for each (i, j) in the rand_walk, the atom features correspond to the atom features
# for the (i, j) atom in the incomp rand_walk, and the bond feautres correspond to the bond features for the
# bond between the (i, j) from-atom and the (i, j) atom.
# The smiles argument should include the smiles used to generate the rand walk.

def sanity_check(rand_walk_comp, rand_walk_incomp, smiles):
    x_shape = rand_walk_incomp.shape[0]
    mol = get_mol(smiles)
    for a in range(x_shape):
        for b in range(x_shape):
            correct = False
            neighbor_atoms = set()
            if a > 0:
                neighbor_atoms.add(rand_walk_incomp[a-1, b, 0])
            if b > 0:
                neighbor_atoms.add(rand_walk_incomp[a, b-1, 0])
            if ((a == 0) and (b == 0)):
                temp = mol.GetAtomWithIdx(int(rand_walk_incomp[a, b, 0])).GetNeighbors()
                for f in temp:
                    neighbor_atoms.add(f.GetIdx())
            for c in neighbor_atoms:
                # The from-atom for each element of the grid is one of the neighbors AND a bond exists between that
                # from-atom and the atom at the element of the grid.
                if ((rand_walk_incomp[a, b, 1] == c) and (mol.GetBondBetweenAtoms(int(c), int(rand_walk_incomp[a, b, 0])) != None)):
                    correct = True
                    continue
            if (not correct):
                print("false1")
                return False
    # Test that all atom and bond features are correctly copied over in the rand_walk
    for a in range(x_shape):
        for b in range(x_shape):
            correct_atom_feat = atom_features(mol.GetAtomWithIdx(int(rand_walk_incomp[a, b, 0])))
            if (not np.array_equal(correct_atom_feat, rand_walk_comp[a, b, :ATOM_FDIM])):
                print("false2")
                return False
            correct_bond_feat = bond_features(mol.GetBondBetweenAtoms(int(rand_walk_incomp[a, b, 1]), int(rand_walk_incomp[a, b, 0])))
            if (not np.array_equal(correct_bond_feat, rand_walk_comp[a, b, ATOM_FDIM:])):
                print("false3")
                return False
    return True

# ---------------------------------------------------   M A I N  M E T H O D   ------------------------------------------------------------------ #

fn = os.path.abspath('../data/moses/train.txt')
with open(fn) as f:
    data = f.read()
    data = data.split('\n')
    random.shuffle(data)
    # data is a list of strings
    rand_walk_comp, rand_walk_incomp = tensorize_test(data[0])
    print(sanity_check(rand_walk_comp, rand_walk_incomp, data[0]))
    #dataset = WalkDataset(data)
    #dataloader = DataLoader(dataset, batch_size=32)

# --------------------------------------------------- G R A V E Y A R D ------------------------------------------------------------- #


# def rand_walk(fatoms, fbonds, neighbor_dict, n, mol):
#     # Initialize grids
#     grid = np.zeros((n, n, ATOM_FDIM + BOND_FDIM))
#     grid.fill(None)
#     test_grid = np.zeros((n, n, 2))
#     test_grid.fill(None)
#     test_grid = np.array(test_grid, dtype=np.object)
#     # print(test_grid.shape) This gives whats expected

#     # Initialize top left fatoms and fbonds for rand_walk
#     init_atom = random.randint(0, fatoms.shape[0]-1)
#     grid[0, 0, :ATOM_FDIM] = fatoms[init_atom, :]
#     random_neighbor = random.sample(neighbor_dict[init_atom], 1)[0]
#     grid[0, 0, ATOM_FDIM:] = fbonds[random_neighbor, init_atom, :]

#     # Initialize top left atom and bond
#     test_grid[0, 0, 0] = mol.GetAtomWithIdx(init_atom)
#     test_grid[0, 0, 1] = mol.GetBondBetweenAtoms(random_neighbor, init_atom)

#     # Execute random walk algorithm
#     to_expand = [(0, 0, init_atom)]
#     while (len(to_expand) > 0):
#         # Get information about current atom, its neighbors, and the position you are in
#         expand_triple = to_expand.pop(random.randint(0, len(to_expand)-1))
#         i = expand_triple[0]
#         j = expand_triple[1]
#         current_atom = expand_triple[2]
#         neighbors = neighbor_dict[current_atom]

#         if len(neighbors) >= 2:
#             # Chooses two random neighbors of the current atom. 
#             expansions = random.sample(neighbors, 2)
#             lower_update_atom = expansions[0]
#             right_update_atom = expansions[1]
#             lower_update_bond = [current_atom, lower_update_atom]
#             right_update_bond = [current_atom, right_update_atom]
#             test_lower_update_atom = mol.GetAtomWithIdx(lower_update_atom)
#             test_right_update_atom = mol.GetAtomWithIdx(right_update_atom)
#             test_lower_update_bond = mol.GetBondBetweenAtoms(current_atom, lower_update_atom)
#             test_right_update_bond = mol.GetBondBetweenAtoms(current_atom, right_update_atom)
#         else:
#             expansion = random.sample(neighbors, 1)[0]
#             lower_update_atom = expansion
#             right_update_atom = expansion
#             lower_update_bond = [current_atom, expansion]
#             right_update_bond = [current_atom, expansion]
#             test_lower_update_atom = mol.GetAtomWithIdx(expansion)
#             test_right_update_atom = mol.GetAtomWithIdx(expansion)
#             test_lower_update_bond = mol.GetBondBetweenAtoms(current_atom, expansion)
#             test_right_update_bond = mol.GetBondBetweenAtoms(current_atom, expansion)
#         if (i+1)<n and np.isnan(grid[i+1, j, 0]):
#             # Updates the grid lower neighbors with the fatoms and fbonds from before
#             grid[i+1, j, 0:ATOM_FDIM] = fatoms[lower_update_atom, :]
#             grid[i+1, j, ATOM_FDIM:] = fbonds[lower_update_bond[0], lower_update_bond[1], :]
#             test_grid[i+1, j, 0] = test_lower_update_atom
#             test_grid[i+1, j, 1] = test_lower_update_bond
#             to_expand.append((i+1, j, lower_update_atom))
#         if (j+1)<n and np.isnan(grid[i, j+1, 0]):
#             # Updates the grid right neighbors with the fatoms and fbonds from before
#             grid[i, j+1, 0:ATOM_FDIM] = fatoms[right_update_atom, :]
#             grid[i, j+1, ATOM_FDIM:] = fbonds[right_update_bond[0], right_update_bond[1]]
#             test_grid[i, j+1, 0] = test_right_update_atom
#             test_grid[i, j+1, 0] = test_right_update_bond
#             to_expand.append((i, j+1, right_update_atom))
#     return grid, test_grid