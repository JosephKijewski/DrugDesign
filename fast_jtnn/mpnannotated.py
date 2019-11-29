import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from nnutils import *
from chemutils import get_mol

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']

#Atom feature dimension I assume? Yes, see atom_features
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
#Bond feature dimension I assume?
BOND_FDIM = 5 + 6
#Max neighbors I assume?
MAX_NB = 6

#Checks if something is in a set and then sets to true all elements of the set that
#are the same as that element. If that element is not in the set, you set the element
#to be the last element of the set.
def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return map(lambda s: x == s, allowable_set)

def atom_features(atom):
    #returns a tensor that is made up of concatenation of lists
    #first line sets 1 where the atoms element is in elem_list
    #Then second sets one in position of atoms degree
    #Then formal charge from all the possibilities, then chiral tag, then aromaticity
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3])
            + [atom.GetIsAromatic()])

def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0,1,2,3,4,5])
    return torch.Tensor(fbond + fstereo)

class MPN(nn.Module):

    def __init__(self, hidden_size, depth):
        super(MPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

    def forward(self, fatoms, fbonds, agraph, bgraph, scope):
        #I believe create_var is a function that is mostly deprecated at this point,
        #I think it basically just returns the tensor you plug in.
        fatoms = create_var(fatoms)
        fbonds = create_var(fbonds)
        #agraph is the tensor where each row contains the indices of the bonds that lead
        #to that atom.
        agraph = create_var(agraph)
        bgraph = create_var(bgraph)

        binput = self.W_i(fbonds)
        message = F.relu(binput)

        for i in xrange(self.depth - 1):
            nei_message = index_select_ND(message, 0, bgraph)
            nei_message = nei_message.sum(dim=1)
            nei_message = self.W_h(nei_message)
            message = F.relu(binput + nei_message)

        nei_message = index_select_ND(message, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        ainput = torch.cat([fatoms, nei_message], dim=1)
        atom_hiddens = F.relu(self.W_o(ainput))

        max_len = max([x for _,x in scope])
        batch_vecs = []
        for st,le in scope:
            cur_vecs = atom_hiddens[st : st + le].mean(dim=0)
            batch_vecs.append( cur_vecs )

        mol_vecs = torch.stack(batch_vecs, dim=0)
        return mol_vecs 

    #This method looks like its intended to load in all the features for the batch
    @staticmethod
    def tensorize(mol_batch):
        padding = torch.zeros(ATOM_FDIM + BOND_FDIM)
        fatoms,fbonds = [],[padding] #Ensure bond is 1-indexed
        in_bonds,all_bonds = [],[(-1,-1)] #Ensure bond is 1-indexed
        scope = []
        total_atoms = 0

        for smiles in mol_batch:
            mol = get_mol(smiles)
            #mol = Chem.MolFromSmiles(smiles)
            n_atoms = mol.GetNumAtoms()
            #It looks like fatoms is a list of tensors of atoms. Inbonds seems
            #to be a list of lists of bonds that each atom is involved in
            for atom in mol.GetAtoms():
                fatoms.append( atom_features(atom) )
                in_bonds.append([])

            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                # Total atoms is updated at end of each smiles string. I believe idx would be order in which
                # molecule iterates through atoms???
                x = a1.GetIdx() + total_atoms
                y = a2.GetIdx() + total_atoms

                b = len(all_bonds) 
                all_bonds.append((x,y))
                # Here we have torch.cat that stacks the features along the dimension 0 as specified in the last argument.
                fbonds.append( torch.cat([fatoms[x], bond_features(bond)], 0) )
                in_bonds[y].append(b)

                b = len(all_bonds)
                all_bonds.append((y,x))
                fbonds.append( torch.cat([fatoms[y], bond_features(bond)], 0) )
                in_bonds[x].append(b)
            
            scope.append((total_atoms,n_atoms))
            total_atoms += n_atoms

        total_bonds = len(all_bonds)
        #fatoms is now the stacking along the row dimension of the atom_features
        #we had from earlier.
        fatoms = torch.stack(fatoms, 0)
        fbonds = torch.stack(fbonds, 0)
        #long just turns data type to long
        agraph = torch.zeros(total_atoms,MAX_NB).long()
        bgraph = torch.zeros(total_bonds,MAX_NB).long()

        for a in xrange(total_atoms):
            for i,b in enumerate(in_bonds[a]):
                agraph[a,i] = b

        for b1 in xrange(1, total_bonds):
            x,y = all_bonds[b1]
            for i,b2 in enumerate(in_bonds[x]):
                if all_bonds[b2][0] != y:
                    bgraph[b1,i] = b2

        return (fatoms, fbonds, agraph, bgraph, scope)