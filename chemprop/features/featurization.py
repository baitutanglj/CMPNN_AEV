from argparse import Namespace
from typing import List, Tuple, Union

from rdkit import Chem
import torch
import numpy as np
from itertools import zip_longest

# Atom feature sizes
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14
#################my addition############
EXTRA_ATOM_FDIM = 0
EXTRA_BOND_FDIM = 0
########################################
# Memoization
SMILES_TO_GRAPH = {}


def clear_cache():
    """Clears featurization cache."""
    global SMILES_TO_GRAPH
    SMILES_TO_GRAPH = {}


# def get_atom_fdim(args: Namespace) -> int:
#     """
#     Gets the dimensionality of atom features.
#
#     :param: Arguments.
#     """
#     return ATOM_FDIM
#################my addition###################
def get_atom_fdim(args: Namespace) -> int:
    """
    Gets the dimensionality of the atom feature vector.

    :param overwrite_default_atom: Whether to overwrite the default atom descriptors
    :return: The dimensionality of the atom feature vector.
    """
    return (not args.overwrite_default_atom_features) * ATOM_FDIM + EXTRA_ATOM_FDIM


def set_extra_atom_fdim(extra):
    """Change the dimensionality of the atom feature vector."""
    global EXTRA_ATOM_FDIM
    EXTRA_ATOM_FDIM = extra

def set_extra_bond_fdim(extra):
    """Change the dimensionality of the bond feature vector."""
    global EXTRA_BOND_FDIM
    EXTRA_BOND_FDIM = extra
###############################################

def get_bond_fdim(args: Namespace) -> int:
    """
    Gets the dimensionality of bond features.

    :param: Arguments.
    """
    return BOND_FDIM


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
           onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


class MolGraph:
    """
    A MolGraph represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:
    - smiles: Smiles string.
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    - a2b: A mapping from an atom index to a list of incoming bond indices.
    - b2a: A mapping from a bond index to the index of the atom the bond originates from.
    - b2revb: A mapping from a bond index to the index of the reverse bond.
    """

    def __init__(self, smiles: str, args: Namespace,
                 atom_features_extra=None,
                 bond_features_extra: np.ndarray = None):
        """
        Computes the graph structure and featurization of a molecule.

        :param smiles: A smiles string.
        :param args: Arguments.
        :param atom_features_extra: A list of 2D tensor containing additional atom features to featurize the molecule
        """
        self.smiles = smiles
        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond
        self.bonds = []
        #############my addition###############
        self.atom_features_extra = atom_features_extra
        self.overwrite_default_atom_features = args.overwrite_default_atom_features
        # self.extra_atom_fdim = args.extra_atom_fdim
        #######################################
        # # Convert smiles to molecule
        # mol = Chem.MolFromSmiles(smiles)
        ###################my addition########################
        if args.ligand_file_type == 'smiles':
            mol = Chem.MolFromSmiles(self.smiles)
        elif args.ligand_file_type == 'pdb':
            mol = Chem.MolFromPDBFile(args.datapaths + '/' + self.smiles)
        elif args.ligand_file_type == 'sdf':
            mol = Chem.SDMolSupplier(args.datapaths + '/' + self.smiles)[0]
        elif args.ligand_file_type == 'mol2':
            mol = Chem.MolFromMol2File(args.datapaths + '/' + self.smiles)[0]
        ######################################################

        # fake the number of "atoms" if we are collapsing substructures
        self.n_atoms = mol.GetNumAtoms()
        
        # # Get atom features
        # for i, atom in enumerate(mol.GetAtoms()):
        #     self.f_atoms.append(atom_features(atom))
        # self.f_atoms = [self.f_atoms[i] for i in range(self.n_atoms)]

        #############my addition###############
        #Get atom features
        self.f_atoms = [atom_features(atom) for atom in mol.GetAtoms()]
        if atom_features_extra is not None:
            if self.overwrite_default_atom_features:
                self.f_atoms = [descs.tolist() for descs in self.atom_features_extra]
            else:
                self.f_atoms = [f_atoms + descs.tolist() for f_atoms, descs in zip(self.f_atoms, self.atom_features_extra)]
        self.n_atoms = len(self.f_atoms)
        if self.atom_features_extra is not None and len(self.atom_features_extra) != self.n_atoms:
            raise ValueError(f'The number of atoms in {Chem.MolToSmiles(mol)} is different from the length of '
                             f'the extra atom features')
        #######################################

        for _ in range(self.n_atoms):
            self.a2b.append([])

        # Get bond features
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue

                f_bond = bond_features(bond)

                if args.atom_messages:
                    self.f_bonds.append(f_bond)
                    self.f_bonds.append(f_bond)
                else:
                    self.f_bonds.append(self.f_atoms[a1] + f_bond)
                    self.f_bonds.append(self.f_atoms[a2] + f_bond)

                # Update index mappings
                b1 = self.n_bonds
                b2 = b1 + 1
                self.a2b[a2].append(b1)  # b1 = a1 --> a2
                self.b2a.append(a1)
                self.a2b[a1].append(b2)  # b2 = a2 --> a1
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2
                self.bonds.append(np.array([a1, a2]))
        # rectify a2b
# =============================================================================
#         for ix in range(len(self.a2b)):
#             if len(self.a2b[ix]) <= 1:
#                 continue
#             if len(self.a2b[ix]) == 2:
#                 self.a2b[ix] = [self.a2b[ix][0], -1, self.a2b[ix][1]]
# =============================================================================
# =============================================================================
#         for ix in range(len(self.a2b)):
#             self.a2b[ix] = sorted(self.a2b[ix])
# =============================================================================

class BatchMolGraph:
    """
    A BatchMolGraph represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a MolGraph plus:
    - smiles_batch: A list of smiles strings.
    - n_mols: The number of molecules in the batch.
    - atom_fdim: The dimensionality of the atom features.
    - bond_fdim: The dimensionality of the bond features (technically the combined atom/bond features).
    - a_scope: A list of tuples indicating the start and end atom indices for each molecule.
    - b_scope: A list of tuples indicating the start and end bond indices for each molecule.
    - max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
    - b2b: (Optional) A mapping from a bond index to incoming bond indices.
    - a2a: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs: List[MolGraph], args: Namespace):
        self.smiles_batch = [mol_graph.smiles for mol_graph in mol_graphs]
        self.n_mols = len(self.smiles_batch)

        self.atom_fdim = get_atom_fdim(args)
        self.bond_fdim = get_bond_fdim(args) + (not args.atom_messages) * self.atom_fdim # * 2

        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]  # mapping from bond index to the index of the reverse bond
        bonds = [[0,0]]
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]]) #  if b!=-1 else 0

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])
                bonds.append([b2a[-1], 
                              self.n_atoms + mol_graph.b2a[mol_graph.b2revb[b]]])
            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds
        
        bonds = np.array(bonds).transpose(1,0)
        
        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b)) # max with 1 to fix a crash in rare case of all single-heavy-atom mols
        
        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor([a2b[a][:self.max_num_bonds] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.bonds = torch.LongTensor(bonds)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages

    def get_components(self) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                      torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                      List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns the components of the BatchMolGraph.

        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope, self.bonds

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """

        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incodming bond indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a


def mol2graph(smiles_batch: List[str],
              args: Namespace,
              atom_features_batch: List[np.ndarray] = None,
              bond_features_batch: List[np.ndarray] = None
              ) -> BatchMolGraph:
    """
    Converts a list of SMILES strings to a BatchMolGraph containing the batch of molecular graphs.

    :param smiles_batch: A list of SMILES strings.
    :param args: Arguments.
    :return: A BatchMolGraph containing the combined molecular graph for the molecules
    """
    mol_graphs = []
    ##############my addition###############
    if (atom_features_batch is not None) and (bond_features_batch is not None):
        for smiles, atom_features, bond_features in zip(smiles_batch, atom_features_batch, bond_features_batch):
            if smiles in SMILES_TO_GRAPH:
                mol_graph = SMILES_TO_GRAPH[smiles]
            else:
                mol_graph = MolGraph(smiles, args, atom_features, bond_features)
                if not args.no_cache:
                    SMILES_TO_GRAPH[smiles] = mol_graph
            mol_graphs.append(mol_graph)
    elif (atom_features_batch is not None) and (bond_features_batch is None):
        for smiles, atom_features in zip(smiles_batch, atom_features_batch):
            if smiles in SMILES_TO_GRAPH:
                mol_graph = SMILES_TO_GRAPH[smiles]
            else:
                mol_graph = MolGraph(smiles, args, atom_features,None)
                if not args.no_cache:
                    SMILES_TO_GRAPH[smiles] = mol_graph
            mol_graphs.append(mol_graph)

    elif (atom_features_batch is None) and (bond_features_batch is not None):
        for smiles, bond_features in zip(smiles_batch, bond_features_batch):
            if smiles in SMILES_TO_GRAPH:
                mol_graph = SMILES_TO_GRAPH[smiles]
            else:
                mol_graph = MolGraph(smiles, args, None, bond_features)
                if not args.no_cache:
                    SMILES_TO_GRAPH[smiles] = mol_graph
            mol_graphs.append(mol_graph)

    else:
        for smiles in smiles_batch:
            if smiles in SMILES_TO_GRAPH:
                mol_graph = SMILES_TO_GRAPH[smiles]
            else:
                mol_graph = MolGraph(smiles, args, None, None)
                if not args.no_cache:
                    SMILES_TO_GRAPH[smiles] = mol_graph
            mol_graphs.append(mol_graph)

    return BatchMolGraph(mol_graphs, args)
